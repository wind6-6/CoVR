"""
Copyright (c) 2023, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from src.model.blip2.blip2 import Blip2Base, disabled_train
from src.tools.utils import all_gather_with_grad, concat_all_gather


class ImageTextAttentionFusion(nn.Module):
    """图像-文本特征融合模块（注意力机制）"""
    def __init__(self, embed_dim: int, img_weight_init: float = 0.8):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 注意力融合层
        self.img_attn = nn.Linear(embed_dim, embed_dim)
        self.txt_attn = nn.Linear(embed_dim, embed_dim)
        self.fusion_attn = nn.Linear(embed_dim, 1)
        
        # 特征投影层
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # 初始化图像权重
        self.img_weight = nn.Parameter(torch.tensor(img_weight_init))
        self.txt_weight = nn.Parameter(torch.tensor(1.0 - img_weight_init))
        
        # 层归一化
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, img_feat: torch.Tensor) -> torch.Tensor:
        """
        输入图像特征，生成匹配的文本特征
        Args:
            img_feat: [B, D] 图像特征
        Returns:
            fused_feat: [B, D] 生成的文本特征
        """
        # 注意力加权
        img_att = torch.tanh(self.img_attn(img_feat))
        attn_weight = torch.sigmoid(self.fusion_attn(img_att))
        
        # 加权融合 + 投影
        img_feat_weighted = self.img_weight * img_feat
        fused = self.fusion_proj(img_feat_weighted)
        fused = self.ln(fused)
        
        # 归一化输出（匹配文本特征分布）
        fused_feat = F.normalize(fused, dim=-1)
        return fused_feat


class BLIP2CirImg2Txt(Blip2Base):
    """从tar_img_feat生成tar_txt_feat的BLIP2模型"""
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        loss: Any,
        vit_model="eva_clip_g",
        image_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp32",
        train_vit=False,
        vit="large",
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        temperature=0.07,
        si_ti_weight=1.0,
        si_tc_weight=0.2,
        fusion_img_weight_init=0.8,
    ):
        super().__init__()

        self.loss = loss
        self.tokenizer = self.init_tokenizer()
        self.temp = temperature
        self.max_txt_len = max_txt_len
        self.si_ti_weight = si_ti_weight
        self.si_tc_weight = si_tc_weight

        # 1. 初始化视觉编码器（冻结，复用预训练权重）
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, image_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        self.train_vit = train_vit
        if not train_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("Freeze vision encoder")

        # 2. 初始化Qformer（部分冻结，只训练核心层）
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        
        # 加载Qformer权重并初始化query参数
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])
        
        # 冻结Qformer非核心层
        for p in self.Qformer.cls.parameters():
            p.requires_grad = False
        self.query_tokens.requires_grad = False
        for name, param in self.Qformer.bert.encoder.named_parameters():
            if "crossattention" in name or "output_query" in name or "intermediate_query" in name:
                param.requires_grad = False

        # 3. 投影层（解冻，适配融合模块）
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        for p in self.vision_proj.parameters():
            p.requires_grad = True
        for p in self.text_proj.parameters():
            p.requires_grad = True

        # 4. 核心：图像特征转文本特征的融合模块
        self.img2txt_fusion = ImageTextAttentionFusion(
            embed_dim=embed_dim,
            img_weight_init=fusion_img_weight_init
        )

        # 校验损失权重
        assert si_ti_weight + si_tc_weight > 0, "No loss term is enabled"

    def forward(self, batch, fabric):
        """
        前向传播：输入tar_img_feat，生成tar_txt_feat并计算损失
        Args:
            batch: 包含tar_img_feat/tar_txt_feat的批次数据
            fabric: Fabric分布式训练对象
        Returns:
            total_loss: 融合损失（图像匹配+文本匹配）
        """
        device = next(self.parameters()).device
        # 1. 加载批次数据
        tar_img_feat = batch["tar_img_feat"].float().to(device)
        tar_txt_feat = batch["tar_txt_feat"].float().to(device)

        # 2. 分布式数据聚合
        tar_img_feat = concat_all_gather(tar_img_feat, fabric)
        tar_txt_feat = all_gather_with_grad(tar_txt_feat, fabric)

        tar_img_feat = tar_img_feat.mean(dim=1)

        # 4. 核心：从图像特征生成文本特征
        gen_txt_feat = self.img2txt_fusion(tar_img_feat)

        # 5. 损失计算
        total_loss = 0.0
        
        total_loss = self.loss(gen_txt_feat, tar_img_feat, self.temp)
        
        return total_loss


def blip2_cirimg2txt(model, ckpt_path, **kwargs):
    """模型加载函数（适配配置文件）"""
    if ckpt_path:
        model.load_from_pretrained(url_or_filename=ckpt_path)
    return model
