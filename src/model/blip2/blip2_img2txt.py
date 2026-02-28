import logging
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from src.model.blip2.blip2 import Blip2Base, disabled_train
from src.tools.utils import all_gather_with_grad, concat_all_gather


class BLIP2Cirimg2txt(Blip2Base):
    """
    优化版：复用BLIP2 Qformer跨模态能力，对齐save_blip2_embs_txts.py的特征生成逻辑
    通过tar_img_feat（图像特征）生成tar_txt_feat（文本特征），适配图像到文本特征的生成训练
    """
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        loss: Any,
        vit_model="eva_clip_g",  # 对齐save_blip2_embs_txts的vit模型
        image_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=True,  # 启用梯度检查点，适配大模型
        vit_precision="fp32",
        train_vit=False,
        vit="large",
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,  # 图像/文本特征维度对齐
        max_txt_len=32,
        temperature=0.07,  # 对齐代码库默认温度系数（关键！原1.0效果差）
        si_ti_weight=0,  # 关闭图像-图像损失，聚焦图像-文本
        si_tc_weight=1,  # 启用图像-文本损失
    ):
        super().__init__()

        self.loss = loss
        self.tokenizer = self.init_tokenizer()

        # 1. 初始化视觉编码器（完全对齐BLIP2Cir的配置）
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, image_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        self.train_vit = train_vit
        if not train_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")

        # 2. 初始化Qformer（复用BLIP2跨模态桥接能力，关键！）
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        # 3. 特征投影层（对齐BLIP2Cir的text_proj设计，而非简单的Sequential）
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        
        # 4. 冻结非训练层（对齐代码库中BLIP2Cir的参数冻结逻辑）
        for p in self.vision_proj.parameters():
            p.requires_grad = False
        for p in self.ln_vision.parameters():
            p.requires_grad = False
        for p in self.Qformer.cls.parameters():
            p.requires_grad = False

        # 关键超参数（对齐save_blip2_embs_txts的配置）
        self.temp = temperature
        self.max_txt_len = max_txt_len
        self.embed_dim = embed_dim

        # 损失权重（聚焦图像→文本的对比损失）
        assert si_ti_weight + si_tc_weight > 0, "No loss term is enabled"
        self.si_ti_weight = si_ti_weight
        self.si_tc_weight = si_tc_weight

    def forward(self, batch, fabric):
        # 1. 读取输入（兼容原batch结构，同时复用BLIP2的视觉编码逻辑）
        tar_img_feat = batch["tar_img_feat"]  # 输入图像特征
        tar_txt_feat = batch["tar_txt_feat"]  # 目标文本特征
        device = tar_img_feat.device

        # 2. 特征预处理（对齐BLIP2Cir的精度和分布式处理）
        tar_img_feat = tar_img_feat.float().to(device)
        tar_txt_feat = tar_txt_feat.float().to(device)

        # 3. 分布式特征聚合（保持原逻辑，但补充Qformer的视觉特征编码）
        # （可选：如果tar_img_feat是原始图像的embedding，可跳过Qformer编码；如果是原始图像，需走visual_encoder）
        if len(tar_img_feat.shape) == 4:  # 如果输入是原始图像（[B, C, H, W]），走BLIP2视觉编码
            with torch.no_grad() if not self.train_vit else autocast():
                tar_img_embs = self.ln_vision(self.visual_encoder(tar_img_feat))
                image_atts = torch.ones(tar_img_embs.size()[:-1], dtype=torch.long).to(device)
                query_tokens = self.query_tokens.expand(tar_img_embs.shape[0], -1, -1)
                
                # Qformer编码（跨模态特征增强）
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=tar_img_embs,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
                tar_img_feat = query_output.last_hidden_state  # [B, 32, 768]

        # 4. 特征聚合与投影（对齐BLIP2Cir的mean+归一化逻辑）
        tar_img_feat = concat_all_gather(tar_img_feat, fabric)
        tar_txt_feat = concat_all_gather(tar_txt_feat, fabric)

        # 对query token取均值（对齐代码库中BLIP2Cir的处理）
        tar_img_feat = tar_img_feat.mean(dim=1)  # [B*N, 768]
        tar_txt_feat_pred = self.text_proj(tar_img_feat)  # 图像特征→文本特征空间

        # 5. 归一化（保证对比损失有效性，必须！）
        tar_txt_feat_pred = F.normalize(tar_txt_feat_pred, dim=-1)
        tar_txt_feat = F.normalize(tar_txt_feat, dim=-1)

        # 6. 损失计算（对齐BLIP2Cir的多权重损失逻辑）
        loss = 0
        if self.si_tc_weight > 0:
            si_tc_loss = self.loss(tar_txt_feat_pred, tar_txt_feat, self.temp)
            loss += si_tc_loss * self.si_tc_weight
        if self.si_ti_weight > 0:
            si_ti_loss = self.loss(tar_txt_feat_pred, tar_img_feat.mean(dim=1), self.temp)
            loss += si_ti_loss * self.si_ti_weight

        return loss


def blip2_cirimg2txt(model, ckpt_path, **kwargs):
    if ckpt_path:
        # 加载预训练权重（关键：使用save_blip2_embs_txts生成的ckpt）
        model.load_from_pretrained(url_or_filename=ckpt_path)
    return model
