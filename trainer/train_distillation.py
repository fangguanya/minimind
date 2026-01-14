"""
================================================================================
                    MiniMind 知识蒸馏训练脚本 (Knowledge Distillation)
================================================================================

【什么是知识蒸馏】
知识蒸馏是一种模型压缩技术，让小模型 (学生) 学习大模型 (教师) 的知识:
- 教师模型: 大型、高性能的模型 (如 768维/16层)
- 学生模型: 小型、高效的模型 (如 512维/8层)
- 目标: 让学生模型获得接近教师模型的性能

【核心思想】
教师模型的"软标签" (soft labels) 包含比硬标签更多的信息:
- 硬标签: [0, 0, 1, 0, 0] (只有正确答案)
- 软标签: [0.01, 0.05, 0.8, 0.1, 0.04] (包含类别间的相似度信息)

例如: "猫"和"狗"的相似度比"猫"和"汽车"更高
软标签能传递这种知识，而硬标签不能

【蒸馏损失函数】
L_total = α * L_CE + (1-α) * L_KL

其中:
- L_CE: 交叉熵损失 (学生 vs 真实标签)
  L_CE = -Σ y_true * log(p_student)

- L_KL: KL 散度损失 (学生 vs 教师)
  L_KL = T² * KL(softmax(z_teacher/T) || softmax(z_student/T))

# ════════════════════════════════════════════════════════════════════════════════
# 【什么是 KL 散度 (Kullback-Leibler Divergence)？】
# ════════════════════════════════════════════════════════════════════════════════
#
# 【一句话定义】
# KL 散度衡量两个概率分布 P 和 Q 之间的"差异程度"
# 可以理解为: 用 Q 来近似 P 时，损失了多少信息
#
# ────────────────────────────────────────────────────────────────────────────────
# 【数学公式】
# ────────────────────────────────────────────────────────────────────────────────
#
#   KL(P || Q) = Σ P(x) × log(P(x) / Q(x))
#              = Σ P(x) × log(P(x)) - Σ P(x) × log(Q(x))
#              = -H(P) + H(P,Q)
#              = 交叉熵(P,Q) - 熵(P)
#
# 其中:
#   P: 真实分布 (这里是教师模型的输出)
#   Q: 近似分布 (这里是学生模型的输出)
#   H(P): P 的熵 (P 自身的不确定性)
#   H(P,Q): P 和 Q 的交叉熵
#
# 【直觉理解】
# - KL(P||Q) = 0 当且仅当 P = Q (完美匹配)
# - KL(P||Q) > 0 总是成立 (非负性)
# - KL(P||Q) ≠ KL(Q||P) (不对称！)
#
# ────────────────────────────────────────────────────────────────────────────────
# 【为什么 KL 散度不对称？】
# ────────────────────────────────────────────────────────────────────────────────
#
# 假设 P = [0.9, 0.1], Q = [0.5, 0.5]
#
# KL(P||Q) = 0.9×log(0.9/0.5) + 0.1×log(0.1/0.5)
#          = 0.9×0.59 + 0.1×(-1.61) = 0.37
#
# KL(Q||P) = 0.5×log(0.5/0.9) + 0.5×log(0.5/0.1)
#          = 0.5×(-0.59) + 0.5×1.61 = 0.51
#
# 结果不同！这就是为什么写成 KL(P||Q) 而不是 KL(P,Q)
#
# 【在蒸馏中的含义】
# KL(teacher || student): 
#   衡量学生"覆盖"教师知识的程度
#   如果教师认为某词重要(概率高)，学生也必须给高概率
#
# ────────────────────────────────────────────────────────────────────────────────
# 【与交叉熵的关系】
# ────────────────────────────────────────────────────────────────────────────────
#
# 交叉熵: H(P,Q) = -Σ P(x) × log(Q(x))
# KL散度: KL(P||Q) = H(P,Q) - H(P)
#
# 因为 H(P) 是常数(教师输出固定)，所以:
#   最小化 KL(P||Q) ≡ 最小化 H(P,Q)
#
# 那为什么还要用 KL 而不是交叉熵？
# 1. KL 的值域是 [0, ∞)，0 表示完美匹配，更直观
# 2. KL 可以分解为"信息损失"，更有解释性
# 3. 在某些场景下梯度行为更好
#
# ────────────────────────────────────────────────────────────────────────────────
# 【工程上的作用】
# ────────────────────────────────────────────────────────────────────────────────
#
# 1.【知识蒸馏】让学生模型模仿教师的概率分布
#    - 不是只学"对错"(硬标签)
#    - 而是学"程度"(软标签): 猫和狗的相似度 > 猫和汽车
#
# 2.【分布匹配】让生成模型的输出分布接近目标分布
#    - VAE 中用 KL 约束隐变量分布接近正态分布
#    - GAN 的变体用 KL 衡量生成分布和真实分布的差异
#
# 3.【正则化】防止模型输出过于尖锐
#    - 添加 KL(uniform || model) 让输出更平滑
#    - 类似 label smoothing 的效果
#
# ────────────────────────────────────────────────────────────────────────────────
# 【实际效果 (蒸馏场景)】
# ────────────────────────────────────────────────────────────────────────────────
#
# 【实验数据】(典型场景)
#   教师: 768维/16层 (~104M参数), 准确率 85%
#   学生: 512维/8层 (~26M参数)
#
#   | 训练方式           | 学生准确率 | 相对教师 |
#   |-------------------|-----------|---------|
#   | 只用硬标签 (CE)    | 78%       | 92%     |
#   | 只用软标签 (KL)    | 80%       | 94%     |
#   | 混合 (CE + KL)    | 82%       | 96%     |
#
# 【为什么 KL 蒸馏更好？】
# 1. 软标签包含"暗知识" (dark knowledge)
#    - 硬标签: "这是猫" (只有结论)
#    - 软标签: "这是猫(0.8)，可能是狗(0.15)，不像汽车(0.001)"
#              包含了类别间的相似性关系
#
# 2. 梯度更平滑
#    - 硬标签: 只有正确类有梯度
#    - 软标签: 所有类都有梯度，训练更稳定
#
# 3. 正则化效果
#    - 教师模型的软标签自带平滑
#    - 防止学生过拟合到训练集的噪声
#
# ════════════════════════════════════════════════════════════════════════════════
  
  其中 T 是温度参数:
  - T=1: 标准 softmax
  - T>1: 更平滑的分布，突出类别间关系
  - T² 是因为梯度会被 T 缩小，需要补偿

- α: 权衡系数，控制两种损失的比例

【温度 (Temperature) 的作用】
温度越高，softmax 输出越平滑:
- T=1: softmax([3,1,0]) ≈ [0.84, 0.11, 0.04] (尖锐)
- T=2: softmax([3,1,0]/2) ≈ [0.61, 0.24, 0.14] (平滑)
- T=∞: softmax([3,1,0]/∞) ≈ [0.33, 0.33, 0.33] (均匀)

平滑的分布包含更多关于类别相似性的信息

【与预训练/微调的区别】
- 预训练: 从头学习语言规律
- 微调: 在特定任务上调整
- 蒸馏: 从大模型学习知识，压缩到小模型

【使用方法】
python train_distillation.py --student_hidden_size 512 --teacher_hidden_size 768

【典型设置】
- 教师: hidden_size=768, num_layers=16 (~104M 参数)
- 学生: hidden_size=512, num_layers=8 (~26M 参数)
- alpha=0.5: CE 和 KL 各占一半
- temperature=1.5: 略微平滑的软标签
"""

import os
import sys

# 设置包名，确保相对导入正常工作
__package__ = "trainer"
# 将父目录添加到路径，以便导入其他模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

# 忽略不重要的警告信息
warnings.filterwarnings('ignore')


def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    """
    计算知识蒸馏损失 (KL 散度)
    
    【数学公式】
    L_KL = T² * KL(P_teacher || P_student)
         = T² * Σ P_teacher * log(P_teacher / P_student)
         = T² * Σ P_teacher * (log P_teacher - log P_student)
    
    其中:
    - P_teacher = softmax(z_teacher / T)  # 教师的软标签
    - P_student = softmax(z_student / T)  # 学生的预测
    - T: 温度参数
    - T²: 梯度补偿因子
    
    【温度的作用】
    高温使 softmax 输出更平滑:
    - 突出教师模型对不同类别的相对偏好
    - 让学生更容易学习类别间的关系
    - 常用值: 1.0 ~ 4.0
    
    【为什么乘以 T²】
    当温度 T > 1 时，softmax 的梯度被缩小了 T 倍
    为了保持梯度量级不变，需要乘以 T²
    (证明涉及 softmax 导数的链式法则)
    
    【参数】
    - student_logits: 学生模型输出 [batch*seq_len, vocab_size]
    - teacher_logits: 教师模型输出 [batch*seq_len, vocab_size]
    - temperature: 蒸馏温度，越大分布越平滑
    - reduction: 损失归约方式
    
    【返回】
    - kl_loss: 标量损失值
    """
    # 教师模型的软标签 (不计算梯度)
    # softmax(logits / T) 使分布更平滑
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    # 学生模型的 log 概率
    # 使用 log_softmax 比 log(softmax) 数值更稳定
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # 计算 KL 散度
    # F.kl_div 期望输入是 log_probs，目标是 probs
    kl = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction=reduction  # 'batchmean': 除以 batch size
    )
    
    # 乘以 T² 补偿温度对梯度的缩放
    return (temperature ** 2) * kl


def train_epoch(epoch, loader, iters, teacher_model, lm_config_student, start_step=0, wandb=None, alpha=0.0, temperature=1.0):
    """
    训练一个 epoch
    
    【训练流程】
    for each batch:
        1. 学生前向传播: student_logits = student(X)
        2. 教师前向传播: teacher_logits = teacher(X) [无梯度]
        3. 计算 CE 损失: L_CE = CrossEntropy(student_logits, Y)
        4. 计算蒸馏损失: L_KL = KL(teacher_probs || student_probs)
        5. 总损失: L = α * L_CE + (1-α) * L_KL
        6. 反向传播: L.backward()
        7. 参数更新: optimizer.step()
    
    # ════════════════════════════════════════════════════════════════════════════
    # 【蒸馏 vs 用教师Q/A做SFT，核心区别是什么？】
    # ════════════════════════════════════════════════════════════════════════════
    #
    # 【方法对比】
    # ┌─────────────────┬─────────────────────────────────────────────────────────┐
    # │                 │  SFT教师输出              │  知识蒸馏 (KD)              │
    # ├─────────────────┼───────────────────────────┼─────────────────────────────┤
    # │ 监督信号        │ 硬标签 (argmax)           │ 软标签 (完整概率分布)        │
    # │ 信息量          │ 1 个数 (token ID)         │ vocab_size 个数 (概率)       │
    # │ 损失函数        │ CrossEntropy              │ KL Divergence               │
    # │ 教师参与        │ 离线生成一次              │ 在线每步都参与               │
    # └─────────────────┴───────────────────────────┴─────────────────────────────┘
    #
    # ────────────────────────────────────────────────────────────────────────────
    # 【数学原理区别】
    # ────────────────────────────────────────────────────────────────────────────
    #
    # 【SFT 教师输出】
    # 教师生成: "今天天气很好" → 提取 token IDs: [102, 345, 678, 901]
    # 学生学习: L = -log P_student(102) - log P_student(345) - ...
    #
    # 学生只知道: "下一个词是 102"
    # 学生不知道: "103 也挺像，但 999 完全不像"
    #
    # 【知识蒸馏】
    # 教师输出: P_teacher = [0.001, 0.7, 0.2, 0.05, 0.001, ...]  # 完整分布
    # 学生学习: L = KL(P_teacher || P_student)
    #            = Σ P_teacher(i) × log(P_teacher(i) / P_student(i))
    #
    # 学生知道: "102 概率 0.7，103 概率 0.2，999 概率 0.001"
    # 学生学到: 词与词之间的相似性关系！
    #
    # ────────────────────────────────────────────────────────────────────────────
    # 【具体例子】预测 "我喜欢吃___"
    # ────────────────────────────────────────────────────────────────────────────
    #
    # 教师模型输出分布:
    #   苹果: 0.35  ← 最高
    #   香蕉: 0.25
    #   橙子: 0.20
    #   西瓜: 0.15
    #   汽车: 0.001
    #   电脑: 0.0005
    #   ...
    #
    # 【SFT 方式】
    # 提取硬标签: "苹果" (argmax)
    # 学生只学到: "答案是苹果"
    # 梯度只在 "苹果" 位置有值，其他位置梯度=0
    #
    # 【蒸馏方式】
    # 使用软标签: [0.35, 0.25, 0.20, 0.15, 0.001, ...]
    # 学生学到:
    #   - "苹果、香蕉、橙子、西瓜" 都是合理答案 (都是水果)
    #   - "香蕉" 比 "汽车" 更像正确答案
    #   - "汽车、电脑" 完全不合理 (概率极低)
    #
    # 这就是 "暗知识" (Dark Knowledge)！
    #
    # ────────────────────────────────────────────────────────────────────────────
    # 【梯度信号对比】
    # ────────────────────────────────────────────────────────────────────────────
    #
    # 假设 vocab_size = 6400，学生预测分布 Q，教师分布 P
    #
    # 【SFT 的梯度】(只有 1 个位置有梯度)
    #   ∂L/∂z_苹果 = Q(苹果) - 1  # 只有正确答案
    #   ∂L/∂z_香蕉 = Q(香蕉) - 0  # = Q(香蕉)，但乘以 0 没有方向指导
    #   ∂L/∂z_其他 = Q(其他) - 0
    #
    # 【蒸馏的梯度】(所有位置都有梯度)
    #   ∂L/∂z_苹果 = Q(苹果) - P(苹果) = Q - 0.35
    #   ∂L/∂z_香蕉 = Q(香蕉) - P(香蕉) = Q - 0.25  ← 有方向！
    #   ∂L/∂z_橙子 = Q(橙子) - P(橙子) = Q - 0.20  ← 有方向！
    #   ∂L/∂z_汽车 = Q(汽车) - P(汽车) = Q - 0.001 ← 压低它！
    #
    # 蒸馏给每个词都提供了 "应该往哪个方向调" 的信号！
    #
    # ────────────────────────────────────────────────────────────────────────────
    # 【信息论视角】
    # ────────────────────────────────────────────────────────────────────────────
    #
    # 【SFT 传递的信息量】
    #   每个位置: log2(vocab_size) ≈ log2(6400) ≈ 12.6 bits
    #   只告诉你 "是哪个词"
    #
    # 【蒸馏传递的信息量】
    #   每个位置: vocab_size × 32bit = 6400 × 32 ≈ 200KB
    #   告诉你 "每个词的概率是多少"
    #
    # 信息量差了 ~16000 倍！
    #
    # ────────────────────────────────────────────────────────────────────────────
    # 【实际效果对比】
    # ────────────────────────────────────────────────────────────────────────────
    #
    # 场景: 768维教师 → 512维学生
    #
    # | 方法                    | 学生性能 | 相对教师 | 特点              |
    # |------------------------|---------|---------|-------------------|
    # | SFT 教师 Q/A           | 78%     | 92%     | 简单，离线即可     |
    # | 蒸馏 (KL)              | 82%     | 96%     | 需要教师在线推理   |
    # | 蒸馏 + SFT (CE+KL)     | 83%     | 98%     | 最佳效果          |
    #
    # 【蒸馏额外的计算成本】
    # - 每个 batch 需要教师前向传播 (无需反向)
    # - 显存: 需要同时加载教师和学生
    # - 时间: 约增加 30-50%
    #
    # 【什么时候用 SFT 教师输出？】
    # - 教师模型太大，无法在线推理
    # - 对性能要求不高，追求简单
    # - 只有教师生成的文本，没有模型权重
    #
    # 【什么时候用蒸馏？】
    # - 追求最佳性能
    # - 有足够的计算资源
    # - 有教师模型权重
    # ════════════════════════════════════════════════════════════════════════════
    
    【参数说明】
    - epoch: 当前 epoch 编号
    - loader: 数据加载器
    - iters: 每个 epoch 的迭代次数
    - teacher_model: 教师模型 (冻结，不训练)
    - lm_config_student: 学生模型配置
    - start_step: 起始步数 (用于断点续训)
    - wandb: 实验追踪工具
    - alpha: CE 损失权重，L = α*CE + (1-α)*KL
    - temperature: 蒸馏温度
    
    【alpha 的选择】
    - alpha=1.0: 纯 CE 损失，不使用蒸馏
    - alpha=0.0: 纯蒸馏损失，不使用真实标签
    - alpha=0.5: CE 和蒸馏各占一半 (推荐)
    
    【temperature 的选择】
    - T=1.0: 标准 softmax
    - T=1.5~2.0: 轻微平滑，常用设置
    - T>3.0: 过于平滑，可能丢失信息
    """
    start_time = time.time()
    
    # 确保教师模型在评估模式且不计算梯度
    # 教师只提供知识，不需要更新
    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.requires_grad_(False)

    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        # 将数据移到指定设备 (CPU/GPU)
        X = X.to(args.device)           # 输入 [batch, seq_len-1]
        Y = Y.to(args.device)           # 标签 [batch, seq_len-1]
        loss_mask = loss_mask.to(args.device)  # 损失掩码 [batch, seq_len-1]
        
        # 动态调整学习率 (余弦调度)
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ==================== 前向传播 ====================
        
        # 学生模型前向传播 (需要计算梯度)
        with autocast_ctx:  # 混合精度上下文
            res = model(X)
            student_logits = res.logits  # [batch, seq_len, vocab_size]

        # 教师模型前向传播 (无梯度，只提供软标签)
        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(X).logits
                # 如果学生词表比教师小，截断教师输出
                # (通常两者词表相同，这是一个安全检查)
                vocab_size_student = student_logits.size(-1)
                teacher_logits = teacher_logits[..., :vocab_size_student]

        # ==================== 计算损失 ====================
        
        # 1) Ground-Truth CE Loss (学生 vs 真实标签)
        # 使用交叉熵损失，只在非 padding 位置计算
        loss_mask_flat = loss_mask.view(-1)  # 展平为 [batch*seq_len]
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),  # [B*L, vocab_size]
            Y.view(-1),                                         # [B*L]
            ignore_index=0,  # 忽略 padding token (ID=0)
            reduction='none'  # 返回每个位置的损失
        )
        # 应用损失掩码，只计算有效位置的平均损失
        ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / loss_mask_flat.sum()
        
        # 如果使用 MoE，添加辅助损失 (负载均衡)
        if lm_config_student.use_moe:
            ce_loss = ce_loss_raw + res.aux_loss
        else:
            ce_loss = ce_loss_raw

        # 2) Distillation Loss (学生 vs 教师)
        if teacher_model is not None:
            # 只在有效位置计算蒸馏损失
            distill_loss = distillation_loss(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],
                temperature=temperature
            )
        else:
            # 如果没有教师模型，蒸馏损失为 0
            distill_loss = torch.tensor(0.0, device=args.device)

        # 3) 总损失 = α * CE + (1-α) * Distill
        # 除以梯度累积步数，因为梯度会累积
        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps

        # ==================== 反向传播和参数更新 ====================
        
        # 使用 GradScaler 处理混合精度的梯度缩放
        scaler.scale(loss).backward()

        # 每累积 N 步后更新参数
        if (step + 1) % args.accumulation_steps == 0:
            # 反缩放梯度，恢复原始值
            scaler.unscale_(optimizer)
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 优化器更新参数
            scaler.step(optimizer)
            # 更新 scaler 的缩放因子
            scaler.update()
            # 清零梯度，准备下一次累积
            optimizer.zero_grad(set_to_none=True)

        # ==================== 日志记录 ====================
        
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_ce_loss = ce_loss_raw.item()
            current_aux_loss = res.aux_loss.item() if lm_config_student.use_moe else 0.0
            current_lr = optimizer.param_groups[-1]['lr']
            # 计算剩余时间 (分钟)
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, ce: {current_ce_loss:.4f}, aux_loss: {current_aux_loss:.4f}, distill: {distill_loss.item():.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')
            
            # 记录到 wandb/swanlab
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "ce_loss": current_ce_loss,
                    "aux_loss": current_aux_loss,
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        # ==================== 保存检查点 ====================
        
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()  # 切换到评估模式
            moe_suffix = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config_student.hidden_size}{moe_suffix}.pth'
            
            # 处理 DDP 包装的模型
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            # 转为半精度节省空间，移到 CPU 避免 GPU 内存问题
            state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            
            # 保存完整检查点 (包含优化器状态等，用于断点续训)
            lm_checkpoint(lm_config_student, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            
            model.train()  # 切换回训练模式
            del state_dict

        # 及时释放内存，避免 OOM
        del X, Y, loss_mask, res, student_logits, teacher_logits, ce_loss, distill_loss, loss


if __name__ == "__main__":
    # ==================== 参数解析 ====================
    parser = argparse.ArgumentParser(description="MiniMind Knowledge Distillation")
    
    # 保存配置
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_dist', type=str, help="保存权重的前缀名")
    
    # 训练配置
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--max_seq_len", type=int, default=340, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl", help="训练数据路径")
    
    # 学生模型配置
    parser.add_argument('--student_hidden_size', default=512, type=int, help="学生模型隐藏层维度")
    parser.add_argument('--student_num_layers', default=8, type=int, help="学生模型隐藏层数量")
    
    # 教师模型配置
    parser.add_argument('--teacher_hidden_size', default=768, type=int, help="教师模型隐藏层维度")
    parser.add_argument('--teacher_num_layers', default=16, type=int, help="教师模型隐藏层数量")
    
    # MoE 配置
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    
    # 权重加载
    parser.add_argument('--from_student_weight', default='full_sft', type=str, help="学生模型基于哪个权重")
    parser.add_argument('--from_teacher_weight', default='full_sft', type=str, help="教师模型基于哪个权重")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    
    # 蒸馏超参数
    parser.add_argument('--alpha', default=0.5, type=float, help="CE损失权重，总损失=alpha*CE+(1-alpha)*KL")
    parser.add_argument('--temperature', default=1.5, type=float, help="蒸馏温度（推荐范围1.0-2.0）")
    
    # 实验追踪
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Distillation", help="wandb项目名")
    
    args = parser.parse_args()

    # ==================== 1. 初始化环境和随机种子 ====================
    # 初始化分布式训练 (如果使用多 GPU)
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    
    # 设置随机种子，确保可复现性
    # 不同 rank 使用不同种子避免数据重复
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ==================== 2. 配置目录、模型参数、检查ckp ====================
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 学生模型配置
    lm_config_student = MiniMindConfig(
        hidden_size=args.student_hidden_size,
        num_hidden_layers=args.student_num_layers,
        use_moe=bool(args.use_moe)
    )
    
    # 教师模型配置
    lm_config_teacher = MiniMindConfig(
        hidden_size=args.teacher_hidden_size,
        num_hidden_layers=args.teacher_num_layers,
        use_moe=bool(args.use_moe)
    )
    
    # 检查是否有断点续训的检查点
    ckp_data = lm_checkpoint(lm_config_student, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ==================== 3. 设置混合精度 ====================
    # 混合精度训练可以加速训练并减少显存使用
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # CPU 不支持混合精度，使用空上下文
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ==================== 4. 配wandb ====================
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Distill-S{args.student_hidden_size}T{args.teacher_hidden_size}-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ==================== 5. 定义学生和教师模型 ====================
    # 初始化学生模型 (需要训练)
    model, tokenizer = init_model(lm_config_student, args.from_student_weight, device=args.device)
    Logger(f'学生模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    
    # 初始化教师模型 (冻结，只用于推理)
    teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, device=args.device)
    teacher_model.eval()  # 评估模式
    teacher_model.requires_grad_(False)  # 冻结参数
    Logger(f'教师模型总参数量：{sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f} M')
    
    # 加载数据集
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # 初始化 GradScaler (用于混合精度训练)
    # float16 需要梯度缩放防止下溢，bfloat16 不需要
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # 初始化优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ==================== 6. 从ckp恢复状态 ====================
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ==================== 7. DDP包模型 ====================
    # 分布式数据并行，多 GPU 训练
    if dist.is_initialized():
        # 忽略 RoPE 的 buffer，它们不需要同步
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ==================== 8. 开始训练 ====================
    for epoch in range(start_epoch, args.epochs):
        # 设置分布式采样器的 epoch，确保每个 epoch 数据打乱方式不同
        train_sampler and train_sampler.set_epoch(epoch)
        
        if epoch == start_epoch and start_step > 0:
            # 第一个 epoch 且存在检查点，需要跳过已训练的步数
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + start_step + 1, teacher_model, lm_config_student, start_step, wandb, args.alpha, args.temperature)
        else:
            # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(epoch, loader, len(loader), teacher_model, lm_config_student, 0, wandb, args.alpha, args.temperature)
    
    # ==================== 9. 清理分布进程 ====================
    if dist.is_initialized():
        dist.destroy_process_group()
