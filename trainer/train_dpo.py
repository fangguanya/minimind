"""
================================================================================
                    MiniMind DPO 训练脚本 (Direct Preference Optimization)
================================================================================

【什么是 DPO】
DPO (直接偏好优化) 是一种让模型"懂得分辨好坏"的训练方法:
- 给模型看两个回复: 一个好的 (chosen), 一个差的 (rejected)
- 让模型增加生成好回复的概率，减少生成差回复的概率
- 这让模型的输出更符合人类偏好

【形象比喻】
想象你在教一个学生写作:
- 你给他看两篇作文: A 篇写得好，B 篇写得差
- 学生通过对比学习，理解什么是好的写作
- DPO 就是这个过程的数学表达

【为什么需要 DPO】
预训练后的模型只学会了"像人类说话"，但不知道"说什么是对的":
- 可能生成有害内容
- 可能回答错误
- 可能风格不好

DPO 让模型学会人类的偏好，输出更安全、更有帮助的回复。

【DPO vs RLHF/PPO】
传统 RLHF 流程复杂:
1. 收集人类偏好数据
2. 训练奖励模型 (Reward Model)
3. 用 PPO 优化策略

DPO 简化了这个流程:
1. 收集人类偏好数据
2. 直接优化策略 (不需要奖励模型!)

【DPO 损失函数的直观理解】

设想:
- π(y|x): 策略模型生成回复 y 的概率
- π_ref(y|x): 参考模型 (训练前的模型) 生成 y 的概率

DPO 的目标:
- 让 π(chosen)/π_ref(chosen) 变大 (相对于参考模型，提高好回复的概率)
- 让 π(rejected)/π_ref(rejected) 变小 (相对于参考模型，降低差回复的概率)

损失函数:
L_DPO = -log σ(β * (log(π/π_ref)(chosen) - log(π/π_ref)(rejected)))

其中:
- σ 是 sigmoid 函数
- β 是温度参数 (越大越强调差异)

【为什么要除以 π_ref】
直接优化 π(chosen)/π(rejected) 可能导致模型"走极端":
- 可能让所有概率都降到 0，只有 chosen 很高
- 这会让模型"忘记"之前学到的知识

除以 π_ref 起到"正则化"作用:
- 只要求模型比参考模型"更好一点"
- 不要求模型完全改变
- 保持语言能力的同时学习偏好

【β (beta) 参数的作用】
β 控制偏好学习的强度:
- β 小 (如 0.05): 学习轻微，模型变化小
- β 大 (如 0.5): 学习强烈，模型变化大
- 推荐值: 0.1

【数据格式】
{
    "chosen": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "好回复"}],
    "rejected": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "差回复"}]
}

【使用方法】
python train_dpo.py --epochs 1 --batch_size 4 --learning_rate 4e-8 --beta 0.1
"""

import os
import sys

# ==================== 模块路径设置 ====================
# __package__ 设置确保相对导入正常工作
# 例如: from .trainer_utils import ... 能正确找到模块
__package__ = "trainer"

# 将父目录添加到 Python 路径
# 这样可以 import model.xxx, dataset.xxx 等顶层模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ==================== 导入必要的库 ====================
import argparse              # 命令行参数解析
import time                  # 时间测量
import warnings              # 警告控制
import torch                 # PyTorch 核心
import torch.nn.functional as F  # 函数式接口 (softmax, log_softmax 等)
import torch.distributed as dist # 分布式训练支持
from contextlib import nullcontext  # 空上下文管理器
from torch import optim      # 优化器 (AdamW 等)
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载

# 导入自定义模块
from model.model_minimind import MiniMindConfig  # 模型配置
from dataset.lm_dataset import DPODataset        # DPO 数据集
from trainer.trainer_utils import (
    get_lr,                   # 获取当前学习率 (余弦调度)
    Logger,                   # 日志打印工具
    is_main_process,          # 判断是否是主进程
    lm_checkpoint,            # 检查点管理
    init_distributed_mode,    # 初始化分布式训练
    setup_seed,               # 设置随机种子
    init_model,               # 初始化模型
    SkipBatchSampler          # 断点续训时跳过已训练的 batch
)

# 忽略不重要的警告信息
warnings.filterwarnings('ignore')


def logits_to_log_probs(logits, labels):
    """
    将模型输出的 logits 转换为目标 token 的 log 概率
    
    【为什么需要这个函数】
    DPO 需要计算模型生成某个具体回复的概率。
    但模型输出的是 logits (未归一化的分数)，不是概率。
    我们需要:
    1. 将 logits 转换为概率分布 (softmax)
    2. 取 log 得到 log 概率 (数值稳定)
    3. 提取目标 token 对应的 log 概率
    
    【数学过程】
    logits: [batch, seq_len, vocab_size]
    ↓ log_softmax (按最后一维)
    log_probs: [batch, seq_len, vocab_size]  # 每个位置的词表概率分布
    ↓ gather (提取 labels 对应的概率)
    log_probs_per_token: [batch, seq_len]    # 只保留目标 token 的 log 概率
    
    【为什么用 log 概率】
    1. 数值稳定: 概率乘法变成 log 加法，避免下溢
    2. 梯度友好: log 的梯度更平滑
    
    【参数】
    - logits: 模型输出的原始分数
      形状: [batch_size, sequence_length, vocab_size]
      例如: [4, 512, 6400] 表示 4 个样本，每个 512 个位置，词表 6400
    
    - labels: 目标 token 的 ID
      形状: [batch_size, sequence_length]
      例如: [[1, 234, 567, ...], [...]] 每个数是词表中的 ID
    
    【返回】
    - log_probs_per_token: 每个位置目标 token 的 log 概率
      形状: [batch_size, sequence_length]
      
    【示例】
    假设词表大小 V=5, 序列长度 L=3:
    logits = [[[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [1, 1, 1, 1, 1]]]  # [1, 3, 5]
    labels = [[4, 3, 2]]  # 我们要取的 token ID
    
    log_softmax(logits) 后每行和为 0 (概率和为 1 的 log)
    然后 gather 提取 labels 对应位置的值
    返回: [[log_prob[4], log_prob[3], log_prob[2]]]  # [1, 3]
    """
    # 1. 计算 log softmax
    # log_softmax = log(softmax(logits))
    # 比先 softmax 再 log 更数值稳定
    # dim=2 表示在词表维度上做 softmax
    log_probs = F.log_softmax(logits, dim=2)  # [B, L, V]
    
    # 2. 使用 gather 提取目标 token 的 log 概率
    # gather 按照 labels 的索引从 log_probs 中取值
    # labels.unsqueeze(2): [B, L] → [B, L, 1]
    # gather 在 dim=2 上按索引取值
    # 结果: [B, L, 1] → squeeze → [B, L]
    log_probs_per_token = torch.gather(
        log_probs,           # 输入张量 [B, L, V]
        dim=2,               # 在第 2 维 (词表维度) 上取值
        index=labels.unsqueeze(2)  # 索引 [B, L, 1]
    ).squeeze(-1)  # 移除最后一维 [B, L, 1] → [B, L]
    
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """
    计算 DPO (Direct Preference Optimization) 损失
    
    【DPO 损失函数完整推导】
    
    原始目标: 让模型更偏好 chosen 而非 rejected
    
    定义隐式奖励:
    r(x, y) = β * log(π(y|x) / π_ref(y|x)) + β * log Z(x)
    
    其中 Z(x) 是配分函数 (归一化常数)
    
    Bradley-Terry 偏好模型:
    P(y_w > y_l | x) = σ(r(y_w) - r(y_l))
    
    代入隐式奖励:
    P(y_w > y_l) = σ(β * (log π(y_w)/π_ref(y_w) - log π(y_l)/π_ref(y_l)))
    
    最大化这个概率等价于最小化负对数:
    L = -log σ(β * (log π(y_w)/π_ref(y_w) - log π(y_l)/π_ref(y_l)))
    
    展开 log 比:
    L = -log σ(β * ((log π(y_w) - log π_ref(y_w)) - (log π(y_l) - log π_ref(y_l))))
    
    【参数说明】
    - ref_log_probs: 参考模型的 log 概率
      形状: [batch_size, seq_len]
      这是冻结的模型，提供基线
    
    - policy_log_probs: 策略模型 (正在训练) 的 log 概率
      形状: [batch_size, seq_len]
    
    - mask: 损失掩码，只在 assistant 回复部分计算
      形状: [batch_size, seq_len]
      1 表示计算损失，0 表示忽略
    
    - beta: 温度参数
      控制偏好学习的强度
      推荐值: 0.1
    
    【数据组织】
    batch 的前半部分是 chosen (好回复)
    batch 的后半部分是 rejected (差回复)
    
    例如 batch_size=4:
    [chosen_0, chosen_1, rejected_0, rejected_1]
    
    【计算步骤】
    1. 对每个序列，计算平均 log 概率 (序列级概率)
    2. 分开 chosen 和 rejected
    3. 计算 log ratio: log(π/π_ref)
    4. 应用 DPO 公式计算损失
    
    【返回】
    - loss: 标量损失值
    """
    # ==================== Step 1: 计算序列级 log 概率 ====================
    # token 级 log 概率需要聚合成序列级概率
    # 使用平均而非求和，避免长序列主导损失
    
    # 计算每个序列的有效长度 (mask=1 的位置数量)
    # clamp_min(1e-8) 防止除以 0
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
    
    # 只对 mask=1 的位置求和，然后除以有效长度
    # (log_probs * mask).sum(dim=1): 只累加有效位置的 log 概率
    # / seq_lengths: 求平均
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    # 现在形状: [batch_size]，每个样本一个标量

    # ==================== Step 2: 分开 chosen 和 rejected ====================
    # batch 的前半部分是 chosen，后半部分是 rejected
    batch_size = ref_log_probs.shape[0]
    half = batch_size // 2
    
    # 参考模型的 log 概率
    chosen_ref_log_probs = ref_log_probs[:half]     # 好回复在参考模型中的概率
    reject_ref_log_probs = ref_log_probs[half:]     # 差回复在参考模型中的概率
    
    # 策略模型的 log 概率
    chosen_policy_log_probs = policy_log_probs[:half]   # 好回复在策略模型中的概率
    reject_policy_log_probs = policy_log_probs[half:]   # 差回复在策略模型中的概率

    # ==================== Step 3: 计算 log ratio ====================
    # pi_logratios: 策略模型对 chosen vs rejected 的偏好程度
    # = log π(chosen) - log π(rejected)
    # > 0 表示策略更偏好 chosen
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    
    # ref_logratios: 参考模型对 chosen vs rejected 的偏好程度
    # = log π_ref(chosen) - log π_ref(rejected)
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    
    # ==================== Step 4: 计算 DPO 损失 ====================
    # logits = 策略模型的改进程度 - 参考模型的改进程度
    # = (log π(chosen) - log π(rejected)) - (log π_ref(chosen) - log π_ref(rejected))
    # = log(π(chosen)/π_ref(chosen)) - log(π(rejected)/π_ref(rejected))
    logits = pi_logratios - ref_logratios
    
    # 损失: -log sigmoid(β * logits)
    # 当 logits > 0 (策略比参考更偏好 chosen)，损失小
    # 当 logits < 0 (策略比参考更偏好 rejected)，损失大
    # β 放大这个差异
    loss = -F.logsigmoid(beta * logits)
    
    # 返回 batch 平均损失
    return loss.mean()


def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    """
    训练一个 epoch
    
    【DPO 训练流程详解】
    
    对于每个 batch:
    
    1. 数据准备:
       - 获取 chosen 和 rejected 对
       - 将它们拼接成一个大 batch
    
    2. 参考模型推理 (无梯度):
       - 计算 π_ref(chosen) 和 π_ref(rejected)
    
    3. 策略模型推理 (有梯度):
       - 计算 π(chosen) 和 π(rejected)
    
    4. 计算 DPO 损失:
       - L = -log σ(β * (log(π/π_ref)(chosen) - log(π/π_ref)(rejected)))
    
    5. 反向传播和优化:
       - 计算梯度
       - 更新策略模型参数
       - 参考模型保持不变
    
    【为什么需要参考模型】
    参考模型是训练开始时的策略模型的副本。
    它提供了一个"基线"，防止策略模型变化太大:
    - 只要求"比之前好一点"，而不是"完全重塑"
    - 保持语言能力的同时学习偏好
    - 类似于 PPO 中的 KL 惩罚
    
    【参数说明】
    - epoch: 当前 epoch 编号 (0-indexed)
    - loader: 数据加载器
    - iters: 本 epoch 的总迭代次数
    - ref_model: 参考模型 (冻结，不训练)
    - lm_config: 模型配置
    - start_step: 起始步数 (用于断点续训)
    - wandb: 实验追踪工具
    - beta: DPO 温度参数
    """
    # 记录训练开始时间，用于计算预计完成时间
    start_time = time.time()
    
    # ==================== 主训练循环 ====================
    # enumerate 从 start_step + 1 开始计数
    # 这样 step 从 1 开始 (而不是 0)
    for step, batch in enumerate(loader, start=start_step + 1):
        
        # ==================== Step 1: 数据准备 ====================
        # 从 batch 中提取数据并移到 GPU
        
        # x_chosen: chosen 回复的输入 token (去掉最后一个)
        # 形状: [batch_size, seq_len - 1]
        x_chosen = batch['x_chosen'].to(args.device)
        
        # x_rejected: rejected 回复的输入 token
        x_rejected = batch['x_rejected'].to(args.device)
        
        # y_chosen: chosen 回复的目标 token (去掉第一个)
        # 形状: [batch_size, seq_len - 1]
        # 语言模型是预测下一个 token: X[i] 预测 Y[i] = X[i+1]
        y_chosen = batch['y_chosen'].to(args.device)
        
        # y_rejected: rejected 回复的目标 token
        y_rejected = batch['y_rejected'].to(args.device)
        
        # mask_chosen: 只在 assistant 回复部分计算损失
        # 形状: [batch_size, seq_len - 1]
        # 1 表示这个位置需要计算损失，0 表示忽略
        mask_chosen = batch['mask_chosen'].to(args.device)
        
        # mask_rejected: rejected 的损失掩码
        mask_rejected = batch['mask_rejected'].to(args.device)
        
        # 将 chosen 和 rejected 拼接成一个大 batch
        # 这样可以一次推理计算两者的概率
        # 拼接后: [chosen_0, chosen_1, ..., rejected_0, rejected_1, ...]
        x = torch.cat([x_chosen, x_rejected], dim=0)    # [2*batch_size, seq_len-1]
        y = torch.cat([y_chosen, y_rejected], dim=0)    # [2*batch_size, seq_len-1]
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)  # [2*batch_size, seq_len-1]

        # ==================== Step 2: 学习率调度 ====================
        # 使用余弦学习率调度，从初始值平滑下降到接近 0
        # 计算当前全局步数
        global_step = epoch * iters + step
        total_steps = args.epochs * iters
        lr = get_lr(global_step, total_steps, args.learning_rate)
        
        # 更新优化器的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ==================== Step 3: 前向传播 ====================
        # 使用混合精度上下文 (bfloat16/float16)
        with autocast_ctx:
            
            # ----- 3a: 参考模型推理 (无梯度) -----
            # 参考模型提供基线，不需要计算梯度
            with torch.no_grad():
                # 前向传播得到 logits
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits  # [2*B, L, V]
            
            # 将 logits 转换为目标 token 的 log 概率
            ref_log_probs = logits_to_log_probs(ref_logits, y)  # [2*B, L]
            
            # ----- 3b: 策略模型推理 (有梯度) -----
            # 这是我们正在训练的模型
            outputs = model(x)
            logits = outputs.logits  # [2*B, L, V]
            policy_log_probs = logits_to_log_probs(logits, y)  # [2*B, L]
            
            # ----- 3c: 计算 DPO 损失 -----
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            
            # 添加 MoE 辅助损失 (如果使用 MoE 架构)
            # 辅助损失用于专家负载均衡
            loss = dpo_loss_val + outputs.aux_loss
            
            # 梯度累积: 将损失除以累积步数
            # 这样多个小 batch 的梯度累积后等价于一个大 batch
            loss = loss / args.accumulation_steps

        # ==================== Step 4: 反向传播 ====================
        # 使用 GradScaler 处理混合精度的梯度缩放
        # scale 会将损失放大，防止 float16 的梯度下溢
        scaler.scale(loss).backward()

        # ==================== Step 5: 参数更新 ====================
        # 每累积 N 步后才真正更新参数
        if (step + 1) % args.accumulation_steps == 0:
            # 反缩放梯度，恢复原始量级
            scaler.unscale_(optimizer)
            
            # 梯度裁剪: 将梯度的总范数限制在 grad_clip 以内
            # 这防止梯度爆炸导致训练不稳定
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            # 优化器更新参数
            # scaler.step 会检查是否有 NaN/Inf 梯度，有则跳过更新
            scaler.step(optimizer)
            
            # 更新 scaler 的缩放因子
            # 如果没有溢出，增大缩放因子；如果溢出，减小缩放因子
            scaler.update()
            
            # 清零梯度，准备下一次累积
            # set_to_none=True 比 zero_grad() 更高效
            optimizer.zero_grad(set_to_none=True)

        # ==================== Step 6: 日志记录 ====================
        # 每隔 log_interval 步打印一次日志
        if step % args.log_interval == 0 or step == iters - 1:
            # 计算已花费时间
            spend_time = time.time() - start_time
            
            # 恢复实际损失值 (乘回累积步数)
            current_loss = loss.item() * args.accumulation_steps
            current_dpo_loss = dpo_loss_val.item()
            current_aux_loss = outputs.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            
            # 估计剩余时间 (分钟)
            # = (平均每步时间 * 剩余步数) / 60
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            # 打印日志
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'loss: {current_loss:.4f}, '
                   f'dpo_loss: {current_dpo_loss:.4f}, '
                   f'aux_loss: {current_aux_loss:.4f}, '
                   f'learning_rate: {current_lr:.8f}, '
                   f'epoch_time: {eta_min:.3f}min')
            
            # 记录到 wandb/swanlab
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "dpo_loss": current_dpo_loss,
                    "aux_loss": current_aux_loss,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        # ==================== Step 7: 保存检查点 ====================
        # 每隔 save_interval 步保存一次模型
        # 只在主进程保存 (分布式训练时避免冲突)
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            # 切换到评估模式 (关闭 dropout 等)
            model.eval()
            
            # 构建保存路径
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            # 获取 state_dict
            # DDP 模型需要访问 .module 获取原始模型
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            # 转为 float16 节省磁盘空间 (几百 MB → 几十 MB)
            # 移到 CPU 避免 GPU 内存问题
            state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
            
            # 保存权重
            torch.save(state_dict, ckp)
            
            # 保存完整检查点 (包含优化器状态，用于断点续训)
            lm_checkpoint(
                lm_config, 
                weight=args.save_weight, 
                model=model, 
                optimizer=optimizer, 
                scaler=scaler, 
                epoch=epoch, 
                step=step, 
                wandb=wandb, 
                save_dir='../checkpoints'
            )
            
            # 切回训练模式
            model.train()
            del state_dict

        # ==================== Step 8: 释放内存 ====================
        # 显式删除不再需要的张量，帮助 GPU 内存回收
        # 这对于 GPU 内存紧张时特别重要
        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected
        del x, y, mask
        del ref_outputs, ref_logits, ref_log_probs
        del outputs, logits, policy_log_probs, loss


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    
    # ==================== 参数解析 ====================
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    
    # ----- 保存配置 -----
    parser.add_argument("--save_dir", type=str, default="../out", 
                        help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, 
                        help="保存权重的前缀名 (最终文件名: dpo_512.pth)")
    
    # ----- 训练配置 -----
    parser.add_argument("--epochs", type=int, default=1, 
                        help="训练轮数 (DPO 通常只需要 1 轮)")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="每个 GPU 的 batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-8, 
                        help="初始学习率 (建议 <= 5e-8 避免灾难性遗忘)")
    parser.add_argument("--device", type=str, 
                        default="cuda:0" if torch.cuda.is_available() else "cpu", 
                        help="训练设备 (cuda:0, cuda:1, cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16", 
                        help="混合精度类型 (bfloat16 或 float16)")
    parser.add_argument("--num_workers", type=int, default=8, 
                        help="数据加载的并行线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, 
                        help="梯度累积步数 (模拟更大的 batch size)")
    parser.add_argument("--grad_clip", type=float, default=1.0, 
                        help="梯度裁剪阈值 (防止梯度爆炸)")
    parser.add_argument("--log_interval", type=int, default=100, 
                        help="日志打印间隔 (步)")
    parser.add_argument("--save_interval", type=int, default=100, 
                        help="模型保存间隔 (步)")
    
    # ----- 模型配置 -----
    parser.add_argument('--hidden_size', default=512, type=int, 
                        help="隐藏层维度 (512 对应 ~26M 参数)")
    parser.add_argument('--num_hidden_layers', default=8, type=int, 
                        help="Transformer 层数")
    parser.add_argument('--max_seq_len', default=1024, type=int, 
                        help="最大序列长度 (中文 1 token ≈ 1.5~1.7 字符)")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], 
                        help="是否使用 MoE 架构 (0=否, 1=是)")
    
    # ----- 数据和权重 -----
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", 
                        help="DPO 训练数据路径 (JSONL 格式)")
    parser.add_argument('--from_weight', default='full_sft', type=str, 
                        help="基于哪个权重训练 (通常是 SFT 后的模型)")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], 
                        help="是否从检查点恢复训练 (0=否, 1=是)")
    
    # ----- DPO 超参数 -----
    parser.add_argument('--beta', default=0.1, type=float, 
                        help="DPO 温度参数 (越大越强调偏好差异)")
    
    # ----- 实验追踪 -----
    parser.add_argument("--use_wandb", action="store_true", 
                        help="是否使用 wandb/swanlab 追踪实验")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", 
                        help="wandb 项目名")
    
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 初始化分布式训练 (如果使用多 GPU)
    local_rank = init_distributed_mode()
    
    # 如果是分布式训练，根据 rank 设置设备
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    
    # 设置随机种子确保可复现性
    # 不同 rank 使用不同种子避免数据重复
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    # 创建保存目录 (如果不存在)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 创建模型配置
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )
    
    # 检查是否有可用的检查点用于恢复训练
    ckp_data = None
    if args.from_resume == 1:
        ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints')
    
    # ========== 3. 设置混合精度 ==========
    # 判断设备类型
    device_type = "cuda" if "cuda" in args.device else "cpu"
    
    # 选择精度类型
    # bfloat16: 更稳定，范围大，但精度略低
    # float16: 精度高，但可能溢出
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    
    # 创建混合精度上下文
    # CPU 不支持混合精度，使用空上下文
    if device_type == "cpu":
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置 wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        # 使用 swanlab (国产 wandb 替代品)
        import swanlab as wandb
        
        # 如果有检查点，尝试恢复 wandb 运行
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        
        # 创建运行名称
        wandb_run_name = f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        
        # 初始化 wandb
        wandb.init(
            project=args.wandb_project, 
            name=wandb_run_name, 
            id=wandb_id, 
            resume=resume
        )
    
    # ========== 5. 定义模型和参考模型 ==========
    # 加载策略模型 (正在训练的模型)
    # 通常基于 SFT 后的模型继续训练
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    
    # 初始化参考模型 (Reference Model)
    # 参考模型是策略模型的副本，但保持冻结
    # 它提供了一个"基线"，防止策略模型变化太大
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    
    # 冻结参考模型:
    # 1. 设置为评估模式 (关闭 dropout 等)
    ref_model.eval()
    # 2. 禁用梯度计算
    ref_model.requires_grad_(False)
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')
    
    # 加载 DPO 数据集
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    
    # 创建分布式采样器 (如果使用多 GPU)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # 创建 GradScaler (用于混合精度训练)
    # 只在 float16 时启用 (bfloat16 不需要)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # 创建 AdamW 优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从检查点恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 加载模型权重
        model.load_state_dict(ckp_data['model'])
        # 加载优化器状态 (包括动量等)
        optimizer.load_state_dict(ckp_data['optimizer'])
        # 加载 scaler 状态
        scaler.load_state_dict(ckp_data['scaler'])
        # 恢复训练进度
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 分布式数据并行 (DDP) ==========
    if dist.is_initialized():
        # 忽略 RoPE 的 buffer，它们不需要跨进程同步
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        
        # 用 DDP 包装模型
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 设置分布式采样器的 epoch
        # 这确保每个 epoch 数据打乱方式不同
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        if epoch == start_epoch and start_step > 0:
            # 第一个 epoch 且存在检查点，需要跳过已训练的步数
            batch_sampler = SkipBatchSampler(
                train_sampler or range(len(train_ds)), 
                args.batch_size, 
                start_step + 1
            )
            loader = DataLoader(
                train_ds, 
                batch_sampler=batch_sampler, 
                num_workers=args.num_workers, 
                pin_memory=True
            )
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + start_step + 1, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            # 从头开始训练这个 epoch
            loader = DataLoader(
                train_ds, 
                batch_size=args.batch_size, 
                shuffle=(train_sampler is None),  # 如果没有分布式采样器，手动打乱
                sampler=train_sampler, 
                num_workers=args.num_workers, 
                pin_memory=True  # 加速 CPU→GPU 数据传输
            )
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)
    
    # ========== 9. 清理分布式进程 ==========
    if dist.is_initialized():
        # 销毁进程组，释放资源
        dist.destroy_process_group()
