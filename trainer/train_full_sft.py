"""
================================================================================
                    MiniMind 监督微调脚本 (Supervised Fine-Tuning, SFT)
================================================================================

【什么是监督微调 (SFT)】
SFT 是 LLM 训练的第二阶段，在预训练模型基础上:
- 用高质量的对话数据微调
- 让模型学会"遵循指令"和"进行对话"
- 输出更符合人类期望的回复

【预训练 vs SFT】
- 预训练: 学习"什么是语言" (language modeling)
- SFT: 学习"如何回答问题" (instruction following)

【SFT 的核心改变】
1. 数据格式:
   - 预训练: 纯文本 "今天天气很好..."
   - SFT: 对话格式 [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "回答"}]

2. 损失计算:
   - 预训练: 对所有 token 计算损失
   - SFT: 只对 assistant 回复计算损失 (Loss Masking)

【Loss Masking 的原理】
为什么只在 assistant 部分计算损失？

考虑对话:
User: 今天天气怎么样？
Assistant: 今天天气很好，阳光明媚。

如果对所有 token 计算损失:
- 模型会学习复述用户输入
- 这不是我们想要的

只对 assistant 部分计算损失:
- 模型学习"给定用户输入，如何生成回复"
- 这才是我们需要的能力

【训练目标】
与预训练相同，都是下一个 token 预测:
Loss = -Σ mask_i * log P(token_i | context)

但 mask_i 只在 assistant 回复部分为 1

【学习率选择】
SFT 的学习率通常比预训练小很多:
- 预训练: 1e-4 ~ 5e-4
- SFT: 1e-6 ~ 1e-5

原因: 预训练已经学到了很好的表示，SFT 只需要微调

【使用方法】
单卡训练:
    python train_full_sft.py --epochs 2 --batch_size 16

多卡训练:
    torchrun --nproc_per_node=4 train_full_sft.py --epochs 2 --batch_size 16
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
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

# 忽略不重要的警告信息
warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    训练一个 epoch
    
    【SFT 训练流程】
    for each batch:
        1. 获取数据: X (输入), Y (标签), loss_mask (掩码)
        2. 前向传播: logits = model(X)
        3. 计算损失: loss = CrossEntropy(logits, Y) * loss_mask
           - loss_mask 只在 assistant 回复部分为 1
           - 这样只对 assistant 的回复计算损失
        4. 反向传播: loss.backward()
        5. 参数更新: optimizer.step()
    
    【Loss Mask 的作用】
    假设一个对话:
    <|im_start|>user
    你好
    <|im_end|>
    <|im_start|>assistant
    你好！有什么可以帮助你的？
    <|im_end|>
    
    loss_mask = [0,0,0,...(user部分)...,1,1,1,...(assistant部分)...,0]
    
    这样模型只学习生成 assistant 的回复，不学习用户输入
    
    【参数说明】
    - epoch: 当前 epoch 编号
    - loader: 数据加载器
    - iters: 每个 epoch 的迭代次数
    - start_step: 起始步数 (用于断点续训)
    - wandb: 实验追踪工具
    """
    # 使用交叉熵损失，reduction='none' 返回每个位置的损失
    # 这样我们可以手动应用 loss_mask
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        # ==================== 数据准备 ====================
        # X: 输入 token [batch, seq_len-1]
        # Y: 目标 token [batch, seq_len-1] (X 向后移一位)
        # loss_mask: 只在 assistant 回复部分为 1
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        
        # ==================== 学习率调度 ====================
        # 使用余弦学习率调度，从初始值平滑下降
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ==================== 前向传播和损失计算 ====================
        with autocast_ctx:  # 混合精度上下文
            # 前向传播得到 logits
            res = model(X)
            
            # 计算交叉熵损失
            # logits: [batch, seq_len, vocab_size] -> [batch*seq_len, vocab_size]
            # Y: [batch, seq_len] -> [batch*seq_len]
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())  # 重塑回 [batch, seq_len]

            # 应用 loss_mask，只计算 assistant 部分的损失
            # (loss * loss_mask).sum(): 被 mask 的位置损失为 0
            # loss_mask.sum(): 有效位置的数量
            logits_loss = (loss * loss_mask).sum() / loss_mask.sum()
            
            # 如果使用 MoE，添加辅助损失 (用于专家负载均衡)
            loss = logits_loss + res.aux_loss
            
            # 梯度累积: 损失除以累积步数
            loss = loss / args.accumulation_steps

        # ==================== 反向传播 ====================
        # 使用 GradScaler 处理混合精度的梯度
        scaler.scale(loss).backward()

        # ==================== 参数更新 ====================
        # 每累积 N 步后更新参数
        if (step + 1) % args.accumulation_steps == 0:
            # 反缩放梯度
            scaler.unscale_(optimizer)
            # 梯度裁剪，防止梯度爆炸
            # 将所有参数的梯度范数限制在 grad_clip 以内
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 优化器更新参数
            scaler.step(optimizer)
            # 更新 scaler 的缩放因子
            scaler.update()
            # 清零梯度
            # set_to_none=True 比 zero_grad() 更高效
            optimizer.zero_grad(set_to_none=True)

        # ==================== 日志记录 ====================
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            # 恢复实际损失值 (乘回累积步数)
            current_loss = loss.item() * args.accumulation_steps
            current_logits_loss = logits_loss.item()
            current_aux_loss = res.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            # 估计剩余时间
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')
            
            # 记录到 wandb/swanlab
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "logits_loss": current_logits_loss,
                    "aux_loss": current_aux_loss,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        # ==================== 保存检查点 ====================
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()  # 切换到评估模式
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            # 处理 DDP 包装的模型
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            # 转为半精度节省空间
            state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            
            # 保存完整检查点 (用于断点续训)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()  # 切换回训练模式
            del state_dict

        # 及时释放内存
        del X, Y, loss_mask, res, loss


if __name__ == "__main__":
    # ==================== 参数解析 ====================
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    
    # 保存配置
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    
    # 训练配置
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    
    # 模型配置
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    
    # 数据和权重
    parser.add_argument("--data_path", type=str, default="../dataset/sft_mini_512.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    
    # 实验追踪
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb项目名")
    
    args = parser.parse_args()

    # ==================== 1. 初始化环境和随机种子 ====================
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    # 设置随机种子确保可复现性
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ==================== 2. 配置目录、模型参数、检查ckp ====================
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )
    # 检查是否有可用的检查点用于断点续训
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ==================== 3. 设置混合精度 ====================
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ==================== 4. 配wandb ====================
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ==================== 5. 定义模型、数据、优化器 ====================
    # 加载预训练权重初始化模型
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    
    # 加载 SFT 数据集
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # 初始化 GradScaler (float16 需要，bfloat16 不需要)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # 使用 AdamW 优化器
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
    if dist.is_initialized():
        # 忽略 RoPE 的 buffer，它们不需要跨进程同步
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ==================== 8. 开始训练 ====================
    for epoch in range(start_epoch, args.epochs):
        # 设置 sampler 的 epoch，确保每个 epoch 数据打乱方式不同
        train_sampler and train_sampler.set_epoch(epoch)
        
        if epoch == start_epoch and start_step > 0:
            # 第一个 epoch 且存在检查点，跳过已训练的步数
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + start_step + 1, start_step, wandb)
        else:
            # 默认从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ==================== 9. 清理分布进程 ====================
    if dist.is_initialized():
        dist.destroy_process_group()
