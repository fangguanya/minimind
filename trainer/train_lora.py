"""
================================================================================
                    MiniMind LoRA 微调脚本 (Low-Rank Adaptation)
================================================================================

【什么是 LoRA】
LoRA 是一种参数高效微调方法 (Parameter-Efficient Fine-Tuning, PEFT):
- 冻结预训练模型的所有参数
- 在关键层旁边注入可训练的低秩矩阵
- 只训练这些新增的小矩阵

【数学原理】
对于预训练权重 W ∈ R^(d×k):
- 原始计算: h = W·x
- LoRA 修改: h = W·x + ΔW·x = W·x + B·A·x

其中:
- A ∈ R^(r×k): 下投影矩阵 (降维)
- B ∈ R^(d×r): 上投影矩阵 (升维)
- r << min(d, k): 秩，通常 r=4, 8, 16

【为什么 LoRA 有效】
1. 低秩假设: 微调过程中的权重变化是低秩的
   - 预训练模型已经学到了很好的表示
   - 微调只需要在低维子空间中调整

2. 参数效率:
   - 完整微调: d × k 个参数 (例如 512 × 512 = 262144)
   - LoRA (r=8): r × (d + k) = 8 × 1024 = 8192 (仅 3%)

【初始化策略】
- A: 随机初始化 (小值高斯分布)
- B: 零初始化
- 这确保了训练开始时 ΔW = B·A = 0
- 模型行为与预训练完全一致

【与 Full SFT 的对比】
┌─────────────────┬────────────┬────────────┐
│                 │  Full SFT  │    LoRA    │
├─────────────────┼────────────┼────────────┤
│ 可训练参数      │   100%     │    1-3%    │
│ 显存占用        │   高       │    低      │
│ 训练速度        │   慢       │    快      │
│ 灾难性遗忘      │   可能     │    较少    │
│ 多任务切换      │   需重载   │  快速切换  │
└─────────────────┴────────────┴────────────┘

【典型应用场景】
1. 身份注入 (Identity): 让模型认识自己是"MiniMind"
2. 领域适配 (Domain): 医疗、法律、金融等专业知识
3. 风格调整 (Style): 改变回复风格

【使用方法】
python train_lora.py --lora_name lora_identity --data_path ../dataset/lora_identity.jsonl

【LoRA 参数选择建议】
- rank=4~8: 简单任务 (身份注入、风格调整)
- rank=16~32: 中等任务 (领域适配)
- rank=64+: 复杂任务 (需要更多表达能力)
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
from model.model_lora import save_lora, apply_lora
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

# 忽略不重要的警告信息
warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
    """
    训练一个 epoch (LoRA 版本)
    
    【与 Full SFT 的区别】
    1. 只更新 LoRA 参数，原模型参数冻结
    2. 梯度裁剪只针对 LoRA 参数
    3. 保存时只保存 LoRA 权重
    
    【LoRA 训练流程】
    for each batch:
        1. 前向传播: logits = model(X)
           - 原模型权重: 冻结，不计算梯度
           - LoRA 权重: 参与计算，计算梯度
        2. 计算损失: loss = CrossEntropy(logits, Y) * loss_mask
        3. 反向传播: 梯度只流向 LoRA 参数
        4. 参数更新: 只更新 LoRA 参数
    
    【参数说明】
    - epoch: 当前 epoch 编号
    - loader: 数据加载器
    - iters: 每个 epoch 的迭代次数
    - lora_params: LoRA 参数列表 (用于梯度裁剪)
    - start_step: 起始步数 (用于断点续训)
    - wandb: 实验追踪工具
    """
    # 使用交叉熵损失，reduction='none' 返回每个位置的损失
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    
    for step, (X, Y, loss_mask) in enumerate(loader, start=start_step + 1):
        # 数据移到设备
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        
        # 动态调整学习率
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ==================== 前向传播和损失计算 ====================
        with autocast_ctx:
            # 前向传播
            # 原模型权重冻结，但 LoRA 权重可学习
            # 输出是: original_output + lora_output
            res = model(X)
            
            # 计算交叉熵损失
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            # 应用 loss_mask，只计算 assistant 回复部分
            logits_loss = (loss * loss_mask).sum() / loss_mask.sum()
            
            # 添加 MoE 辅助损失 (如果使用 MoE)
            loss = logits_loss + res.aux_loss
            
            # 梯度累积
            loss = loss / args.accumulation_steps

        # ==================== 反向传播 ====================
        # 梯度只流向 LoRA 参数 (其他参数已冻结)
        scaler.scale(loss).backward()

        # ==================== 参数更新 ====================
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            # 注意: 只对 LoRA 参数进行梯度裁剪
            # 因为其他参数的 grad 是 None (已冻结)
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        # ==================== 日志记录 ====================
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_logits_loss = logits_loss.item()
            current_aux_loss = res.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')
            
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
            model.eval()
            lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}.pth'
            
            # 关键区别: LoRA 只保存 LoRA 权重，非常小
            # 通常只有几 MB，而不是完整模型的几百 MB
            save_lora(model, lora_save_path)
            
            # 保存完整检查点 (用于断点续训)
            lm_checkpoint(lm_config, weight=args.lora_name, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()

        del X, Y, loss_mask, res, loss


if __name__ == "__main__":
    # ==================== 参数解析 ====================
    parser = argparse.ArgumentParser(description="MiniMind LoRA Fine-tuning")
    
    # 保存配置
    parser.add_argument("--save_dir", type=str, default="../out/lora", help="模型保存目录")
    parser.add_argument("--lora_name", type=str, default="lora_identity", help="LoRA权重名称(如lora_identity/lora_medical等)")
    
    # 训练配置
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=10, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1, help="模型保存间隔")
    
    # 模型配置
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    
    # 数据和权重
    parser.add_argument("--data_path", type=str, default="../dataset/lora_identity.jsonl", help="LoRA训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练，默认full_sft")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    
    # 实验追踪
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-LoRA", help="wandb项目名")
    
    args = parser.parse_args()

    # ==================== 1. 初始化环境和随机种子 ====================
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ==================== 2. 配置目录、模型参数、检查ckp ====================
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir='../checkpoints') if args.from_resume==1 else None
    
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
        wandb_run_name = f"MiniMind-LoRA-{args.lora_name}-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ==================== 5. 定义模型、应用LoRA、冻结非LoRA参数 ====================
    # 加载预训练/SFT 模型
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    
    # 为模型注入 LoRA 层
    # apply_lora 会:
    # 1. 找到所有方阵线性层 (通常是 Q/K/V/O 投影)
    # 2. 在每个层旁边添加一个 LoRA 模块
    # 3. 修改 forward 方法: output = original(x) + lora(x)
    apply_lora(model)
    
    # ==================== 参数统计 ====================
    # 计算各类参数数量，展示 LoRA 的参数效率
    total_params = sum(p.numel() for p in model.parameters())
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
    Logger(f"LLM 总参数量: {total_params / 1e6:.3f} M")
    Logger(f"LoRA 参数量: {lora_params_count / 1e6:.3f} M")
    Logger(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")
    
    # ==================== 冻结非LoRA参数，收集LoRA参数 ====================
    # 这是 LoRA 的核心: 只训练 LoRA 参数
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            # LoRA 参数: 可训练
            param.requires_grad = True
            lora_params.append(param)
        else:
            # 原模型参数: 冻结
            param.requires_grad = False
    
    # ==================== 6. 定义数据和优化器 ====================
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    
    # 优化器只优化 LoRA 参数
    # 这比优化全部参数快得多
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    
    # ==================== 7. 从ckp恢复状态 ====================
    start_epoch, start_step = 0, 0
    if ckp_data:
        # strict=False 因为只加载 LoRA 权重
        model.load_state_dict(ckp_data['model'], strict=False)
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ==================== 8. DDP包模型 ====================
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ==================== 9. 开始训练 ====================
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        
        if epoch == start_epoch and start_step > 0:
            # 断点续训
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + start_step + 1, lora_params, start_step, wandb)
        else:
            # 从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(epoch, loader, len(loader), lora_params, 0, wandb)
    
    # ==================== 10. 清理分布进程 ====================
    if dist.is_initialized():
        dist.destroy_process_group()
