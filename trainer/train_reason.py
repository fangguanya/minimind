"""
================================================================================
                    MiniMind 推理模型训练脚本 (Reasoning Distillation)
================================================================================

【什么是推理模型】
推理模型是能够展示"思考过程"的语言模型:
- 输出包含 <think>思考过程</think> 和 <answer>最终答案</answer>
- 类似于 DeepSeek-R1、OpenAI o1 等模型
- 通过展示推理步骤，提高回答的可解释性和准确性

【训练方法】
推理蒸馏 (Reasoning Distillation):
- 使用大模型 (如 DeepSeek-R1) 生成的推理数据
- 小模型学习模仿这种"思考-回答"的模式
- 这是一种监督学习，不是强化学习

【数据格式】
{
    "conversations": [
        {"role": "user", "content": "1+1等于多少？"},
        {"role": "assistant", "content": "<think>\n让我思考一下...\n1+1是基本的加法运算\n结果是2\n</think>\n<answer>\n1+1=2\n</answer>"}
    ]
}

【核心技术: 特殊标签加权】
为了让模型更好地学习推理格式，对关键标签增加损失权重:
- <think>, </think>, <answer>, </answer> 这些标签
- 权重设为 10 倍
- 这强迫模型一定要学会使用正确的格式

【为什么需要特殊标签加权】
1. 格式标签在训练数据中出现次数很少
2. 但它们对于推理格式至关重要
3. 增加权重确保模型不会"跳过"这些标签
4. 没有正确格式，推理能力就无法发挥

【与普通 SFT 的区别】
┌─────────────────┬──────────────┬──────────────┐
│                 │   普通 SFT   │  推理蒸馏    │
├─────────────────┼──────────────┼──────────────┤
│ 回复格式        │   自由文本   │  结构化格式  │
│ 思考过程        │   不显式     │  显式展示    │
│ 标签权重        │   均匀       │  格式标签加权│
│ 基于权重        │   pretrain   │   dpo/sft    │
└─────────────────┴──────────────┴──────────────┘

【训练流程】
1. 加载 DPO 或 SFT 后的模型
2. 使用推理格式数据微调
3. 对格式标签增加损失权重
4. 模型学会"思考-回答"模式

【使用方法】
python train_reason.py --from_weight dpo --data_path ../dataset/r1_mix_1024.jsonl
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


def train_epoch(epoch, loader, iters, tokenizer, lm_config, start_step=0, wandb=None):
    """
    训练一个 epoch (推理蒸馏版本)
    
    【与普通 SFT 的关键区别】
    对特殊思考标签增加损失权重:
    - <think>: 开始思考
    - </think>: 结束思考
    - <answer>: 开始回答
    - </answer>: 结束回答
    
    【标签加权的实现】
    1. 将标签 token 化得到 ID 序列
    2. 在计算损失时，找到这些 ID 的位置
    3. 将这些位置的 loss_mask 乘以 10
    4. 这样这些位置的损失贡献更大
    
    【为什么设为 10 倍】
    - 实验发现 10 倍能有效学习格式
    - 太小: 模型可能忽略格式
    - 太大: 可能影响内容质量
    
    【参数说明】
    - epoch: 当前 epoch 编号
    - loader: 数据加载器
    - iters: 每个 epoch 的迭代次数
    - tokenizer: 分词器 (用于获取特殊标签的 token ID)
    - lm_config: 模型配置
    - start_step: 起始步数
    - wandb: 实验追踪
    """
    # ==================== 预计算特殊标签的 token ID ====================
    # 这些是推理模型的关键格式标签
    # 将它们转换为 token ID，以便在损失计算时识别
    start_of_think_ids = tokenizer('<think>').input_ids      # 开始思考标签
    end_of_think_ids = tokenizer('</think>').input_ids       # 结束思考标签
    start_of_answer_ids = tokenizer('<answer>').input_ids    # 开始回答标签
    end_of_answer_ids = tokenizer('</answer>').input_ids     # 结束回答标签
    
    # 使用交叉熵损失
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
            res = model(X)
            
            # 计算交叉熵损失
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            # ==================== 特殊标签位置增加权重 ====================
            # 这是推理蒸馏的核心技术
            
            # 找到特殊标签的位置
            # torch.isin() 检查 Y 中的每个值是否在特殊标签 ID 列表中
            sp_ids = torch.isin(
                Y.view(-1),
                torch.tensor(
                    start_of_think_ids + end_of_think_ids + 
                    start_of_answer_ids + end_of_answer_ids
                ).to(args.device)
            )
            
            # 展平 loss_mask 以便操作
            loss_mask = loss_mask.view(-1)
            # 保存原始 mask 和用于归一化
            loss_mask_sum = loss_mask.sum()
            
            # 对特殊标签位置增加 10 倍权重
            # 这确保模型一定会学习正确使用这些格式标签
            loss_mask[sp_ids] = 10
            
            # 重塑回原始形状
            loss_mask = loss_mask.view(Y.size())
            
            # 计算加权损失
            # 注意: 归一化使用原始的 loss_mask_sum，不是加权后的
            # 这样不会因为增加权重而改变总体损失量级
            logits_loss = (loss * loss_mask).sum() / loss_mask_sum
            
            # 添加 MoE 辅助损失
            loss = logits_loss + res.aux_loss
            
            # 梯度累积
            loss = loss / args.accumulation_steps

        # ==================== 反向传播 ====================
        scaler.scale(loss).backward()

        # ==================== 参数更新 ====================
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            
            state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        del X, Y, loss_mask, res, loss


if __name__ == "__main__":
    # ==================== 参数解析 ====================
    parser = argparse.ArgumentParser(description="MiniMind Reasoning Distillation")
    
    # 保存配置
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='reason', type=str, help="保存权重的前缀名")
    
    # 训练配置
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    
    # 模型配置
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=720, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    
    # 数据和权重
    parser.add_argument("--data_path", type=str, default="../dataset/r1_mix_1024.jsonl", help="推理蒸馏数据路径")
    parser.add_argument('--from_weight', default='dpo', type=str, help="基于哪个权重训练，默认dpo")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    
    # 实验追踪
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Reasoning", help="wandb项目名")
    
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
        wandb_run_name = f"MiniMind-Reasoning-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ==================== 5. 定义模型、数据、优化器 ====================
    # 通常基于 DPO 后的模型继续训练
    # 因为 DPO 已经让模型学会了更好的回复质量
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    
    # 加载推理蒸馏数据
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
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
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ==================== 8. 开始训练 ====================
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        
        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + start_step + 1, tokenizer, lm_config, start_step, wandb)
        else:
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            train_epoch(epoch, loader, len(loader), tokenizer, lm_config, 0, wandb)
    
    # ==================== 9. 清理分布进程 ====================
    if dist.is_initialized():
        dist.destroy_process_group()
