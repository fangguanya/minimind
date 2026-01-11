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
