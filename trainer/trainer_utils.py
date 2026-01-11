"""
================================================================================
                    训练工具函数集合
================================================================================

本文件包含训练过程中使用的各种工具函数:

1. get_model_params: 计算并打印模型参数量
2. get_lr: 余弦学习率调度器
3. init_distributed_mode: 初始化分布式训练环境
4. setup_seed: 设置随机种子
5. lm_checkpoint: 检查点保存和加载
6. init_model: 模型初始化
7. SkipBatchSampler: 用于断点续训的采样器
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM

def get_model_params(model, config):
    """
    计算并打印模型参数量
    
    对于 MoE 模型，会区分总参数量和激活参数量:
    - 总参数量: 所有专家的参数
    - 激活参数量: 每次推理实际使用的参数 (只激活 top-k 个专家)
    """
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    """检查当前进程是否是主进程 (DDP 中 rank=0)"""
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    """只在主进程打印日志"""
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    余弦学习率调度器
    
    【公式】
    lr = lr_base * (0.1 + 0.45 * (1 + cos(π * step / total_steps)))
    
    【特点】
    - 从 lr 开始
    - 平滑下降到 0.1 * lr
    - 余弦曲线，后期下降更快
    """
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    """
    初始化分布式训练环境 (DDP)
    
    【使用方法】
    torchrun --nproc_per_node=4 train.py
    
    【环境变量】
    - RANK: 全局进程编号
    - LOCAL_RANK: 本机进程编号
    - WORLD_SIZE: 总进程数
    
    【返回】
    - local_rank: 本地 GPU 编号
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非 DDP 模式

    # 初始化进程组，使用 NCCL 后端 (GPU 通信最优)
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    """
    设置随机种子，确保实验可复现
    
    【设置的种子】
    - Python random
    - NumPy
    - PyTorch CPU/GPU
    - cuDNN (设置为确定性模式)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 确定性计算
    torch.backends.cudnn.benchmark = False     # 关闭自动优化 (可能引入随机性)

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        from torch.nn.parallel import DistributedDataParallel
        state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    if isinstance(value, DistributedDataParallel):
                        resume_data[key] = value.module.state_dict()
                    else:
                        resume_data[key] = value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    """
    初始化模型和分词器
    
    【参数】
    - lm_config: 模型配置
    - from_weight: 加载的权重名称，'none' 表示随机初始化
    - tokenizer_path: 分词器路径
    - save_dir: 权重目录
    - device: 设备
    
    【返回】
    - model: 初始化后的模型
    - tokenizer: 分词器
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    # 加载预训练权重 (如果指定)
    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    """
    可跳过指定批次的采样器 (用于断点续训)
    
    【用途】
    当从检查点恢复训练时，需要跳过已经训练过的批次
    
    【参数】
    - sampler: 原始采样器
    - batch_size: 批量大小
    - skip_batches: 要跳过的批次数
    """
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                # 跳过指定数量的批次
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        # 处理最后一个不完整的批次
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)