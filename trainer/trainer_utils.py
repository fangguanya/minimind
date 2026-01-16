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
    余弦学习率调度器 (Cosine Annealing)
    
    【算法来源】
    论文: "SGDR: Stochastic Gradient Descent with Warm Restarts" (Loshchilov & Hutter, ICLR 2017)
    
    【数学原理】
    利用余弦函数 cos(x) 在 [0, π] 区间从 1 平滑下降到 -1 的特性：
    - 当 step=0 时: cos(0)=1, 系数=(0.1 + 0.45×2)=1.0, 输出=lr
    - 当 step=total 时: cos(π)=-1, 系数=(0.1 + 0.45×0)=0.1, 输出=0.1×lr
    
    【公式】
    lr_out = lr × (0.1 + 0.45 × (1 + cos(π × step / total_steps)))
    
    【工程效果】
    - 前期: 学习率高，模型快速学习主要特征
    - 中期: 平滑过渡，避免训练震荡
    - 后期: 学习率低且下降趋缓，精细调优找到更好的局部最优
    - 相比线性衰减: 曲线更平滑，最终模型性能通常更好
    - 相比阶梯衰减: 无突变点，训练 loss 曲线更稳定
    
    【为什么不降到0】
    保留 10% 的最小学习率，防止训练后期完全停滞，保持一定的探索能力
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
    """
    训练检查点保存/加载系统
    
    【设计原理】
    采用"双文件策略"分离不同用途：
    1. ckp_path (推理文件): 仅包含模型权重，体积小，用于部署推理
    2. resume_path (续训文件): 包含完整训练状态，用于断点续训
    
    【核心技术点】
    
    1. 原子写入 (Atomic Write)
       先写入 .tmp 临时文件，再用 os.replace() 原子替换
       原理: os.replace 是操作系统级原子操作，要么完全成功，要么完全失败
       效果: 即使保存过程中断电/崩溃，原检查点文件也不会损坏
    
    2. DDP 模型解包
       DistributedDataParallel 会把模型包装一层，真正的参数在 model.module 里
       保存时需要解包，否则加载时会出现 key 不匹配 (多了 'module.' 前缀)
    
    3. FP16 压缩
       state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
       将 FP32 参数转为 FP16，存储空间减半，加载速度更快
       对于推理来说精度损失可忽略不计
    
    4. World Size 自适应
       记录保存时的 GPU 数量，恢复时如果 GPU 数量变化，自动换算 step
       例: 4卡训练1000步 → 2卡恢复时转为2000步 (因为每步处理的 batch 变小了)
    
    5. WandB ID 保存
       WandB (Weights & Biases) 是机器学习实验跟踪平台，用于:
       - 实时记录训练曲线 (loss, lr, 显存等)
       - 可视化对比不同实验
       - 云端保存，团队共享
       保存 wandb_id 是为了断点续训时接续同一个实验，而不是新建
    
    6. 动态 kwargs 保存
       支持保存额外对象 (如 lr_scheduler)，自动检测是否有 state_dict 方法
    
    【使用方式】
    - 保存: lm_checkpoint(config, model=model, optimizer=opt, epoch=e, step=s)
    - 加载: ckp_data = lm_checkpoint(config)  # model=None 时进入加载模式
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'       # 推理用: 仅模型权重
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'  # 续训用: 完整状态

    if model is not None:  # ========== 保存模式 ==========
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
    
    【理论依据】
    梯度下降是一个确定性的迭代过程:
        θ_{n+1} = θ_n - lr × ∇L(θ_n, data_n)
    
    只要完整保存了 step n 时刻的状态 (模型参数 + 优化器状态)，
    从 step n+1 继续训练，在数学上完全等价于训练从未中断。
    
    【为什么要跳过批次】
    假设 epoch 0 有 [batch0, batch1, batch2, batch3, ...] 这些批次
    如果在训练完 batch2 时崩溃，恢复后:
    - 不跳过: batch0, batch1 被训练了两次 → 数据分布不均匀，相当于过采样
    - 跳过: 直接从 batch3 继续 → 每个样本训练次数一致
    
    【断点续训需要保存的完整状态】
    1. 模型参数 (state_dict): 训练的核心成果
    2. 优化器状态: Adam 的一阶动量(m)和二阶动量(v)
       - 不保存会导致: 恢复后 loss 震荡，收敛变慢
       - 原理: Adam 公式 θ = θ - lr * m / (sqrt(v) + ε)，m和v需要积累
    3. epoch/step: 知道要跳过多少批次
    4. lr_scheduler 状态: 保证学习率曲线连续
    5. 随机数状态 (可选): torch/numpy/python 的 RNG state
       - 严格恢复需要保存，但实践中影响很小
       - 因为每个 epoch 本来就会重新 shuffle
    
    【数学上的等价性证明】
    设 f(θ, D) 表示在数据 D 上训练一步后的参数
    
    连续训练: θ_3 = f(f(f(θ_0, D_0), D_1), D_2)
    断点续训: θ_3 = f(θ_2, D_2)  其中 θ_2 是保存的检查点
    
    只要 θ_2 完全相同，结果就完全相同。
    
    【参数】
    - sampler: 原始采样器 (通常是 DistributedSampler 或 RandomSampler)
    - batch_size: 批量大小
    - skip_batches: 要跳过的批次数 (= 保存时的 step 数)
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