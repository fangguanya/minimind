"""
================================================================================
                    LoRA (Low-Rank Adaptation) 实现
================================================================================

【什么是 LoRA】
LoRA 是一种高效的微调方法，核心思想是:
- 冻结预训练模型的权重
- 在需要微调的层旁边添加低秩分解的小矩阵
- 只训练这些小矩阵，大大减少可训练参数

【数学原理】
原始权重矩阵 W ∈ R^(d×k)
LoRA 添加: ΔW = B @ A，其中 A ∈ R^(r×k), B ∈ R^(d×r), r << min(d,k)

前向传播: y = W @ x + ΔW @ x = W @ x + B @ A @ x

【@ 是什么？矩阵乘法！】
Python 3.5+ 引入: A @ B 等价于 torch.matmul(A, B)

矩阵乘法规则: 结果[i,j] = A的第i行 · B的第j列（点积）

例: A=[2,3矩阵], B=[3,4矩阵] → A@B=[2,4矩阵]

【LoRA 中的计算流程】（假设 dim=512, rank=8）
1. A @ x: [8,512] @ [512] = [8]      压缩到8维
2. B @ (A@x): [512,8] @ [8] = [512]  投影回512维
3. W @ x: [512,512] @ [512] = [512]  原始结果
4. 相加得到最终输出 [512]

低秩的妙处: 用 8192 参数近似 262144 参数的效果！

【A/B 是怎么被训练的？】

关键: requires_grad 控制谁被训练！

原始权重 W: requires_grad = False  ← 冻结，不训练
LoRA 的 A: requires_grad = True   ← 可训练
LoRA 的 B: requires_grad = True   ← 可训练

前向: y = W @ x + B @ A @ x
反向: loss.backward() 只算 A 和 B 的梯度（因为 W 冻结了）
更新: optimizer.step() 只更新 A 和 B

A 和 B 被"同时"训练：链式法则自动算两个的梯度！

【参数如何减少？】

全量微调 W [512×512]: 262,144 参数
LoRA A[8,512] + B[512,8]: 8,192 参数  ← 减少32倍！

为什么能用这么少？
- 微调的权重变化 ΔW 通常是"低秩"的
- 预训练模型已经很好，微调改动很小
- 这些改动集中在几个"方向"上，rank=8 就够了

【为什么有效】
1. 预训练模型的权重已经很好，微调只需要小的调整
2. 这些调整通常是低秩的 (rank << d)
3. 只训练 r×(d+k) 个参数，而不是 d×k 个

【典型设置】
- rank=8 或 rank=16
- 只在 attention 的 Q/K/V/O 投影层使用
- 可以减少 90%+ 的可训练参数

【优势】
1. 显存占用少：只需存储小矩阵的梯度
2. 训练快：参数少，优化器状态小
3. 可插拔：可以轻松切换不同的 LoRA 适配器
4. 保持原模型：原模型权重不变，不会灾难性遗忘
"""

import torch
from torch import optim, nn


class LoRA(nn.Module):
    """
    LoRA 低秩适配器
    
    【结构】
    原始层: y = W @ x
    添加 LoRA 后: y = W @ x + B @ A @ x
    
    其中:
    - A: [in_features, rank] 下投影矩阵
    - B: [rank, out_features] 上投影矩阵
    - rank << in_features, out_features
    
    【初始化】
    - A: 高斯初始化 (std=0.02)
    - B: 零初始化
    - 这样初始时 ΔW = B @ A = 0，LoRA 不改变原模型行为
    """
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA 的秩，控制适配器容量
        
        # 低秩分解: W = B @ A
        # A: 下投影 in_features → rank
        self.A = nn.Linear(in_features, rank, bias=False)
        # B: 上投影 rank → out_features  
        self.B = nn.Linear(rank, out_features, bias=False)
        
        # 初始化策略:
        # A: 小的高斯初始化，让 LoRA 能学到信息
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # B: 零初始化，确保初始时 LoRA 输出为 0
        self.B.weight.data.zero_()

    def forward(self, x):
        """LoRA 前向传播: x → A → B → output"""
        return self.B(self.A(x))


def apply_lora(model, rank=8):
    """
    为模型的线性层添加 LoRA 适配器
    
    【应用策略】
    只为方阵线性层添加 LoRA:
    - 这通常是 attention 层中的 Q/K/V/O 投影
    - 方阵特征: weight.shape[0] == weight.shape[1]
    
    【实现方式】
    1. 为每个目标层创建 LoRA 模块
    2. 修改层的 forward 方法: output = original(x) + lora(x)
    
    【参数】
    - model: 要添加 LoRA 的模型
    - rank: LoRA 的秩，越大容量越大，但参数也越多
    """
    for name, module in model.named_modules():
        # 只为方阵线性层添加 LoRA (通常是 attention 投影)
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 创建 LoRA 适配器
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            # 修改 forward: 原始输出 + LoRA 输出
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora


def load_lora(model, path):
    """
    加载 LoRA 权重
    
    【流程】
    1. 从文件加载状态字典
    2. 处理 DDP 前缀 (module.)
    3. 找到每个 LoRA 模块对应的权重
    4. 加载到模型中
    """
    state_dict = torch.load(path, map_location=model.device)
    # 移除可能的 DDP 前缀
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 提取该层的 LoRA 权重
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """
    保存 LoRA 权重
    
    【注意】
    只保存 LoRA 参数，不保存原模型参数
    - 文件非常小 (通常几 MB)
    - 需要配合原模型使用
    """
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 处理 DDP 前缀
            clean_name = name[7:] if name.startswith("module.") else name
            # 收集该层的 LoRA 权重
            lora_state = {f'{clean_name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)
