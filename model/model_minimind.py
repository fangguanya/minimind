# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
"""
【MiniMind 模型配置类】

这个文件定义了 MiniMind 语言模型的完整架构，包含以下核心组件:

1. MiniMindConfig: 模型配置类，定义所有超参数
2. RMSNorm: 均方根归一化层，比 LayerNorm 更高效
3. RoPE: 旋转位置编码，编码 token 的位置信息
4. Attention: 多头注意力机制，支持 GQA 和 Flash Attention
5. FeedForward: 前馈网络，使用 SwiGLU 激活函数
6. MoE: 混合专家系统，稀疏激活提升模型容量
7. MiniMindBlock: Transformer 基本块
8. MiniMindForCausalLM: 完整的因果语言模型

【学习路径建议】
1. 先理解 Config 类的各个参数含义
2. 学习 RMSNorm 和 RoPE 的数学原理
3. 深入理解 Attention 机制
4. 学习 SwiGLU 和 MoE
5. 最后理解完整的模型组装
"""

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    """
    MiniMind 模型配置类
    
    【作用】存储模型的所有超参数，方便模型创建、保存和加载
    
    【关键参数解释】
    - hidden_size: 隐藏层维度，决定模型"宽度"(512/768)
    - num_hidden_layers: Transformer 层数，决定模型"深度"(8/16)
    - num_attention_heads: 注意力头数，多头注意力从不同角度理解输入
    - num_key_value_heads: GQA 中 KV 头数，通过共享 KV 减少计算量
    - vocab_size: 词表大小 (6400)
    - rope_theta: RoPE 基础频率，影响位置编码的波长
    - use_moe: 是否使用混合专家 (MoE) 架构
    """
    model_type = "minimind"

    def __init__(
            self,
            # ========== 基础参数 ==========
            dropout: float = 0.0,              # Dropout 概率，训练时防止过拟合
            bos_token_id: int = 1,             # 序列开始 token 的 ID (Beginning Of Sequence)
            eos_token_id: int = 2,             # 序列结束 token 的 ID (End Of Sequence)
            hidden_act: str = 'silu',          # 激活函数，SiLU = x * sigmoid(x)，也叫 Swish
            hidden_size: int = 512,            # 隐藏层维度，决定模型容量
            intermediate_size: int = None,    # FFN 中间层维度，默认约 2.67 * hidden_size
            max_position_embeddings: int = 32768,  # 最大序列长度
            num_attention_heads: int = 8,      # 注意力头数量
            num_hidden_layers: int = 8,        # Transformer 层数
            num_key_value_heads: int = 2,      # GQA 中 KV 头数量 (小于 Q 头数可节省内存)
            vocab_size: int = 6400,            # 词表大小
            rms_norm_eps: float = 1e-05,       # RMSNorm 的 epsilon，防止除零
            rope_theta: int = 1000000.0,       # RoPE 基础频率，越大支持越长序列
            inference_rope_scaling: bool = False,  # 是否启用 RoPE 外推 (扩展上下文长度)
            flash_attn: bool = True,           # 是否使用 Flash Attention (更快更省内存)
            ####################################################
            # MoE (混合专家) 相关配置
            # 当 use_moe=False 时，以下参数无效
            ####################################################
            use_moe: bool = False,             # 是否使用 MoE 架构
            num_experts_per_tok: int = 2,      # 每个 token 激活的专家数量
            n_routed_experts: int = 4,         # 路由专家总数
            n_shared_experts: int = 1,         # 共享专家数量 (所有 token 都会经过)
            scoring_func: str = 'softmax',     # 门控评分函数
            aux_loss_alpha: float = 0.01,      # 辅助损失权重 (用于负载均衡)
            seq_aux: bool = True,              # 是否在序列级别计算辅助损失
            norm_topk_prob: bool = True,       # 是否归一化 top-k 概率
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings = 32768
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    """
    RMS Layer Normalization (均方根层归一化)
    
    【背景】
    传统的 LayerNorm 需要计算均值和方差，而 RMSNorm 只计算均方根，更高效。
    
    【数学原理】
    标准 LayerNorm: y = (x - μ) / σ * γ + β，其中 μ=mean(x), σ=std(x)
    RMSNorm: y = x / RMS(x) * γ，其中 RMS(x) = sqrt(mean(x²) + ε)
    
    【优势】
    1. 计算更快 (不需要计算均值)
    2. 参数更少 (没有偏置项 β)
    3. 实际效果与 LayerNorm 相当
    
    【参数】
    - dim: 归一化维度 (通常是 hidden_size)
    - eps: 防止除零的小常数 (1e-5)
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # weight 是可学习的缩放参数 γ，初始化为全 1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        计算 RMS 归一化: x / sqrt(mean(x²) + eps)
        
        torch.rsqrt() 是 1/sqrt() 的高效实现
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 转 float32 计算以保持数值稳定，然后转回原类型
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    """
    预计算 RoPE (Rotary Position Embedding) 的频率
    
    【RoPE 原理】
    RoPE 通过旋转向量来编码位置信息，具有以下优点:
    1. 相对位置信息: 只关心 token 之间的相对距离
    2. 衰减性: 距离越远相关性越弱 (通过旋转实现)
    3. 可外推性: 可处理训练时未见过的长度
    
    【数学公式】
    对于位置 m 和维度 d:
        θ_d = 1 / (base^(2d/dim))  # 每个维度有不同的频率
        旋转角度 = m * θ_d         # 位置越大，旋转越多
    
    【参数】
    - dim: 每个注意力头的维度
    - end: 预计算的最大位置数 (32768)
    - rope_base: 基础频率 (1e6)
    - rope_scaling: YaRN 外推配置
    
    【返回】
    - freqs_cos: [end, dim] 的余弦频率表
    - freqs_sin: [end, dim] 的正弦频率表
    """
    # 计算基础频率 θ_d = 1 / (base^(2d/dim))
    # 例如 dim=64, base=1e6: θ = [1, 1/1e6^(2/64), 1/1e6^(4/64), ...]
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    
    # YaRN 外推: 当需要处理超过训练长度的序列时使用
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            # YaRN 公式: f'(i) = f(i) * ((1-γ) + γ/s), γ∈[0,1] 是线性 ramp
            # 低频维度使用插值，高频维度保持不变
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    # 生成位置索引 [0, 1, 2, ..., end-1]
    t = torch.arange(end, device=freqs.device)
    # 外积得到 [end, dim//2] 的频率矩阵: 每个位置在每个维度的旋转角度
    freqs = torch.outer(t, freqs).float()
    # 计算 cos 和 sin，并拼接 (因为要同时应用于向量的两半)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    应用旋转位置编码到 Query 和 Key
    
    【数学原理】
    RoPE 将向量视为复数，通过旋转编码位置:
    
    对于向量 x = [x_0, x_1, ..., x_{d-1}]:
    1. 分成两半: x_first = x[:d/2], x_second = x[d/2:]
    2. rotate_half(x) = [-x_second, x_first]
    3. rotated(x) = x * cos(mθ) + rotate_half(x) * sin(mθ)
    
    这等价于在复数平面上将 (x_i + i*x_{i+d/2}) 旋转 mθ_i 角度
    
    【参数】
    - q, k: Query 和 Key 张量 [batch, seq_len, heads, head_dim]
    - cos, sin: 预计算的频率 [seq_len, head_dim]
    """
    def rotate_half(x):
        """将向量的前后两半交换并取负: [a,b,c,d] -> [-c,-d,a,b]"""
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # 旋转公式: x * cos + rotate_half(x) * sin
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复 Key/Value 张量以匹配 Query 头数 (用于 GQA)
    
    【GQA (Grouped Query Attention) 原理】
    - 标准 MHA: Q_heads = K_heads = V_heads = 8
    - GQA: Q_heads = 8, K_heads = V_heads = 2
    - 每个 KV 头被 4 个 Q 头共享
    - 优势: 减少 4 倍 KV Cache 内存，推理更快
    
    【参数】
    - x: [batch, seq_len, num_kv_heads, head_dim]
    - n_rep: 每个 KV 头需要复制的次数
    
    【返回】
    - [batch, seq_len, num_kv_heads * n_rep, head_dim]
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # 扩展维度 [B,L,KV,1,D] -> [B,L,KV,n_rep,D] -> [B,L,KV*n_rep,D]
    return (
        x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    多头注意力层 (支持 GQA 和 Flash Attention)
    
    【注意力机制核心公式】
    Attention(Q, K, V) = softmax(Q @ K^T / √d_k) @ V
    
    其中:
    - Q (Query): "我在找什么" - 当前位置的查询向量
    - K (Key): "我有什么" - 所有位置的键向量
    - V (Value): "我的内容" - 所有位置的值向量
    - d_k: Key 的维度，用于缩放防止梯度消失
    
    【多头注意力】
    将 Q,K,V 分成多个"头"，每个头独立计算注意力:
    - 不同的头关注不同类型的模式
    - 增加模型的表达能力
    - 计算可以并行化
    
    【GQA vs MHA】
    - MHA: 每个 Q 头都有对应的 K,V 头 (8Q, 8K, 8V)
    - GQA: 多个 Q 头共享 K,V 头 (8Q, 2K, 2V)
    - GQA 优势: 减少 KV Cache 内存，推理更快
    
    【Flash Attention】
    - 标准注意力需要 O(n²) 内存存储注意力矩阵
    - Flash Attention 分块计算，只需 O(n) 内存
    - 同时由于更好的内存访问模式，速度更快
    """
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        # 确定 KV 头数 (GQA 配置)
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        
        self.n_local_heads = args.num_attention_heads        # Q 头数
        self.n_local_kv_heads = self.num_key_value_heads     # KV 头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # GQA 重复因子
        self.head_dim = args.hidden_size // args.num_attention_heads  # 每头维度
        
        # 投影层: 将 hidden_size 映射到各个头
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        # 检查是否支持 Flash Attention (需要 PyTorch >= 2.0)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # (cos, sin)
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        """
        前向传播
        
        【参数】
        - x: 输入张量 [batch, seq_len, hidden_size]
        - position_embeddings: (cos, sin) RoPE 位置编码
        - past_key_value: KV Cache，用于增量解码
        - use_cache: 是否返回 KV Cache
        - attention_mask: 注意力掩码
        
        【返回】
        - output: [batch, seq_len, hidden_size]
        - past_kv: 更新后的 KV Cache
        """
        bsz, seq_len, _ = x.shape
        
        # 步骤1: 线性投影得到 Q, K, V
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # 重塑为多头格式: [B, L, heads, head_dim]
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 步骤2: 应用 RoPE 位置编码
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # 步骤3: KV Cache 处理 (用于推理时的增量解码)
        # 将新的 K,V 拼接到之前的 cache 上
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 步骤4: GQA 处理 - 复制 KV 以匹配 Q 头数
        # 转置为 [B, heads, L, head_dim] 以便矩阵乘法
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        # 步骤5: 计算注意力
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            # Flash Attention: 更快更省内存
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # 标准注意力: Q @ K^T / sqrt(d_k)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # 因果掩码: 上三角设为 -inf，防止看到未来 token
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)

            # 可选的额外注意力掩码 (如 padding mask)
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            # Softmax 归一化得到注意力权重
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            # 注意力加权求和: weights @ V
            output = scores @ xv

        # 步骤6: 合并多头并投影回原维度
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    """
    前馈神经网络 (SwiGLU 变体)
    
    【传统 FFN】
    FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
    
    【SwiGLU FFN】(LLaMA 使用的变体)
    FFN(x) = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down
    
    其中:
    - SiLU(x) = x * sigmoid(x)，也叫 Swish 激活函数
    - ⊙ 表示逐元素乘法 (Hadamard product)
    - W_gate 提供门控信号，W_up 提供内容
    
    【为什么使用 SwiGLU】
    1. SiLU 比 ReLU 更平滑，梯度更稳定
    2. 门控机制让网络更有选择性
    3. 实践中表现更好 (LLaMA、GPT-4 都在用)
    
    【维度变化】
    hidden_size -> intermediate_size -> hidden_size
    例如: 512 -> 1376 -> 512 (约 2.67 倍扩展)
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # 计算中间层维度，默认约 2.67 倍 hidden_size
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            # 对齐到 64 的倍数，提高硬件计算效率
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        
        # 三个投影层
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)  # 门控
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)  # 下投影
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)    # 上投影
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]  # SiLU 激活函数

    def forward(self, x):
        # SwiGLU: down_proj(act(gate_proj(x)) * up_proj(x))
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    """
    MoE 门控网络 (Mixture of Experts Gate)
    
    【MoE 原理】
    MoE 包含多个"专家"(Expert)，每个专家是一个独立的 FFN。
    门控网络根据输入决定激活哪些专家。
    
    【工作流程】
    1. 门控网络计算每个专家的分数: scores = softmax(x @ W)
    2. 选择 top-k 个分数最高的专家
    3. 每个选中的专家处理输入，输出加权求和
    
    【优势】
    1. 稀疏激活: 每次只激活部分专家，计算量可控
    2. 大容量: 可以有很多专家，模型容量大
    3. 专业化: 不同专家学习不同类型的知识
    
    【辅助损失 (Auxiliary Loss)】
    为防止"专家崩塌"(某些专家过度使用，其他闲置):
    aux_loss = α * Σ(frequency_i * routing_prob_i)
    这鼓励负载均衡，让所有专家都被使用
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok      # 每个 token 激活的专家数
        self.n_routed_experts = config.n_routed_experts  # 总专家数

        self.scoring_func = config.scoring_func      # 评分函数 (softmax)
        self.alpha = config.aux_loss_alpha           # 辅助损失权重
        self.seq_aux = config.seq_aux                # 是否使用序列级辅助损失

        self.norm_topk_prob = config.norm_topk_prob  # 是否归一化 top-k 概率
        self.gating_dim = config.hidden_size
        # 门控权重矩阵: [n_experts, hidden_size]
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Kaiming 初始化"""
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = scores.new_zeros(1).squeeze()
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    """
    MoE 前馈层 (Mixture of Experts Feed-Forward)
    
    【结构】
    - n_routed_experts 个路由专家 (稀疏激活，每次只用 top-k 个)
    - n_shared_experts 个共享专家 (总是激活，所有 token 都经过)
    - MoEGate 门控网络 (决定每个 token 用哪些专家)
    
    【工作流程】
    输入 x -> 门控选择 top-k 专家 -> 各专家处理 -> 加权求和 -> 加上共享专家输出
    
    【例子】
    - 4 个路由专家, top-k=2: 每个 token 激活 2 个专家
    - 1 个共享专家: 所有 token 都会经过
    - 总计算量约等于 3 个 FFN (2 + 1)
    - 但模型容量是 5 个 FFN (4 + 1)
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 创建路由专家列表
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        # 门控网络
        self.gate = MoEGate(config)
        # 共享专家 (可选)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=x.dtype)
            for i, expert in enumerate(self.experts):
                expert_out = expert(x[flat_topk_idx == i])
                if expert_out.shape[0] > 0: y[flat_topk_idx == i] = expert_out.to(y.dtype)
                else: y[flat_topk_idx == i] = expert_out.to(y.dtype) + 0 * sum(p.sum() for p in expert.parameters())
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    """
    Transformer 基本块 (Pre-LN 结构)
    
    【结构图】
    输入 x
        │
        ├──────────────────────┐
        ▼                      │ (残差连接)
    RMSNorm                    │
        │                      │
    Self-Attention             │
        │                      │
        ├──────────────────────┘
        ▼
    相加 (x + attention_output)
        │
        ├──────────────────────┐
        ▼                      │ (残差连接)
    RMSNorm                    │
        │                      │
    FFN/MoE                    │
        │                      │
        ├──────────────────────┘
        ▼
    相加 (x + ffn_output)
        │
        ▼
    输出
    
    【Pre-LN vs Post-LN】
    - Post-LN (原始 Transformer): Norm 在残差连接之后
    - Pre-LN (现代做法): Norm 在残差连接之前，训练更稳定
    
    【残差连接的作用】
    1. 缓解梯度消失: 梯度可以直接流过
    2. 信息高速公路: 浅层特征可以直接传到深层
    3. 使深层网络可训练
    """
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # 自注意力层
        self.self_attn = Attention(config)
        self.layer_id = layer_id
        
        # 归一化层 (Pre-LN: 在 attention/ffn 之前进行归一化)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # FFN 或 MoE
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # 第一个残差块: Self-Attention
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states = hidden_states + residual  # 残差连接
        
        # 第二个残差块: FFN/MoE
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """
    MiniMind 基础模型 (不含 LM Head)
    
    【整体结构】
    Token IDs → Embedding → [Transformer Block × N] → Final Norm → Hidden States
    
    【组件说明】
    1. embed_tokens: 词嵌入层，将 token ID 映射为向量
       - 输入: [batch, seq_len] 的整数 ID
       - 输出: [batch, seq_len, hidden_size] 的向量
    
    2. layers: N 个 Transformer Block 堆叠
       - 每层包含: Attention + FFN/MoE
       - 深度决定模型的表达能力
    
    3. norm: 最终的 RMSNorm 归一化
       - 在输出前进行归一化，稳定输出分布
    
    4. freqs_cos/freqs_sin: 预计算的 RoPE 位置编码
       - 注册为 buffer，不参与训练但会保存
    """
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        
        # 词嵌入: 将 token ID 映射为 hidden_size 维向量
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        
        # 堆叠 N 个 Transformer Block
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        
        # 最终归一化层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算 RoPE 频率并注册为 buffer
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    MiniMind 因果语言模型 (完整模型)
    
    【结构】
    Token IDs → MiniMindModel → Hidden States → LM Head → Logits
    
    【因果语言模型 (Causal LM)】
    自回归模型，从左到右预测下一个 token:
    P(x₁, x₂, ..., xₙ) = P(x₁) × P(x₂|x₁) × ... × P(xₙ|x₁...xₙ₋₁)
    
    【LM Head 的作用】
    将隐藏状态映射回词表:
    - 输入: [batch, seq_len, hidden_size]
    - 输出: [batch, seq_len, vocab_size]
    - 每个位置输出词表大小的 logits，表示下一个 token 的概率分布
    
    【权重共享 (Weight Tying)】
    LM Head 和 Embedding 共享权重:
    - 减少参数量 (vocab_size × hidden_size)
    - 使输入输出表示一致
    - 在小模型上效果尤其好
    
    【继承关系】
    - PreTrainedModel: HuggingFace 预训练模型基类，提供保存/加载等功能
    - GenerationMixin: 提供 generate() 方法用于文本生成
    """
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        
        # 基础模型
        self.model = MiniMindModel(self.config)
        
        # 语言模型头: hidden_size → vocab_size
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        
        # 权重共享: embedding 和 lm_head 使用相同的权重矩阵
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        """
        前向传播
        
        【参数】
        - input_ids: 输入 token ID [batch, seq_len]
        - attention_mask: 注意力掩码 (1=有效, 0=padding)
        - past_key_values: KV Cache，用于增量解码
        - use_cache: 是否返回 KV Cache
        - logits_to_keep: 只保留最后 N 个位置的 logits (节省内存)
        
        【返回】
        CausalLMOutputWithPast 对象，包含:
        - logits: 每个位置的 token 概率分布 [batch, seq_len, vocab_size]
        - past_key_values: KV Cache
        - hidden_states: 隐藏状态
        - aux_loss: MoE 辅助损失
        
        【训练 vs 推理】
        训练时: logits_to_keep=0，返回所有位置的 logits
        推理时: logits_to_keep=1，只需要最后一个位置的 logits
        """
        # 通过基础模型得到隐藏状态
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        
        # 只保留需要的位置 (用于高效推理)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        
        # LM Head: hidden_states → logits
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        # 构建输出对象
        output = CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        output.aux_loss = aux_loss
        return output
