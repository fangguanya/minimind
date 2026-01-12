"""
================================================================================
语言模型数据集实现
================================================================================

本文件实现了多种用于语言模型训练的数据集类：

1. PretrainDataset: 预训练数据集
   - 用于从头训练语言模型
   - 数据格式: {"text": "文本内容"}

2. SFTDataset: 监督微调数据集
   - 用于对话能力微调
   - 数据格式: {"conversations": [{"role": "user/assistant", "content": "..."}]}

3. DPODataset: 直接偏好优化数据集
   - 用于 RLHF 中的偏好学习
   - 数据格式: {"chosen": [...], "rejected": [...]}

4. RLAIFDataset: 强化学习数据集
   - 用于 PPO/GRPO/SPO 等强化学习训练
   - 数据格式: {"conversations": [...]}

【数据处理核心概念】

1. Token 化
   - 将文本转换为数字 ID 序列
   - 使用 Tokenizer 完成

2. Loss Mask
   - 指示哪些位置需要计算损失
   - 对于 SFT，只在 assistant 回复部分计算损失
   - 这防止模型学习"鹦鹉学舌"用户输入

3. 输入输出对齐
   - 语言模型是预测下一个 token
   - X = tokens[:-1] (输入)
   - Y = tokens[1:]  (标签)

作者：MiniMind 团队
"""

from torch.utils.data import Dataset
import torch
import os
from datasets import load_dataset

# 禁用 tokenizers 的并行处理（避免 fork 警告）
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    """
    预训练数据集
    
    【用途】
    用于语言模型的从头预训练，学习语言的基本规律和知识。
    
    【数据格式】
    JSONL 文件，每行一个 JSON 对象:
    {"text": "这是一段预训练文本..."}
    
    【训练目标】
    预测文本中每个位置的下一个 token:
    P(token_i | token_{1:i-1})
    
    对所有 token 都计算损失（除了 padding）。
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        """
        初始化预训练数据集
        
        参数:
            data_path: JSONL 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用 HuggingFace datasets 加载数据
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, index):
        """
        获取单个样本
        
        返回:
            X: 输入 token 序列 [max_length-1]
            Y: 目标 token 序列 [max_length-1]（X 向后移一位）
            loss_mask: 损失掩码 [max_length-1]（非 padding 位置为 1）
        """
        sample = self.samples[index]

        # Token 化文本
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',      # 填充到 max_length
            truncation=True,           # 超长截断
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()  # [max_length]
        
        # 创建损失掩码：非 padding 位置为 1
        #
        # 【loss_mask vs causal_mask 的区别】
        #
        # 1. Causal Mask（因果掩码）- 在模型的 Attention 层实现
        #    位置: model/model_minimind.py 的 Attention 类
        #    作用: 控制模型在计算时"能看到哪些 token"
        #    实现: 使用 torch.triu 生成上三角矩阵，填充 -∞
        #
        #    Causal Mask 矩阵示例（4x4）:
        #            A    B    C    D   (Key)
        #        A [ 0   -∞   -∞   -∞ ]  ← 位置0只能看A
        #        B [ 0    0   -∞   -∞ ]  ← 位置1能看A,B
        #        C [ 0    0    0   -∞ ]  ← 位置2能看A,B,C
        #        D [ 0    0    0    0 ]  ← 位置3能看A,B,C,D
        #      (Query)
        #    上三角为 -∞，经过 softmax 后变成 0，即"看不到"未来的 token
        #
        # 2. Loss Mask（损失掩码）- 在数据集这里定义
        #    作用: 控制"哪些位置的预测误差要计入损失"
        #    - PretrainDataset: 非 padding 位置为 1（所有真实 token 都学）
        #    - SFTDataset: 只有 assistant 回复部分为 1（只学回复）
        #
        # 两者是独立的:
        # - Causal mask 保证因果性（不偷看未来），在模型内部生效
        # - Loss mask 决定关心哪些位置的预测，在计算损失时使用
        #
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        # 构建输入输出对
        # 语言模型预测下一个 token，所以:
        # X = tokens[:-1]  输入
        # Y = tokens[1:]   标签
        #
        # 【切片逻辑详解】
        # 假设 input_ids = [A, B, C, D, E]（5个token）：
        #   X = input_ids[:-1] = [A, B, C, D]  → 模型输入（去掉最后一个）
        #   Y = input_ids[1:]  = [B, C, D, E]  → 预测目标（去掉第一个）
        #
        # 【训练时 vs 推理时的区别】
        #
        # 训练时（一次前向传播，并行计算所有位置的损失）：
        #   位置0: 模型只能看到 [A]          → 预测 B
        #   位置1: 模型只能看到 [A, B]       → 预测 C
        #   位置2: 模型只能看到 [A, B, C]    → 预测 D
        #   位置3: 模型只能看到 [A, B, C, D] → 预测 E
        #
        #   通过 causal mask（因果掩码）确保每个位置只能看到它之前的 token
        #   一次前向传播同时计算所有位置的预测损失，效率很高
        #
        # 推理/生成时（逐个生成）：
        #   输入 "今天天气" → 预测下一个 token → "很"
        #   输入 "今天天气很" → 预测 → "好"
        #   ...逐个生成，直到结束
        #
        # 即：给定 token_i，模型学习预测 token_{i+1}
        #
        # 【torch.tensor 的作用】
        # 将 Python 列表转换为 PyTorch 张量 (Tensor)
        # - 神经网络只能处理 Tensor，不能直接处理 Python 列表
        # - Tensor 支持 GPU 加速和自动求导
        # - dtype=torch.long 指定为 64 位整数（适合存储 token ID）
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐到 Y（计算预测位置的损失）
        
        return X, Y, loss_mask


class SFTDataset(Dataset):
    """
    监督微调 (Supervised Fine-Tuning) 数据集
    
    【用途】
    用于训练模型的对话能力，学习如何根据用户指令生成回复。
    
    【数据格式】
    JSONL 文件，每行一个对话:
    {
        "conversations": [
            {"role": "system", "content": "你是一个助手"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮助你的？"}
        ]
    }
    
    【训练目标】
    只在 assistant 回复部分计算损失。
    这样模型只学习"如何回复"，而不是"如何复述用户输入"。
    
    【为什么只计算 assistant 部分的损失】
    1. 用户输入是给定的上下文，模型只需理解，不需要生成
    2. 如果对用户输入也计算损失，模型会学习复述输入
    3. 只对 assistant 部分计算损失，模型专注于生成高质量回复
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        """
        初始化 SFT 数据集
        
        参数:
            jsonl_path: JSONL 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        
        # 预计算 assistant 开始和结束的 token 序列
        # 用于定位 assistant 回复的位置
        # 格式: <|im_start|>assistant
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        # 格式: <|im_end|>
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, cs):
        """
        创建对话提示
        
        使用 tokenizer 的 chat_template 将对话格式化为训练文本。
        
        参数:
            cs: 对话列表 [{"role": "...", "content": "..."}, ...]
        
        返回:
            格式化后的文本字符串
        
        【apply_chat_template 是干嘛的？】
        
        核心作用: 把对话列表转换成模型能理解的格式化文本
        
        输入（对话列表）:
        [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮助你的？"}
        ]
        
        输出（格式化文本）:
        <|im_start|>user
        你好<|im_end|>
        <|im_start|>assistant
        你好！有什么可以帮助你的？<|im_end|>
        
        【为什么需要这个？】
        
        1. 模型只认识纯文本，不认识JSON
           模型输入必须是连续文本，不能是字典列表
        
        2. 需要特殊标记区分角色
           没有标记: "你好你好！有什么..."（分不清谁说的）
           有标记: "<|im_start|>user\n你好<|im_end|>..."（清晰区分）
        
        3. 不同模型有不同格式
           ChatML: <|im_start|>user\n内容<|im_end|>（MiniMind用）
           Llama2: [INST] 内容 [/INST]
           Alpaca: ### Instruction:\n内容\n### Response:
           apply_chat_template 自动处理这些差异！
        
        【参数说明】
        tokenize=False: 返回字符串（而不是token ID）
        add_generation_prompt=False: 不添加 <|im_start|>assistant（训练时不需要）
        add_generation_prompt=True: 添加（推理时让模型开始生成）
        """
        messages = cs.copy()
        # 检查是否有工具调用 (function calling)
        tools = cs[0]["functions"] if (cs and cs[0]["role"] == "system" and cs[0].get("functions")) else None
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_loss_mask(self, input_ids):
        """
        生成损失掩码
        
        只对 assistant 回复部分标记为需要计算损失。
        
        【算法】
        1. 找到每个 <|im_start|>assistant 的位置
        2. 找到对应的 <|im_end|> 位置
        3. 在 assistant 回复范围内设置 mask=1
        
        参数:
            input_ids: token ID 列表
        
        返回:
            loss_mask: 0/1 列表，1 表示需要计算损失
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 检查是否是 assistant 开始标记
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                # assistant 内容开始位置（跳过开始标记）
                start = i + len(self.bos_id)
                end = start
                # 找到结束标记
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 标记 assistant 回复部分（包含结束标记）
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                # 移动到下一个位置
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        """
        获取单个样本
        
        返回:
            X: 输入 token 序列
            Y: 目标 token 序列
            loss_mask: 只在 assistant 回复部分为 1
        """
        sample = self.samples[index]
        
        # 使用 chat template 格式化对话
        prompt = self.create_chat_prompt(sample['conversations'])
        
        # Token 化
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        
        # 填充到 max_length
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        
        # 生成损失掩码
        loss_mask = self.generate_loss_mask(input_ids)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐到预测位置
        
        return X, Y, loss_mask


class DPODataset(Dataset):
    """
    DPO (Direct Preference Optimization) 数据集
    
    【用途】
    用于偏好学习，让模型学习区分好的回复和差的回复。
    
    【数据格式】
    JSONL 文件，每行包含:
    {
        "chosen": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "好回复"}],
        "rejected": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "差回复"}]
    }
    
    【DPO 原理】
    DPO 损失函数:
    L = -log(σ(β * (log π(y_w|x) - log π(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))))
    
    【符号说明】
    - x: 用户输入（prompt）
    - y_w: winner，chosen（好回复）
    - y_l: loser，rejected（差回复）
    - π: 当前训练的策略模型（policy）
    - π_ref: 参考模型（冻结的初始模型）
    - β: 温度参数，控制偏离参考模型的程度（典型值 0.1~0.5）
    - σ: sigmoid 函数，σ(x) = 1/(1+e^(-x))
    
    【为什么用 Sigmoid？】
    核心目的: 把"偏好差距"（任意实数）转换成"概率"（0到1）
    - Δ=5  → σ(5)≈0.993  → 模型非常偏好好回复
    - Δ=0  → σ(0)=0.5    → 模型无法区分好坏
    - Δ=-5 → σ(-5)≈0.007 → 模型错误偏好差回复
    
    Sigmoid 优点:
    - 概率解释清晰（输出就是"正确偏好的概率"）
    - 梯度平滑，训练稳定
    - 两端饱和，对极端值不敏感（防止过拟合）
    
    Sigmoid 缺点:
    - 两端梯度很小（梯度消失），偏好已明显时学习变慢
    
    【为什么选 Sigmoid 而不是其他 [0,1] 映射？】
    
    常见的 [0,1] 映射函数:
    - Sigmoid: 1/(1+e^(-x))
    - Tanh变换: (tanh(x)+1)/2
    - 归一化: x/(1+|x|)
    - 正态CDF: Φ(x)
    - Hard Sigmoid: clip(0.2x+0.5, 0, 1)
    
    为什么 DPO 必须用 Sigmoid:
    
    1. 理论基础（Bradley-Terry 模型）
       DPO 基于偏好模型假设: P(y_w比y_l好) = σ(r(y_w) - r(y_l))
       这是 Bradley-Terry 模型的标准形式，σ 是理论推导结果，不是随便选的！
       
       【Bradley-Terry 模型推导】
       问题: 如何建模"A比B好"的概率？
       假设: 每个选项有隐藏的"实力分" r
       
       核心公式: P(A胜B) = e^r(A) / (e^r(A) + e^r(B))
       
       【这个公式怎么来的？不是拍脑袋！】
       
       直觉来源（1952年 Bradley & Terry 研究比赛排名）:
       - 如果A实力是B的2倍，A赢的概率应该是 2/(2+1)=2/3
       - 如果A实力是B的4倍，A赢的概率应该是 4/(4+1)=4/5
       - 推广: P(A胜B) = s(A) / (s(A) + s(B))，s是实力值
       
       为什么用 e^r 而不是直接用 r？
       - 实力值 s 必须是正数（概率才有意义）
       - 但我们希望 r ∈ (-∞, +∞) 方便优化
       - 解决: s = e^r，r可以任意，e^r 总是正数
       
       概率论推导（更严谨）:
       假设每个选手表现是随机的: X_A ~ 分布(均值=r(A))
       P(A胜B) = P(X_A > X_B)
       如果 X_A - X_B 服从 Logistic 分布:
       P(A胜B) = 1/(1+e^(-(r(A)-r(B)))) = σ(r(A)-r(B))
       
       历史: 1927年Zermelo(象棋)→1952年Bradley-Terry→现在用于Elo/RLHF
       
       推导 sigmoid:
       P(A>B) = e^r(A) / (e^r(A) + e^r(B))
       分子分母同除 e^r(B):
       = e^(r(A)-r(B)) / (e^(r(A)-r(B)) + 1)
       = 1 / (1 + e^(-(r(A)-r(B))))
       = σ(r(A) - r(B))  ← 这就是 sigmoid！
       
       DPO 中: r(y) = β × log(π(y)/π_ref(y)) 作为隐式奖励
       所以 P(chosen>rejected) = σ(β × (log比值差))
    
    2. 对数几率 (Log-odds) 的自然选择
       sigmoid 的逆函数是 logit: logit(p) = log(p/(1-p))
       偏好差 Δ 直接对应"对数几率"，解释性强
       
       【对数几率 vs 概率的区别】
       概率 p: 事件发生的可能性，范围 [0, 1]
       几率 odds: 发生vs不发生的比值 = p/(1-p)
         p=0.8 → odds=0.8/0.2=4（发生是不发生的4倍）
         p=0.5 → odds=1（五五开）
       对数几率 logit: 几率取对数 = log(odds)
         p=0.8 → odds=4 → logit=log(4)=1.39
         p=0.5 → odds=1 → logit=log(1)=0
       
       为什么用对数几率？
       - 概率范围 [0,1] 有边界，优化困难
       - 对数几率范围 (-∞,+∞) 无边界，好优化
       - sigmoid 就是 logit 的逆函数: p = σ(logit)
    
    3. 导数形式优美
       σ'(x) = σ(x) × (1 - σ(x))
       - x=0 时梯度最大（最需要学习的地方）
       - |x|很大时梯度小（已确定的不用学）
    
    4. 与交叉熵配合完美
       -log(σ(x)) 的梯度 = -σ(-x)，形式简洁，数值稳定
    
    其他函数的问题:
    - 正态CDF: 没有解析形式，计算慢
    - Hard Sigmoid: 不可导，梯度问题更严重
    - 归一化: 没有概率论基础，解释性差
    - Tanh变换: 可以用，但没有 Bradley-Terry 理论支持
    
    【其他选择】
    1. Hinge Loss: L = max(0, margin - Δ)
       - 优点: 梯度恒定不消失，训练更快，超过margin就停止优化
       - 缺点: 不是概率，边界处不可导
       - 用于: IPO 变体、类SVM方法
    
    2. 平方损失 (IPO): L = (Δ - margin)²
       - 优点: 可导，有明确目标值
       - 缺点: 对异常值敏感（平方放大）
    
    选择建议:
    - 通用偏好学习 → Sigmoid (DPO)，稳定、理论基础好
    - 需要快速收敛 → Hinge，梯度恒定
    - 精确控制偏好差距 → IPO，有明确目标margin
    
    【π(y|x) 条件概率】
    π(y_w|x) = 模型给定输入 x 时，生成回复 y_w 的概率
    对于自回归模型: π(y|x) = P(t1|x) × P(t2|x,t1) × ... × P(tN|x,t1...tN-1)
    log π(y|x) 就是概率取对数（通常是负数，因为概率<1）
    
    【为什么要用 log？】
    
    1. 数值稳定性（防止下溢）
       假设 100 个 token，每个概率 0.1：
       P = 0.1^100 = 10^(-100) → 太小，计算机存不下！
       log P = -100 → 正常数值，没问题
    
    2. 乘法变加法，计算更稳定
       原始: π(y|x) = P(t1) × P(t2|t1) × P(t3|t1,t2) × ...
       取log: log π = log P(t1) + log P(t2|t1) + ...
       加法比乘法快，梯度传播更稳定
    
    3. 交叉熵损失的本质
       最大化 P(正确答案) = π(y|x)
       ↓ log是单调函数
       最大化 log π(y|x)
       ↓ 加负号变损失
       最小化 -log π(y|x) ← 这就是交叉熵损失！
       
       【什么是交叉熵？】
       熵 H(P) = -Σ P(x)×log P(x)，衡量分布的"不确定性"
       交叉熵 H(P,Q) = -Σ P(x)×log Q(x)，用Q编码P的平均信息量
       
       【P 和 Q 分别是什么？】
       P = 真实分布（标签，one-hot向量）
         例: P = [0, 0, 1, 0, 0] 表示正确答案是第3类
         只有正确类别=1，其他都=0
       
       Q = 模型预测分布（softmax输出）
         例: Q = [0.1, 0.1, 0.6, 0.1, 0.1]
         模型认为: 第1类10%, 第2类10%, 第3类60%, ...
       
       【交叉熵详细计算过程】
       H(P,Q) = -Σ P(i)×log Q(i)
       = -(P[0]×log Q[0] + P[1]×log Q[1] + P[2]×log Q[2] + ...)
       = -(0×log(0.1) + 0×log(0.1) + 1×log(0.6) + 0×log(0.1) + 0×log(0.1))
       = -(0 + 0 + log(0.6) + 0 + 0)
       = -log(0.6)
       = -(-0.511)
       = 0.511
       
       因为P是one-hot，只有正确类别的项不为0！
       简化: 交叉熵 = -log(正确类别的预测概率)
       
       【损失值详细计算】（使用自然对数ln，底数e≈2.718）
       预测概率 0.9 → 损失 = -ln(0.9) = -(-0.105) = 0.105（好）
       预测概率 0.5 → 损失 = -ln(0.5) = -(-0.693) = 0.693（中）
       预测概率 0.1 → 损失 = -ln(0.1) = -(-2.303) = 2.303（差）
       
       验证: ln(0.9)≈-0.105, ln(0.5)≈-0.693, ln(0.1)≈-2.303
    
    4. 比较差异时更方便
       原始比值: π(y_w)/π(y_l) = 0.8/0.2 = 4（除法不稳定）
       用log: log π(y_w) - log π(y_l) = 1.39（减法更稳定）
       比值为1（无偏好）→ log差为0，对称且直观
       
       【log减法=1.39的详细计算】
       log π(y_w) - log π(y_l)
       = log(0.8) - log(0.2)
       = -0.223 - (-1.609)
       = -0.223 + 1.609
       = 1.386 ≈ 1.39 ✓
       
       验证: 根据log性质 log(a/b) = log(a) - log(b)
       log(0.8/0.2) = log(4) = 1.386 ✓
    
    5. 信息论意义
       -log P(x) = 事件x的"惊讶程度"（信息量）
       P=1 (必然) → -log(1) = 0 → 不惊讶
       P=0.001 (罕见) → -log(0.001) ≈ 6.9 → 很惊讶
       损失函数本质: 让模型对正确答案"不惊讶"
    
    【核心差值逻辑】
    公式可以重新整理为:
    (log π(y_w|x) - log π(y_l|x)) - (log π_ref(y_w|x) - log π_ref(y_l|x))
    
    即: 当前模型的偏好差 - 参考模型的偏好差 = 相对偏好变化
    我们希望训练后的模型比原始模型更能区分好坏回复
    
    【各层函数作用】
    1. β * (...): 温度缩放，β越大惩罚偏离越严重，模型变化越保守
    2. σ(...): 把"偏好差距"压缩到(0,1)，转换成"正确偏好的概率"
    3. -log(σ(...)): 二分类交叉熵损失
       - σ≈1（正确偏好好回复）→ -log(1)≈0（损失小）
       - σ≈0（错误偏好差回复）→ -log(0)→∞（损失大）
    
    【训练目标】
    最小化损失 L → σ变大 → 模型更偏好好回复 → 学会区分好坏
    
    【损失如何参与梯度下降】
    
    完整训练流程:
    1. 前向传播: X → 模型 → logits → log_softmax → log_probs
    2. 计算损失: L = -log σ(β × (chosen偏好 - rejected偏好))
    3. 反向传播: loss.backward() → PyTorch 自动计算 ∂L/∂θ
    4. 参数更新: optimizer.step() → θ_new = θ_old - lr × grad
    
    【前向传播各步骤详解】
    
    输入: X = [token1, token2, ..., tokenN]（token ID序列）
    
    Step 1: X → 模型
      X 通过 Embedding → Transformer层 → LM Head
      输出: logits
    
    Step 2: logits（原始分数，未归一化）
      形状: [batch, seq_len, vocab_size] 如 [1, 5, 6400]
      含义: 每个位置输出6400个分数，表示下一个词是每个词的"可能性分数"
      例如某位置: [2.1, -0.5, 3.2, 0.8, ...]（6400个数，可正可负）
    
    Step 3: softmax（转换为概率）
      公式: softmax(x_i) = e^(x_i) / Σ e^(x_j)
        i = 当前计算的位置
        j = 所有位置（用于求和）
      
      【详细计算过程】
      输入: logits = [2.1, -0.5, 3.2]
      
      第1步: 每个元素算e指数
        e^(2.1) = 8.166
        e^(-0.5) = 0.607
        e^(3.2) = 24.533
      
      第2步: 求和
        Σ = 8.166 + 0.607 + 24.533 = 33.306
      
      第3步: 每个除以总和
        位置0: 8.166/33.306 = 0.245 ≈ 0.24 ✓
        位置1: 0.607/33.306 = 0.018 ≈ 0.02 ✓
        位置2: 24.533/33.306 = 0.737 ≈ 0.74 ✓
      
      验证: 0.245+0.018+0.737 = 1.0 ✓
      
      为什么用e指数?
        原始分数可正可负 → e指数后全正 → 除以总和变概率
        分数越大 → e指数越大 → 概率越大
    
    Step 4: log_softmax（取log，数值更稳定）
      公式: log_softmax(x) = log(softmax(x))
      例如: [0.24, 0.02, 0.74] → [-1.43, -3.91, -0.30]（log概率，总是负数）
    
    Step 5: log_probs（提取目标token的log概率）
      形状: [batch, seq_len]
      用 gather 从6400个log概率中取出目标token对应的那个
      这就是模型对"正确答案"的log概率
    
    具体代码 (trainer/train_dpo.py) 与数学公式对应:
    
    # Step 1: 前向传播
    outputs = model(x)
    logits = outputs.logits
    # 数学: X → Embedding(W_embed) → Transformer → LM_Head(W_lm) → logits
    
    # Step 2: log_softmax
    log_probs = F.log_softmax(logits, dim=2)
    # 数学: log_softmax(xᵢ) = xᵢ - log(Σⱼ e^(xⱼ))
    
    # Step 3: gather提取目标token
    log_probs = torch.gather(log_probs, dim=2, index=labels)
    # 数学: 从6400个log概率中按labels索引取值
    
    # Step 4: 序列级概率（平均）
    policy_log_probs = (log_probs * mask).sum(dim=1) / seq_lengths
    # 数学: log π(y|x) = (1/N) × Σᵢ log P(tᵢ|t₁...tᵢ₋₁)
    #
    # 【什么是序列级概率？】
    # Token级: 每个位置单独的概率 [-1.2, -0.8, -1.5, -0.9]（4个数）
    # 序列级: 整个回复的平均概率 = (-1.2-0.8-1.5-0.9)/4 = -1.1（1个数）
    # DPO比较的是"整个回复"，所以要合并成一个数
    #
    # 【mask 干嘛的？只算有效部分】
    # tokens    = [你, 好, 啊, PAD, PAD]
    # mask      = [1,  1,  1,  0,   0]  ← 1=有效，0=padding
    # log_probs = [-1.2, -0.8, -1.5, -0.3, -0.1]
    # log_probs * mask = [-1.2, -0.8, -1.5, 0, 0]  ← PAD位置变0
    # sum = -3.5, seq_length = 3
    # 平均 = -3.5/3 = -1.17（只算真实token）
    #
    # 【为什么dim从2变成1？】
    # log_softmax时: [batch, seq_len, vocab] → dim=2在词表维度做softmax
    # gather之后: [batch, seq_len] → vocab维度没了！
    # sum(dim=1): 对seq_len维度求和 → [batch]
    #
    # 【seq_lengths是什么？】
    # seq_lengths = mask.sum(dim=1) = [3, 4]  ← 每个样本的有效token数
    # 不是=1！是每个样本实际有多少个非PAD的token
    
    # Step 5: 计算偏好差
    logits = (chosen_policy - reject_policy) - (chosen_ref - reject_ref)
    # 数学: log(π(w)/π_ref(w)) - log(π(l)/π_ref(l))
    #
    # 【这些变量哪来的？】
    # 数据组织: batch = [chosen_1, chosen_2, rejected_1, rejected_2]
    # 策略模型: batch → model → policy_log_probs = [-1.2, -0.8, -1.5, -1.1]
    # 参考模型: batch → ref_model → ref_log_probs = [-1.0, -0.9, -1.3, -1.0]
    # 
    # 按前后半分开:
    # chosen_policy = policy_log_probs[:half] = [-1.2, -0.8]（好回复）
    # reject_policy = policy_log_probs[half:] = [-1.5, -1.1]（差回复）
    # chosen_ref = ref_log_probs[:half] = [-1.0, -0.9]
    # reject_ref = ref_log_probs[half:] = [-1.3, -1.0]
    
    # Step 6: DPO损失
    loss = -F.logsigmoid(beta * logits)
    # 数学: L = -log(σ(β×Δ)) = log(1 + e^(-β×Δ))
    
    # Step 7: 反向传播
    scaler.scale(loss).backward()
    # 数学: 链式法则 ∂L/∂θ = ∂L/∂loss × ∂loss/∂logits × ... × ∂layer/∂θ
    #       结果存入 param.grad
    #
    # 【backward()怎么知道更新谁？不用传θ？】
    # 
    # PyTorch 的魔法：计算图 (Computational Graph)
    # 
    # 当你写 y = model(x) 时，PyTorch 自动记录:
    #   x → Embedding(W_embed) → Transformer(W_qkv) → LM_Head(W_lm) → logits → loss
    #   每个操作都记住: 输入是谁、输出是谁、怎么求导
    #
    # 模型参数在创建时就标记了 requires_grad=True:
    #   model.embedding.weight.requires_grad = True
    #   model.layers[0].wq.weight.requires_grad = True
    #   ...
    #
    # loss.backward() 做的事:
    #   1. 从 loss 开始，沿着计算图往回走
    #   2. 对每个操作，用链式法则算梯度
    #   3. 把梯度存到 param.grad
    #
    # 所以不用传θ，因为计算图已经记住了所有连接！
    
    # Step 8: 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # 数学: 如果 ||g|| > max_norm: g = g × (max_norm/||g||)
    
    # Step 9: 参数更新 (AdamW)
    scaler.step(optimizer)
    # 数学: m = β₁m + (1-β₁)g  (动量)
    #       v = β₂v + (1-β₂)g² (二阶矩)
    #       θ = θ - lr×m̂/(√v̂+ε) - λ×θ
    #
    # 【高中生版：动量和二阶矩是啥？】
    #
    # ═══ 问题1：梯度方向不稳定 ═══
    # 想象你在山上找最低点，每步看脚下坡度决定方向
    # 但地面坑坑洼洼: 第1步往左←, 第2步往右→, 第3步又往左←
    # 一直来回晃，走不远！
    #
    # ═══ 动量 (Momentum) 解决方案 ═══
    # 类比：球从山上滚下来有惯性，不会因为小坑就停
    # 
    # m = 0.9×m_old + 0.1×当前梯度
    # 
    # 含义: 90%保留之前方向（惯性），10%听当前梯度
    # 效果: 来回晃的←→←→ 会互相抵消，一致的方向会累积
    #       原本[←,→,←,→] → 平滑后[→,→,→,→]
    #
    # ═══ 问题2：不同参数需要不同步长 ═══
    # 有的参数梯度大（经常更新）→ 需要小步走
    # 有的参数梯度小（很少更新）→ 需要大步走
    #
    # ═══ 二阶矩 解决方案 ═══
    # v = 0.999×v_old + 0.001×梯度²
    # 
    # v 记录"梯度波动有多大"
    # 
    # 效果: 
    #   - 梯度大的参数 → v大 → √v大 → 实际步长小（慢点走）
    #   - 梯度小的参数 → v小 → √v小 → 实际步长大（快点走）
    #   自动给每个参数分配合适的学习率！
    #
    # ═══ 完整公式解释 ═══
    # θ = θ - lr × m̂/(√v̂+ε) - λ×θ
    #
    # 符号含义:
    # - θ: 模型参数（就是那2600万个数）
    # - lr (learning rate): 学习率，如0.0001，控制每步走多远
    # - m̂: 动量（平滑后的梯度方向）← 不是原始梯度！
    # - v̂: 二阶矩（梯度波动大小）
    # - ε (epsilon): 防止除零，如0.00000001
    # - λ (lambda): 权重衰减，如0.01（让参数不要太大）
    #
    # 分解:
    # m̂/(√v̂+ε) = 用动量方向，用二阶矩调节步长
    # lr × ...  = 乘以学习率，控制总步长
    # - λ×θ     = 权重衰减，参数越大惩罚越大（防止过拟合）
    
    自动微分链式法则:
    L = -log(σ(β×Δ))
    ∂L/∂θ = ∂L/∂Δ × ∂Δ/∂logits × ∂logits/∂θ
    PyTorch 自动完成这个链式求导，无需手动计算！
    
    【θ 是什么？】
    θ (theta) = 模型的所有可学习参数，包括:
    - Embedding层的权重矩阵
    - 每层Transformer的 Q/K/V/O 权重矩阵
    - FFN的权重矩阵
    - LM Head的权重矩阵
    - 所有的偏置项 (bias)
    
    例如 MiniMind-26M 有约2600万个参数:
    θ = [w1, w2, w3, ..., w26000000]（一个巨大的向量）
    
    【梯度下降公式各参数详解】
    θ_new = θ_old - lr × grad
    
    - θ_old: 更新前的参数值（当前权重）
    - θ_new: 更新后的参数值（新权重）
    - lr (learning_rate): 学习率，控制步长大小
      例如 lr=0.0001，每次只走一小步
    - grad = ∂L/∂θ: 损失对参数的梯度（偏导数）
      指向损失增大的方向
    
    为什么要减？
    - grad 指向损失增大的方向
    - 我们要最小化损失，所以往相反方向走
    
    【AdamW优化器】（实际使用的更复杂版本）
    θ_new = θ_old - lr × m/(√v+ε) - λ×θ_old
    
    额外参数:
    - m: 梯度的一阶矩估计（动量，平滑梯度）
    - v: 梯度的二阶矩估计（自适应学习率）
    - ε: 防止除零的小常数（如1e-8）
    - λ: 权重衰减系数（正则化，防止过拟合）
    
    【代码与训练流程的详细对应】
    
    # Step 1: 前向传播
    outputs = model(x)           # X → Embedding → Transformer → LM Head
    logits = outputs.logits      # 输出原始分数 [B, L, V]
    
    # Step 2: 计算log概率
    log_probs = F.log_softmax(logits, dim=2)  # 转换为log概率
    log_probs = torch.gather(log_probs, 2, labels)  # 提取目标token的概率
    #
    # 【gather原理：按索引取值】
    # 
    # 问题: log_probs形状[2,4,6400]，每个位置有6400个概率
    #       但我们只关心"正确答案"那个词的概率！
    #       labels = [[42,15,88,7], [33,21,5,99]] 是正确token的ID
    #
    # gather做的事（类比取书）:
    #   书架有6400本书，清单说要第42本
    #   → 取出第42本书
    #
    # 具体:
    #   log_probs[0][0] = [-2.1, -1.5, ..., -0.8(第42个), ...]
    #   labels[0][0] = 42
    #   gather后: 取出 log_probs[0][0][42] = -0.8
    #
    # 结果: [2,4,6400] → [2,4]（每个位置只剩1个数）
    
    # Step 3: 计算DPO损失
    loss = -F.logsigmoid(beta * logits)
    # 内部: 1/(1+e^(-x)) 然后取log，再加负号
    
    # Step 4: 反向传播
    scaler.scale(loss).backward()
    # 内部:
    # 1. scaler.scale: 放大loss（防止float16下溢）
    # 2. backward(): 链式法则计算 ∂L/∂θ
    #    从loss反向沿计算图，逐层计算梯度
    #    结果存在每个参数的 .grad 属性中
    
    # Step 5: 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # 内部:
    # 1. 计算总范数: ||grad|| = √(Σ grad_i²)
    # 2. 如果 ||grad|| > grad_clip:
    #    grad = grad × (grad_clip / ||grad||)
    # 作用: 防止梯度爆炸
    #
    # 【裁剪会损失梯度信息吗？】
    #
    # 确实会！但这是可接受的权衡：
    #
    # 1. 保持方向，只缩短长度
    #    原始[3,4]长度5 → 裁剪后[1.5,2]长度2.5
    #    方向完全一样，只是这一步走得保守一点
    #
    # 2. 极端梯度往往是噪声
    #    正常梯度很少突然变得很大
    #    如果出现，通常是异常样本或数值不稳定
    #    "不要太相信这个梯度"反而是对的
    #
    # 3. 不裁剪的后果更严重
    #    梯度=8000时，一步更新 θ -= 0.8
    #    参数直接跳到很远 → 模型崩溃，loss变NaN
    #
    # 4. 多步会补偿
    #    一次裁剪的影响会被之后的正常更新稀释
    #    动量机制也会平滑这些异常
    #
    # 典型值 grad_clip=1.0，正常训练中大部分时候不会触发
    
    # Step 6: 参数更新
    scaler.step(optimizer)
    # 内部 (AdamW):
    # 1. 读取每个参数的 .grad
    # 2. 更新动量 m 和 v
    # 3. 计算: Δθ = lr × m / (√v + ε)
    # 4. 权重衰减: θ = θ - λ × θ
    # 5. 更新参数: θ = θ - Δθ
    
    # Step 7: 清零梯度
    optimizer.zero_grad()
    # 将所有 .grad 清零，为下一次迭代准备
    """
    def __init__(self, file_path, tokenizer, max_length=4096):
        """
        初始化 DPO 数据集
        
        参数:
            file_path: JSONL 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        # assistant 开始和结束标记
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids
        
        self.data = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        获取单个样本
        
        返回一个字典，包含:
        - x_chosen, y_chosen, mask_chosen: 好回复的输入、标签、掩码
        - x_rejected, y_rejected, mask_rejected: 差回复的输入、标签、掩码
        """
        item = self.data[index]
        chosen = item['chosen']      # 好回复的对话
        rejected = item['rejected']  # 差回复的对话
        
        # 使用 chat template 格式化
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        
        # Token 化
        # 【tokenizer 是干嘛的？】
        # 模型只认识数字，不认识文字！
        # tokenizer = 文本 ↔ 数字 的翻译器
        # 
        # 输入: "你好啊"
        # 输出: {'input_ids': [42, 15, 88, 0, 0, ...], 
        #        'attention_mask': [1, 1, 1, 0, 0, ...]}
        #
        # input_ids = token ID 序列（每个数字对应词表中的一个词）
        # attention_mask = 哪些是真实token(1)，哪些是padding(0)
        #
        # 【为什么两个 prompt 都要 tokenize？】
        # DPO 需要比较两个回复的概率，所以都要转成数字
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        # 生成损失掩码
        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        
        # 构建输入输出对
        # 
        # 【为什么区分 chosen 和 rejected？】
        # DPO 训练需要同时计算两个回复的概率:
        #   - chosen（好回复）: 模型应该更偏好这个
        #   - rejected（差回复）: 模型应该不偏好这个
        #
        # 训练时两部分拼成一个 batch 一起过模型:
        #   batch = [chosen_1, chosen_2, ..., rejected_1, rejected_2, ...]
        # 然后分别计算概率，用 DPO 公式: L = -log σ(β × (好-差))
        #
        # 【mask 为什么用 [1:] 而不是 [:-1]？】
        #
        # 对齐问题！mask 要和 y 对齐，不是和 x 对齐
        #
        # 原始:     [A, B, C, D, E]    长度5
        # x[:-1]:   [A, B, C, D]       长度4（输入）
        # y[1:]:    [B, C, D, E]       长度4（标签）
        # mask[1:]: [0, 1, 1, 1]       长度4（和y对齐！）
        #
        # mask 标记的是"y的哪些位置要算损失"
        # y 从位置1开始，所以 mask 也从位置1开始
        #
        # 【Loss Mask vs Causal Mask（两种完全不同的mask！）】
        #
        # ┌─────────────────────────────────────────────────────────┐
        # │ Causal Mask（因果掩码）- 在 Attention 层                 │
        # │ 位置: model/model_minimind.py                           │
        # │ 值: 0=可看, -∞=看不见（softmax后变0）                    │
        # │ 用途: 防止模型"偷看"未来的token                          │
        # └─────────────────────────────────────────────────────────┘
        #
        # ┌─────────────────────────────────────────────────────────┐
        # │ Loss Mask（损失掩码）- 在这里！                          │
        # │ 值: 1=要算损失, 0=不算损失                               │
        # │ 用途: 只学习"回复部分"，忽略"用户问题部分"               │
        # │ 计算: loss = (log_prob × mask).sum() / mask.sum()       │
        # └─────────────────────────────────────────────────────────┘
        #
        # 为什么设计不同？
        # - Causal Mask 配合 softmax: softmax(-∞)=0
        # - Loss Mask 是简单乘法: ×0 就没了
        #
        # 例: log_prob=[-1.2,-0.8,-0.5,-1.1], mask=[0,1,1,1]
        # 结果: (0 + -0.8 + -0.5 + -1.1) / 3 = -0.8
        #
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        """
        生成损失掩码（与 SFTDataset 相同的逻辑）
        
        只在 assistant 回复部分计算损失。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    """
    RLAIF (Reinforcement Learning from AI Feedback) 数据集
    
    【用途】
    用于 PPO/GRPO/SPO 等强化学习训练。
    提供 prompt，让模型生成回复，然后通过 reward model 评分。
    
    【数据格式】
    JSONL 文件，每行包含对话:
    {
        "conversations": [
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "参考答案（可选）"}
        ]
    }
    
    【与 SFTDataset 的区别】
    - SFTDataset: 返回完整对话，用于监督学习
    - RLAIFDataset: 返回 prompt 和参考答案，用于强化学习
      模型需要根据 prompt 生成回复，然后评估回复质量
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        """
        初始化 RLAIF 数据集
        
        参数:
            jsonl_path: JSONL 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        
        # assistant 标记
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """
        创建对话提示（只包含到 user 的问题）
        
        用于强化学习：提供上下文，让模型生成回复
        
        参数:
            conversations: 对话列表
        
        返回:
            prompt: 格式化的提示文本
            answer: 参考答案（如果有的话）
        """
        messages = []
        answer = ''
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
            answer = turn['content']  # 最后一个回复作为参考答案
        
        # 使用 chat template 格式化，添加生成提示
        # add_generation_prompt=True 会添加 <|im_start|>assistant\n
        return self.tokenizer.apply_chat_template(
            messages[:-1],  # 不包含最后的 assistant 回复
            tokenize=False,
            add_generation_prompt=True
        ), answer

    def __getitem__(self, index):
        """
        获取单个样本
        
        返回:
            prompt: 格式化的提示文本（用于模型生成）
            answer: 参考答案（用于评估或辅助训练）
        """
        sample = self.samples[index]
        prompt, answer = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': answer
        }


if __name__ == "__main__":
    # 测试代码
    pass
