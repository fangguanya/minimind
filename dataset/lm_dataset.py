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
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        # 构建输入输出对
        # 语言模型预测下一个 token，所以:
        # X = tokens[:-1]  输入
        # Y = tokens[1:]   标签
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐到 Y
        
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
    
    其中:
    - y_w: chosen (好回复)
    - y_l: rejected (差回复)
    - π: 策略模型
    - π_ref: 参考模型
    - β: 温度参数
    
    通过这个损失，模型学习提高好回复的概率，降低差回复的概率。
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
