"""
================================================================================
                    MiniMind GRPO 训练脚本 (Group Relative Policy Optimization)
================================================================================

【什么是 GRPO】
GRPO 是 DeepSeek-R1 提出的强化学习方法，比 PPO 更简单高效:
- 不需要 Critic 模型估计价值
- 通过组内相对比较计算优势 (Advantage)
- 直接使用 Reward Model 的打分作为奖励

【形象比喻】
想象一个老师在批改作文:
- PPO: 老师对每篇作文打绝对分 (需要统一标准)
- GRPO: 老师对同一组作文打相对分 (组内比较更容易)

GRPO 的核心是"比较":
- 同一个问题，生成 8 个回复
- 看哪个比平均水平好/差
- 好的加分，差的减分

【GRPO 训练流程】
1. 对每个 prompt 生成 N 个回复 (num_generations=8)
2. 用 Reward Model 对每个回复打分
3. 在组内计算相对优势:
   advantage = (reward - group_mean) / group_std
4. 用 Policy Gradient 更新模型

【GRPO 损失公式】
L = -E[exp(log π - log π_old) * A - β * KL(π || π_ref)]

展开:
L = -Σ (ratio * advantage - β * KL)

其中:
- ratio = exp(log π - log π_old): 新旧策略的比值
- A: 相对优势 (组内归一化后的奖励)
- KL: 与参考模型的 KL 散度
- β: KL 惩罚系数，防止模型偏离太远

# ════════════════════════════════════════════════════════════════════════════════
# 【完整数值例子：一步步算清楚！】
# ════════════════════════════════════════════════════════════════════════════════
#
# 假设 prompt = "1+1等于几？"，生成 4 个回复 (简化为 N=4)
#
# ────────────────────────────────────────────────────────────────────────────────
# 【Step 1: 生成回复并打分】
# ────────────────────────────────────────────────────────────────────────────────
#
#   回复1: "2"           → Reward = 0.9 (正确简洁)
#   回复2: "等于2"        → Reward = 0.8 (正确)
#   回复3: "1+1=3"       → Reward = 0.2 (错误)
#   回复4: "不知道"       → Reward = 0.1 (拒绝回答)
#
#   rewards = [0.9, 0.8, 0.2, 0.1]
#
# ────────────────────────────────────────────────────────────────────────────────
# 【Step 2: 计算组内均值和标准差】
# ────────────────────────────────────────────────────────────────────────────────
#
#   group_mean = (0.9 + 0.8 + 0.2 + 0.1) / 4 = 0.5
#   
#   方差 = [(0.9-0.5)² + (0.8-0.5)² + (0.2-0.5)² + (0.1-0.5)²] / 4
#        = [0.16 + 0.09 + 0.09 + 0.16] / 4 = 0.125
#   
#   group_std = √0.125 ≈ 0.354
#
# ────────────────────────────────────────────────────────────────────────────────
# 【Step 3: 计算相对优势 Advantage】
# ────────────────────────────────────────────────────────────────────────────────
#
#   advantage = (reward - mean) / std
#
#   回复1: A1 = (0.9 - 0.5) / 0.354 = +1.13  ← 比平均好很多，大力鼓励！
#   回复2: A2 = (0.8 - 0.5) / 0.354 = +0.85  ← 比平均好，鼓励
#   回复3: A3 = (0.2 - 0.5) / 0.354 = -0.85  ← 比平均差，惩罚
#   回复4: A4 = (0.1 - 0.5) / 0.354 = -1.13  ← 比平均差很多，大力惩罚！
#
#   advantages = [+1.13, +0.85, -0.85, -1.13]
#
#   【关键】正的优势 → 增加概率，负的优势 → 降低概率
#
# ────────────────────────────────────────────────────────────────────────────────
# 【Step 4: 计算 ratio (新旧策略比)】
# ────────────────────────────────────────────────────────────────────────────────
#
#   ratio = exp(log π - log π_old) = π / π_old
#
#   假设模型刚更新过一轮，各回复的概率变化:
#
#   回复1: π=0.35, π_old=0.30 → ratio = 0.35/0.30 = 1.17 (概率增加了)
#   回复2: π=0.30, π_old=0.28 → ratio = 1.07
#   回复3: π=0.20, π_old=0.22 → ratio = 0.91 (概率降低了)
#   回复4: π=0.15, π_old=0.20 → ratio = 0.75
#
#   ratios = [1.17, 1.07, 0.91, 0.75]
#
# ────────────────────────────────────────────────────────────────────────────────
# 【Step 5: 计算 KL 散度惩罚】
# ────────────────────────────────────────────────────────────────────────────────
#
#   KL = log(π) - log(π_ref)
#   
#   假设 π_ref 是 SFT 模型的概率，β = 0.1:
#
#   回复1: π=0.35, π_ref=0.25 → KL = log(0.35/0.25) = 0.34
#   回复2: π=0.30, π_ref=0.25 → KL = 0.18
#   回复3: π=0.20, π_ref=0.25 → KL = -0.22
#   回复4: π=0.15, π_ref=0.25 → KL = -0.51
#
#   KL 惩罚项 = β * KL = [0.034, 0.018, -0.022, -0.051]
#
# ────────────────────────────────────────────────────────────────────────────────
# 【Step 6: 计算每个回复的损失】
# ────────────────────────────────────────────────────────────────────────────────
#
#   单样本损失 = -(ratio * advantages - β * KL)
#
#   回复1: -(1.17 * 1.13 - 0.034) = -(1.32 - 0.034) = -1.29 (损失为负 → 鼓励!)
#   回复2: -(1.07 * 0.85 - 0.018) = -0.89
#   回复3: -(0.91 * (-0.85) - (-0.022)) = -(-0.77 + 0.022) = +0.75 (损失为正 → 惩罚!)
#   回复4: -(0.75 * (-1.13) - (-0.051)) = +0.80
#
#   总损失 L = mean([-1.29, -0.89, +0.75, +0.80]) = -0.16
#
# ────────────────────────────────────────────────────────────────────────────────
# 【总结：GRPO 的直觉】
# ────────────────────────────────────────────────────────────────────────────────
#
#   1. 好回复 (A>0) + 概率增加 (ratio>1) → 损失为负 → 继续保持!
#   2. 差回复 (A<0) + 概率降低 (ratio<1) → 损失为正 → 继续降低!
#   3. KL 项防止模型跑太远
#
#   【为什么叫"Group Relative"?】
#   因为优势 A 是组内相对计算的，不需要额外的 Critic 模型估计绝对价值
#   这比 PPO 简单很多！
# ════════════════════════════════════════════════════════════════════════════════

【与 PPO 的关键区别】
┌─────────────────┬────────────────────┬────────────────────┐
│                 │        PPO         │        GRPO        │
├─────────────────┼────────────────────┼────────────────────┤
│ Advantage 计算  │ A = R - V(s)       │ A = (R-μ)/σ 组内   │
│ Critic 模型     │ 需要               │ 不需要             │
│ 每 prompt 生成  │ 1 个               │ N 个 (默认 8)      │
│ 计算复杂度      │ 高                 │ 中                 │
│ 内存占用        │ 高 (Critic)        │ 中 (多次生成)      │
└─────────────────┴────────────────────┴────────────────────┘

【推理模型训练 (reasoning=1)】
当训练推理模型时，添加格式奖励:
- 正确使用 <think>...</think><answer>...</answer>: +0.5
- 每个正确标签: +0.25
这帮助模型学会"思考-回答"的推理模式

【使用方法】
python train_grpo.py --epochs 1 --batch_size 2 --num_generations 8 --reasoning 1
"""

import os
import sys

# ==================== 模块路径设置 ====================
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ==================== 导入必要的库 ====================
import argparse              # 命令行参数解析
import re                    # 正则表达式 (用于匹配推理格式)
import gc                    # 垃圾回收
import warnings              # 警告控制
import torch                 # PyTorch 核心
import torch.distributed as dist  # 分布式训练
from transformers import AutoTokenizer  # HuggingFace 分词器
from contextlib import nullcontext      # 空上下文管理器
from torch import optim                 # 优化器
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR  # 余弦学习率调度
from transformers import AutoModel      # 加载奖励模型

# 导入自定义模块
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import (
    Logger, is_main_process, lm_checkpoint, 
    init_distributed_mode, setup_seed, SkipBatchSampler, init_model
)

warnings.filterwarnings('ignore')


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """
    计算回复的奖励分数
    
    【奖励组成】
    总奖励 = 格式奖励 + Reward Model 分数
    
    1. 格式奖励 (仅 reasoning=1):
       - 完整格式 <think>...</think><answer>...</answer>: +0.5
       - 每个正确标签 (<think>, </think>, <answer>, </answer>): +0.25
       
    2. Reward Model 分数:
       - 使用外部奖励模型评估回复质量
       - 范围限制在 [-3, 3]
    
    【为什么需要格式奖励】
    - Reward Model 可能不理解推理格式
    - 格式奖励确保模型学会正确使用标签
    - 防止奖励稀疏 (只有完美回复才有高分)
    
    【参数】
    - prompts: 输入提示列表 [batch_size]
    - responses: 生成的回复列表 [batch_size * num_generations]
    - reward_model: 奖励模型
    - reward_tokenizer: 奖励模型的 tokenizer
    
    【返回】
    - rewards: 奖励张量 [batch_size * num_generations]
    """
    
    def reasoning_model_reward(rewards):
        """
        计算推理格式奖励
        
        【检查两种格式】
        1. <think>\n...\n</think>\n<answer>\n...\n</answer>
        2. <think>\n...\n</think>\n\n<answer>\n...\n</answer> (多一个换行)
        """
        # 完整格式的正则表达式
        # ^$ 表示整个字符串必须匹配
        # .*? 是非贪婪匹配
        # re.S 让 . 也匹配换行符
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        # 完整格式奖励: +0.5
        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern or match_pattern2:
                format_rewards.append(0.5)  # 完整格式匹配
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        # 部分标签奖励 (防止奖励稀疏)
        def mark_num(text):
            """检查每个标签是否出现恰好一次，每个 +0.25"""
            reward = 0
            if text.count("<think>") == 1: reward += 0.25
            if text.count("</think>") == 1: reward += 0.25
            if text.count("<answer>") == 1: reward += 0.25
            if text.count("</answer>") == 1: reward += 0.25
            return reward

        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    # 初始化奖励为 0
    rewards = torch.zeros(len(responses), device=args.device)
    
    # 添加格式奖励 (如果是推理模型)
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    # 使用 Reward Model 计算奖励
    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)  # 原始 prompt 数量
        scale = 3.0  # 奖励范围限制

        # 遍历每个 prompt
        for i in range(batch_size):
            # 每个 prompt 有 num_generations 个回复
            for j in range(args.num_generations):
                # 计算在 responses 中的索引
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                # 从 ChatML 格式中解析对话历史
                # 匹配 <|im_start|>role\ncontent<|im_end|>
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]

                # 构建完整对话 (加上生成的回复)
                tmp_chat = messages + [{"role": "assistant", "content": response}]
                
                # 调用 Reward Model 计算分数
                score = reward_model.get_score(reward_tokenizer, tmp_chat)
                
                # 裁剪到 [-scale, scale] 防止极端值
                score = max(min(score, scale), -scale)

                # 对推理模型，额外评估 <answer> 内容
                if args.reasoning == 1:
                    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                    if answer_match:
                        answer_content = answer_match.group(1).strip()
                        # 单独评估 answer 内容的质量
                        tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                        answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                        answer_score = max(min(answer_score, scale), -scale)
                        # 加权组合: 40% 完整回复 + 60% 答案内容
                        # 这样更强调最终答案的质量
                        score = score * 0.4 + answer_score * 0.6

                reward_model_scores.append(score)

        # 转为张量并累加到总奖励
        # ────────────────────────────────────────────────────────────────────
        # 【奖励是针对每个回复单独计算的！】
        # ────────────────────────────────────────────────────────────────────
        #
        # 【数据结构】
        #   假设 batch_size=2 (2个prompt), num_generations=4 (每个生成4个回复)
        #   
        #   prompts = ["1+1=?", "首都是?"]  # 2个prompt
        #   responses = [                   # 2*4=8个回复，展平成一维列表
        #       # prompt 0 的 4 个回复
        #       "2",           # responses[0]
        #       "等于2",        # responses[1]
        #       "1+1=3",       # responses[2]
        #       "不知道",       # responses[3]
        #       # prompt 1 的 4 个回复
        #       "北京",         # responses[4]
        #       "中国北京",     # responses[5]
        #       "上海",         # responses[6]
        #       "不清楚",       # responses[7]
        #   ]
        #
        # 【奖励计算：每个回复单独打分】
        #   rewards = [
        #       0.9, 0.8, 0.2, 0.1,  # prompt 0 的 4 个回复的奖励
        #       0.9, 0.85, 0.3, 0.2  # prompt 1 的 4 个回复的奖励
        #   ]  # 长度 = 8
        #
        # 【后续在 grpo_train_epoch 中，按组计算优势】
        #   组0 (prompt 0): rewards[0:4] = [0.9, 0.8, 0.2, 0.1]
        #       mean = 0.5, std = 0.354
        #       advantages[0:4] = [+1.13, +0.85, -0.85, -1.13]
        #
        #   组1 (prompt 1): rewards[4:8] = [0.9, 0.85, 0.3, 0.2]
        #       mean = 0.5625, std = 0.32
        #       advantages[4:8] = [+1.05, +0.90, -0.82, -1.13]
        #
        # 【总结】
        #   - 奖励: 对每个回复单独计算 (调用 Reward Model)
        #   - 优势: 在组内归一化 (同一个 prompt 的回复互相比较)
        #   - 这就是 "Group Relative" 的含义！
        # ────────────────────────────────────────────────────────────────────
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def grpo_train_epoch(epoch, loader, iters, ref_model, reward_model, reward_tokenizer, start_step=0, wandb=None):
    """
    GRPO 训练一个 epoch
    
    【GRPO 训练流程】
    for each batch (每个 batch 包含 B 个 prompt):
        1. 对每个 prompt 生成 N 个回复 (共 B*N 个回复)
        2. 用 Reward Model 对每个回复打分
        3. 在组内计算相对优势:
           advantage = (reward - group_mean) / group_std
        4. 计算策略梯度损失:
           loss = -log π * advantage + β * KL
        5. 反向传播更新模型
    
    【参数说明】
    - epoch: 当前 epoch 编号
    - loader: 数据加载器
    - iters: 本 epoch 总迭代次数
    - ref_model: 参考模型 (冻结，用于 KL 惩罚)
    - reward_model: 奖励模型 (冻结)
    - reward_tokenizer: 奖励模型的 tokenizer
    - start_step: 起始步数 (用于断点续训)
    - wandb: 实验追踪工具
    """
    for step, batch in enumerate(loader, start=start_step + 1):
        
        # ==================== Step 1: 准备输入 ====================
        # prompts 是 batch 中的提示列表
        prompts = batch['prompt']  # list[str], 长度 B
        
        # Token 化 prompt
        # padding_side="left": 左填充，这样生成时可以从右边继续
        prompt_inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            return_token_type_ids=False,
            padding_side="left",           # 左填充
            add_special_tokens=False       # 不添加特殊 token (prompt 已经包含了)
        ).to(args.device)  # input_ids: [B, P], attention_mask: [B, P]
        
        # 截断到最大长度
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # ==================== Step 2: 生成多个回复 ====================
        with torch.no_grad():
            # DDP 模型需要使用 .module 访问 generate 方法
            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            
            # 生成回复
            # num_return_sequences=N: 每个 prompt 生成 N 个回复
            # 输出形状: [B * num_generations, P + R]
            outputs = model_for_gen.generate(
                **prompt_inputs, 
                max_new_tokens=args.max_gen_len,  # 最大生成长度
                do_sample=True,                    # 采样而非贪婪
                temperature=0.8,                   # 采样温度
                num_return_sequences=args.num_generations,  # 每 prompt 生成 N 个
                pad_token_id=tokenizer.pad_token_id
            )  # [B*num_gen, P+R]

        # 提取生成的部分 (去掉 prompt)
        # completion_ids 只包含模型新生成的 token
        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]  # [B*num_gen, R]
        
        # ==================== Step 3: 计算 log 概率 ====================
        def get_per_token_logps(mdl, input_ids, n_keep):
            """
            计算每个 token 的 log 概率
            
            【参数】
            - mdl: 模型
            - input_ids: 输入 token ID [batch, seq_len]
            - n_keep: 只保留最后 n_keep 个位置的 log 概率
            
            【返回】
            - log_probs: [batch, n_keep]
            """
            # 确保 input_ids 可以计算梯度
            input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
            
            # 前向传播得到 logits
            # logits_to_keep 只计算最后几个位置的 logits，节省内存
            # ────────────────────────────────────────────────────────────────────
            # 【如何判断是否产生梯度？】
            # ────────────────────────────────────────────────────────────────────
            #
            # 【PyTorch 梯度产生的三个条件】
            #
            #   1. 输入张量 requires_grad=True
            #      - 这里 input_ids 是整数 ID，默认 requires_grad=False
            #
            #   2. 模型参数 requires_grad=True
            #      - 默认情况下，nn.Module 的所有参数都有 requires_grad=True
            #      - 除非明确调用 mdl.requires_grad_(False) 或在 torch.no_grad() 中
            #
            #   3. 不在 torch.no_grad() 上下文内
            #      - 此函数内没有 torch.no_grad()
            #      - 但函数外可能有，比如下面 ref_model 调用时就在 torch.no_grad() 中
            #
            # 【那么这里会产生梯度吗？】
            #
            #   看调用位置：
            #   - 对 model: per_token_logps = get_per_token_logps(model, outputs, ...)
            #     在 autocast_ctx 中，但不是 no_grad，所以会产生梯度
            #
            #   - 对 ref_model: ref_per_token_logps = get_per_token_logps(ref_model, ...)
            #     在 torch.no_grad() 中调用，不会产生梯度
            #
            # 【总结】
            #   - 同一个函数，传入不同的模型，梯度行为会不同！
            #   - model(inputs) → 有梯度
            #   - with torch.no_grad(): ref_model(inputs) → 无梯度
            # ────────────────────────────────────────────────────────────────────
            logits = mdl(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
            
            # 对每个样本，提取对应 token 的 log 概率
            per_token_logps = []
            for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
                ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
                # log_softmax 得到 log 概率分布
                # gather 提取目标 token 的 log 概率
                per_token_logps.append(
                    torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
                )
            return torch.stack(per_token_logps)  # [batch, n_keep]

        # 策略模型的 log 概率
        with autocast_ctx:
            per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))  # [B*num_gen, R]
            
            # 如果使用 MoE，计算辅助损失
            res = model(outputs) if lm_config.use_moe else None
            aux_loss = res.aux_loss if res is not None else torch.tensor(0.0, device=args.device)
        
        # 参考模型的 log 概率 (用于 KL 惩罚)
        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))  # [B*num_gen, R]

        # ==================== Step 4: 计算奖励 ====================
        # 解码生成的文本
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        
        # 计算奖励 (格式奖励 + Reward Model 分数)
        rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer).to(args.device)  # [B*num_gen]

        # ==================== Step 5: 计算组内相对优势 ====================
        # 这是 GRPO 的核心！
        # 不使用 Critic，而是用组内比较
        
        # 将奖励按组分组 [B*num_gen] → [B, num_gen]
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        
        # 计算组内均值和标准差
        mean_r = grouped_rewards.mean(dim=1)  # [B]
        std_r = grouped_rewards.std(dim=1)    # [B]
        
        # 扩展回原始形状
        mean_r = mean_r.repeat_interleave(args.num_generations)  # [B*num_gen]
        std_r = std_r.repeat_interleave(args.num_generations)    # [B*num_gen]
        
        # 计算相对优势: (reward - group_mean) / group_std
        # clamp 限制在 [-10, 10] 防止极端值
        advantages = torch.clamp((rewards - mean_r) / (std_r + 1e-4), -10, 10)
        
        # 再做一次全局归一化 (可选，有助于稳定训练)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # [B*num_gen]

        # ==================== Step 6: 构建回复掩码 ====================
        # 找到 EOS token 位置，EOS 之后的 token 不计入损失
        is_eos = completion_ids == tokenizer.eos_token_id  # [B*num_gen, R]
        
        # 找到第一个 EOS 的位置
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        
        # 创建掩码: EOS 之前 (包含) 为 1，之后为 0
        completion_mask = (
            torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) 
            <= eos_idx.unsqueeze(1)
        ).int()  # [B*num_gen, R]

        # ==================== Step 7: 计算 GRPO 损失 ====================
        # KL 散度: 与参考模型的差异
        # kl_div = log π_ref - log π
        kl_div = ref_per_token_logps - per_token_logps
        
        # 使用近似 KL: E[exp(diff) - diff - 1]
        # 这比标准 KL 更稳定
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]
        
        # 策略梯度损失
        # ratio = exp(log π - log π_old)，这里简化为 log π
        # loss = -(ratio * advantage) + β * kl
        # ────────────────────────────────────────────────────────────────────
        # 【组内多个输出的梯度下降：详细例子】
        # ────────────────────────────────────────────────────────────────────
        #
        # 假设 prompt="1+1=?"，生成 4 个回复:
        #   回复0: "2"      → advantage = +1.13 (好)
        #   回复1: "等于2"   → advantage = +0.85 (好)
        #   回复2: "1+1=3"  → advantage = -0.85 (差)
        #   回复3: "不知道"  → advantage = -1.13 (差)
        #
        # 【每个回复的损失计算】
        #
        #   per_token_loss = -(log_prob * advantage)
        #
        #   回复0: loss0 = -(log_prob_0 * 1.13)   # 负损失！
        #   回复1: loss1 = -(log_prob_1 * 0.85)   # 负损失！
        #   回复2: loss2 = -(log_prob_2 * -0.85)  # 正损失！
        #   回复3: loss3 = -(log_prob_3 * -1.13)  # 正损失！
        #
        # 【梯度的方向】
        #
        #   回复0 (advantage > 0):
        #     loss = -log_prob * 1.13
        #     梯度 ∂loss/∂θ = -1.13 * ∂log_prob/∂θ
        #     → 梯度下降会让 log_prob 变大 → 概率增加！
        #
        #   回复2 (advantage < 0):
        #     loss = -log_prob * (-0.85) = +0.85 * log_prob
        #     梯度 ∂loss/∂θ = +0.85 * ∂log_prob/∂θ
        #     → 梯度下降会让 log_prob 变小 → 概率降低！
        #
        # 【总损失是所有回复的平均】
        #
        #   total_loss = mean([loss0, loss1, loss2, loss3])
        #
        # 【backward() 时发生什么？】
        #
        #   1. 计算每个回复对模型参数的梯度
        #   2. 梯度会累加 (不是覆盖!)
        #
        #   假设模型有个参数 θ，控制输出 "2" 的概率:
        #
        #   回复0 ("2"):    ∂loss0/∂θ = -1.13 * ∂log_prob("2")/∂θ  (让"2"概率↑)
        #   回复2 ("1+1=3"): ∂loss2/∂θ = +0.85 * ∂log_prob("3")/∂θ  (让"3"概率↓)
        #
        #   总梯度 = ∂loss0/∂θ + ∂loss1/∂θ + ∂loss2/∂θ + ∂loss3/∂θ
        #
        # 【数值例子】
        #
        #   假设 θ 是某个权重，当前值 θ = 0.5
        #   
        #   回复0 贡献梯度: -0.02  (想让 θ 变大)
        #   回复1 贡献梯度: -0.01
        #   回复2 贡献梯度: +0.015 (想让 θ 变小)
        #   回复3 贡献梯度: +0.018
        #
        #   总梯度 = -0.02 + (-0.01) + 0.015 + 0.018 = 0.003
        #
        #   更新: θ_new = θ - lr * 总梯度
        #              = 0.5 - 0.001 * 0.003 = 0.499997
        #
        # 【关键理解】
        #
        #   1. 好回复 (A>0) 的梯度方向: 增加该回复的概率
        #   2. 差回复 (A<0) 的梯度方向: 降低该回复的概率
        #   3. 梯度会互相"拉扯"，最终效果是净效果
        #   4. 如果好回复数量多/优势大，模型会朝好的方向更新更多
        #
        # 【类比：拔河】
        #   - 好回复队: 把模型往"好"的方向拉 (增加好回复概率)
        #   - 差回复队: 把模型往"差"的方向拉 (其实是阻止好的)
        #   - 最终模型移动方向 = 两队力量的净效果
        # ────────────────────────────────────────────────────────────────────
        per_token_loss = -(
            torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1) 
            - args.beta * per_token_kl
        )  # [B*num_gen, R]
        
        # 对每个序列求平均，再对 batch 求平均
        # ────────────────────────────────────────────────────────────────────
        # 【为什么要 mean？】
        #   所有回复的损失加起来取平均，这样:
        #   - 好回复的负损失 + 差回复的正损失 = 净损失
        #   - backward() 时，所有回复的梯度都会累加
        #   - 最终模型更新的方向取决于"好回复"和"差回复"的博弈
        # ────────────────────────────────────────────────────────────────────
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        # 总损失 (加上 MoE 辅助损失)
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        # ==================== Step 8: 参数更新 ====================
        if (step + 1) % args.accumulation_steps == 0:
            # 梯度裁剪
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 优化器更新
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ==================== Step 9: 日志记录 ====================
        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Actor Loss: {policy_loss_val:.4f}, Aux Loss: {current_aux_loss:.4f}, Reward: {avg_reward_val:.4f}, '
                   f'Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')

            if wandb and is_main_process():
                wandb.log({
                    "policy_loss": policy_loss_val,
                    "aux_loss": current_aux_loss,
                    "reward": avg_reward_val,
                    "avg_response_len": avg_len_val,
                    "advantages_mean": advantages.mean().item(),
                    "learning_rate": current_lr
                })

        # ==================== Step 10: 保存检查点 ====================
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            
            lm_checkpoint(
                lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler
            )
            model.train()
            del state_dict

        # ==================== Step 11: 释放内存 ====================
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    
    # ----- 保存配置 -----
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    
    # ----- 训练配置 -----
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="每个 GPU 的 batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    
    # ----- 模型配置 -----
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=66, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")
    
    # ----- GRPO 特定配置 -----
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", help="RLAIF数据路径")
    parser.add_argument("--num_generations", type=int, default=8, help="每个prompt生成的样本数 (GRPO核心)")
    parser.add_argument("--beta", type=float, default=0.02, help="KL惩罚系数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    
    # ----- 实验追踪 -----
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb项目名")
    
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 注意: max_seq_len 需要包含生成的长度
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len + args.max_gen_len,  # prompt + 生成
        use_moe=bool(args.use_moe)
    )
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    # 根据是否是推理模型选择基础权重
    base_weight = "reason" if args.reasoning == 1 else "full_sft"
    
    # Policy模型 (正在训练)
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    
    # Reference模型 (冻结，用于 KL 惩罚)
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    
    # Reward模型 (冻结，用于评估回复质量)
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    
    # 数据和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 计算总步数用于学习率调度
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        
        if epoch == start_epoch and start_step > 0:
            # 断点续训
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + start_step + 1, ref_model, reward_model, reward_tokenizer, start_step, wandb)
        else:
            # 从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True,
                              drop_last=False, shuffle=(train_sampler is None),
                              num_workers=args.num_workers, sampler=train_sampler)
            grpo_train_epoch(epoch, loader, len(loader), ref_model, reward_model, reward_tokenizer, 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()
