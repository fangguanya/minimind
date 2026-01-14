"""
================================================================================
                    MiniMind PPO 训练脚本 (Proximal Policy Optimization)
================================================================================

【什么是 PPO】
PPO (近端策略优化) 是 RLHF 中最经典的强化学习算法:
- OpenAI 的 InstructGPT、ChatGPT 都使用 PPO
- 稳定、高效、效果好
- 是 LLM 对齐的标准方法

【形象比喻】
想象你在教一个学生写作:
- PPO 就像有两个老师:
  - Actor (演员): 负责写作
  - Critic (评论家): 负责评估

每次写完作文:
1. Critic 预测这篇作文能得几分
2. 实际批改后得到真实分数
3. 如果真实分数 > 预测分数 → Actor 这次写得好，鼓励
4. 如果真实分数 < 预测分数 → Actor 这次写得差，惩罚

【PPO 核心概念】

1. Actor (策略模型):
   - 就是我们要训练的 LLM
   - 输入 prompt，输出回复

2. Critic (价值模型):
   - 估计状态的价值 V(s)
   - 预测从当前状态开始能获得多少奖励

3. Advantage (优势):
   - A = R - V(s): 实际奖励 - 预测价值
   - A > 0: 比预期好，应该鼓励
   - A < 0: 比预期差，应该惩罚

4. 重要性采样:
   - 用旧策略 π_old 收集经验
   - 用新策略 π 来更新
   - ratio = π(a|s) / π_old(a|s)

5. 裁剪 (Clipping):
   - 限制 ratio 在 [1-ε, 1+ε] 范围内
   - 防止策略更新太大导致不稳定

【PPO 损失函数】
L_PPO = -min(r(θ) * A, clip(r(θ), 1-ε, 1+ε) * A) + c₁ * L_VF + c₂ * KL(π || π_ref)
                                                              ↑
                                                    这里的 π_ref 就是 Reference 模型！

其中:
- r(θ) = π(a|s) / π_old(a|s): 新旧策略比率
- A: 优势函数
- ε: 裁剪参数 (通常 0.1~0.2)
- L_VF: Critic 的 MSE 损失
- KL(π || π_ref): π 是当前 Actor，π_ref 是 Reference (冻结的 SFT 模型)

# ════════════════════════════════════════════════════════════════════════════════
# 【Reference 模型从哪来？RLHF 完整训练流程】
# ════════════════════════════════════════════════════════════════════════════════
#
# 你问得好！这确实是"鸡生蛋蛋生鸡"的问题
# 但其实 RLHF 有严格的训练顺序，不是同时训练的！
#
# ────────────────────────────────────────────────────────────────────────────────
# 【阶段 1: 预训练】从零开始
# ────────────────────────────────────────────────────────────────────────────────
#
#   大量文本 ──→ [随机初始化模型] ──→ 预训练 ──→ Base Model
#                                              (只会续写，不会对话)
#
#   输入: "今天天气"
#   输出: "很好，适合出门..." (只是续写，不是回答问题)
#
# ────────────────────────────────────────────────────────────────────────────────
# 【阶段 2: SFT (监督微调)】教会对话
# ────────────────────────────────────────────────────────────────────────────────
#
#   Base Model + 对话数据 ──→ SFT训练 ──→ SFT Model
#                                        (会对话了，但可能说废话/有害内容)
#
#   输入: "今天天气怎么样？"
#   输出: "今天天气晴朗，温度适宜..." (学会了问答格式)
#
#   ★ 这个 SFT Model 就是 Reference 模型！★
#   ★ 也是 Actor 的初始化权重！★
#
# ────────────────────────────────────────────────────────────────────────────────
# 【阶段 3: 训练 Reward Model】教会打分
# ────────────────────────────────────────────────────────────────────────────────
#
#   收集人类偏好数据:
#   ┌─────────────────────────────────────────────────────────────────┐
#   │ prompt: "写一首诗"                                              │
#   │ 回复A: "春风又绿江南岸，明月何时照我还" (人类选择: 好)          │
#   │ 回复B: "诗诗诗诗诗诗诗" (人类选择: 差)                          │
#   └─────────────────────────────────────────────────────────────────┘
#
#   用这些数据训练一个打分模型:
#   SFT Model + 偏好数据 ──→ 训练 ──→ Reward Model
#                                    (输入回复，输出分数)
#
#   注意: Reward Model 通常也是从 SFT Model 初始化的！
#
# ────────────────────────────────────────────────────────────────────────────────
# 【阶段 4: PPO 训练】用 RL 优化
# ────────────────────────────────────────────────────────────────────────────────
#
#   现在所有模型都准备好了:
#
#   ┌──────────────────────────────────────────────────────────────────────────┐
#   │                                                                          │
#   │  Actor ─────────────┐                                                    │
#   │  (从SFT初始化,训练)  │                                                    │
#   │         ↓           │                                                    │
#   │      生成回复 ──────→ Reward Model ──→ 奖励 R                            │
#   │         ↓           │  (冻结,打分)                                       │
#   │  Reference ─────────┤                                                    │
#   │  (SFT模型,冻结)      │──→ KL(Actor || Reference) 惩罚                    │
#   │         ↓           │                                                    │
#   │  Critic ────────────┘──→ 预测价值 V ──→ 优势 A = R - V                  │
#   │  (从SFT初始化,训练)                                                       │
#   │                                                                          │
#   └──────────────────────────────────────────────────────────────────────────┘
#
# ────────────────────────────────────────────────────────────────────────────────
# 【总结：各模型的来源】
# ────────────────────────────────────────────────────────────────────────────────
#
# ┌─────────────────┬─────────────────────────┬──────────────────────────────┐
# │ 模型            │ 初始化来源              │ PPO阶段是否训练              │
# ├─────────────────┼─────────────────────────┼──────────────────────────────┤
# │ Actor           │ SFT Model               │ ✓ 训练 (主角)                │
# │ Reference       │ SFT Model (同一个)      │ ✗ 冻结 (KL锚点)              │
# │ Critic          │ SFT Model 或 随机初始化 │ ✓ 训练 (估值)                │
# │ Reward Model    │ SFT Model               │ ✗ 冻结 (在阶段3已训练好)     │
# │ Old Actor       │ Actor 的副本            │ ✗ 定期从Actor复制            │
# └─────────────────┴─────────────────────────┴──────────────────────────────┘
#
# 【所以不是鸡生蛋！】
# 1. 先有 Base Model (预训练)
# 2. 再有 SFT Model (监督微调) ← 这就是 Reference 和 Actor 的起点
# 3. 再训练 Reward Model (用人类偏好数据)
# 4. 最后才是 PPO (用 RM 指导 Actor 优化)
#
# Reference = SFT Model，Actor = SFT Model 的可训练副本
# 它们起点相同，但 PPO 训练后 Actor 会变化，Reference 保持不变
# ════════════════════════════════════════════════════════════════════════════════

【PPO vs GRPO】
┌─────────────────┬────────────────────┬────────────────────┐
│                 │        PPO         │        GRPO        │
├─────────────────┼────────────────────┼────────────────────┤
│ Advantage 计算  │ A = R - V(s)       │ A = (R-μ)/σ 组内   │
│ Critic 模型     │ 需要训练           │ 不需要             │
│ 每 prompt 生成  │ 1 个               │ N 个               │
│ 稳定性          │ 较高               │ 中等               │
│ 复杂度          │ 高                 │ 低                 │
└─────────────────┴────────────────────┴────────────────────┘

# ════════════════════════════════════════════════════════════════════════════════
# 【PPO vs DPO：看起来像，本质完全不同！】
# ════════════════════════════════════════════════════════════════════════════════
#
# 你可能觉得它们很像：
# - 都有"训练模型"和"参考模型"
# - 都有 KL 散度
# - 都是让模型输出更好
#
# 但它们的核心思路完全不同！
#
# ────────────────────────────────────────────────────────────────────────────────
# 【本质区别：有没有"在线生成"】
# ────────────────────────────────────────────────────────────────────────────────
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                        PPO (强化学习)                                       │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │                                                                             │
# │   prompt ──→ [Actor模型] ──→ 生成回复 ──→ [Reward模型] ──→ 奖励分数         │
# │                  ↑                              ↓                           │
# │                  └──────── 根据奖励更新 ←───────┘                           │
# │                                                                             │
# │   特点：模型自己生成，自己被打分，自己改进                                   │
# │   类比：学生写作文 → 老师打分 → 学生根据分数改进                            │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                        DPO (监督学习)                                       │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │                                                                             │
# │   数据集: [(prompt, 好回复, 差回复), ...]  ← 人类标注好的！                 │
# │                     ↓                                                       │
# │   [模型] 学习：P(好回复) > P(差回复)                                        │
# │                                                                             │
# │   特点：直接从人类标注的对比数据学习，不需要模型自己生成                     │
# │   类比：给学生看范文和差文，让他学会分辨好坏                                │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# ────────────────────────────────────────────────────────────────────────────────
# 【训练流程对比】
# ────────────────────────────────────────────────────────────────────────────────
#
# 【PPO 一次迭代】
#   1. Actor 生成回复: "今天天气很好"
#   2. Reward 模型打分: 0.8 分
#   3. Critic 预测价值: 0.6 分
#   4. 计算优势: A = 0.8 - 0.6 = 0.2 (比预期好!)
#   5. 更新 Actor: 增加这个回复的概率
#   6. 更新 Critic: 让预测更准
#
# 【DPO 一次迭代】
#   1. 从数据集读取: (prompt, chosen="好回复", rejected="差回复")
#   2. 计算: log P(好回复) 和 log P(差回复)
#   3. 优化: 让 P(好)/P(差) 变大
#   4. 没有生成！没有打分！纯监督学习
#
# ────────────────────────────────────────────────────────────────────────────────
# 【模型数量对比】
# ────────────────────────────────────────────────────────────────────────────────
#
# PPO 需要 4-5 个模型:
#   1. Actor (训练): 生成回复
#   2. Critic (训练): 估计价值
#   3. Reference (冻结): KL 惩罚的锚点
#   4. Old Actor (定期更新): 重要性采样
#   5. Reward Model (冻结): 打分
#
# DPO 只需要 2 个模型:
#   1. Policy (训练): 学习偏好
#   2. Reference (冻结): KL 惩罚的锚点
#
# ────────────────────────────────────────────────────────────────────────────────
# 【KL 散度的作用不同】
# ────────────────────────────────────────────────────────────────────────────────
#
# 【PPO 中的 KL】
#   KL(π || π_ref): 限制 Actor 不要偏离参考太远
#   作用: 防止 reward hacking (找到奖励模型的漏洞)
#   
#   例如: 奖励模型对"好好好"给高分
#        没有 KL → Actor 学会输出"好好好好好好好"
#        有 KL → Actor 必须保持像正常语言
#
# 【DPO 中的 KL】
#   隐式包含在 log(π/π_ref) 中
#   作用: 防止模型在学习偏好时遗忘语言能力
#   
#   本质: DPO 是把 RLHF+KL 的解析解直接拿来用
#
# ────────────────────────────────────────────────────────────────────────────────
# 【优缺点对比】
# ────────────────────────────────────────────────────────────────────────────────
#
# ┌────────────┬─────────────────────────┬─────────────────────────┐
# │            │          PPO            │          DPO            │
# ├────────────┼─────────────────────────┼─────────────────────────┤
# │ 优点       │ - 能探索新回复          │ - 简单，不需要RM        │
# │            │ - 能持续优化            │ - 训练稳定              │
# │            │ - 理论基础强            │ - 显存占用小            │
# ├────────────┼─────────────────────────┼─────────────────────────┤
# │ 缺点       │ - 复杂，需要多个模型    │ - 依赖数据质量          │
# │            │ - 显存占用大            │ - 不能发现新回复        │
# │            │ - 训练不稳定            │ - 受限于数据分布        │
# ├────────────┼─────────────────────────┼─────────────────────────┤
# │ 适用场景   │ 有好的奖励模型          │ 有好的人类偏好数据      │
# │            │ 需要模型自主探索        │ 资源有限，追求简单      │
# └────────────┴─────────────────────────┴─────────────────────────┘
#
# ────────────────────────────────────────────────────────────────────────────────
# 【一句话总结】
# ────────────────────────────────────────────────────────────────────────────────
#
# PPO: 模型自己生成 → 被打分 → 根据分数改进 (强化学习)
# DPO: 直接从人类对比数据学习好坏 (监督学习)
#
# PPO 像"实习生在公司边干边学"
# DPO 像"学生看案例分析学习"
# ════════════════════════════════════════════════════════════════════════════════

【模型组件】
1. Actor (策略模型): 生成回复，需要训练
2. Critic (价值模型): 估计状态价值，需要训练
3. Reference (参考模型): 计算 KL 惩罚，冻结
4. Old Actor: 用于重要性采样，定期更新
5. Reward Model: 计算奖励，冻结

【使用方法】
python train_ppo.py --epochs 1 --batch_size 2 --learning_rate 8e-8
"""

import os
import sys

# ==================== 模块路径设置 ====================
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ==================== 导入必要的库 ====================
import argparse              # 命令行参数解析
import re                    # 正则表达式
import warnings              # 警告控制
import torch                 # PyTorch 核心
import torch.distributed as dist  # 分布式训练
import torch.nn.functional as F   # 函数式接口
from transformers import AutoTokenizer  # HuggingFace 分词器
from contextlib import nullcontext      # 空上下文管理器
from torch import optim, nn            # 优化器和神经网络模块
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_  # 梯度裁剪
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


class CriticModel(MiniMindForCausalLM):
    """
    Critic 模型 (价值网络)
    
    【作用】
    估计状态的价值 V(s)，用于计算优势函数:
    Advantage = R (实际奖励) - V(s) (预测价值)
    
    【直观理解】
    Critic 就像一个"预言家":
    - 看到当前对话状态
    - 预测最终能获得多少奖励
    - 如果实际奖励 > 预测 → Actor 做得好
    - 如果实际奖励 < 预测 → Actor 做得差
    
    【结构】
    复用 Actor (LLM) 的主干网络，但替换输出头:
    
    原始 (LM Head):
    hidden_states [B, L, H] → Linear → logits [B, L, V]
    用于预测下一个 token
    
    Critic (Value Head):
    hidden_states [B, L, H] → Linear → values [B, L, 1] → squeeze → [B, L]
    用于预测每个位置的价值
    
    通常我们只取最后一个有效位置的价值作为整个序列的价值。
    
    【训练目标】
    最小化预测价值与实际奖励的 MSE:
    L_critic = MSE(V(s), R)
    
    【为什么复用 Actor 的主干】
    1. 节省内存: 不需要训练两个完整模型
    2. 共享表示: Critic 可以利用 LLM 学到的语言理解能力
    3. 更稳定: 初始化从预训练模型开始
    """
    
    def __init__(self, params):
        """
        初始化 Critic 模型
        
        【参数】
        - params: MiniMindConfig 模型配置
        """
        # 继承 MiniMindForCausalLM 的所有组件
        super().__init__(params)
        
        # 替换 LM Head 为 Value Head
        # 输入: hidden_size (如 512)
        # 输出: 1 (标量价值)
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """
        前向传播
        
        【流程】
        1. 输入经过 Transformer 主干网络
        2. 对隐藏状态进行 RMSNorm
        3. 通过 Value Head 得到每个位置的价值
        
        【参数】
        - input_ids: 输入 token ID [batch, seq_len]
        - attention_mask: 注意力掩码 [batch, seq_len]
        
        【返回】
        - values: 每个位置的价值 [batch, seq_len]
        """
        # 通过 Transformer 主干获取隐藏状态
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # outputs[0] 是最后一层的隐藏状态 [B, L, H]
        # 应用 RMSNorm (与 Actor 保持一致)
        hidden_states = self.model.norm(outputs[0])
        
        # 通过 Value Head 得到每个位置的价值
        # [B, L, H] → [B, L, 1] → squeeze → [B, L]
        values = self.value_head(hidden_states).squeeze(-1)
        
        return values


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """
    计算回复的奖励分数
    
    【奖励组成】
    总奖励 = 格式奖励 (可选) + Reward Model 分数
    
    【参数】
    - prompts: 输入提示列表 [batch_size]
    - responses: 生成的回复列表 [batch_size]
    - reward_model: 奖励模型
    - reward_tokenizer: 奖励模型的 tokenizer
    
    # ────────────────────────────────────────────────────────────────────
    # 【为什么 model 和 tokenizer 要分开？】
    # ────────────────────────────────────────────────────────────────────
    #
    # 这是 HuggingFace 的设计模式，不是我们自己搞的：
    #
    # 【原因 1: 职责分离】
    #   - Model: 只负责神经网络计算 (forward, backward)
    #   - Tokenizer: 只负责文本处理 (编码, 解码)
    #   它们是两个独立的类，没有继承关系
    #
    # 【原因 2: 文件存储分开】
    #   model_path/
    #   ├── config.json           # 模型配置
    #   ├── pytorch_model.bin     # 模型权重  ← AutoModel 加载这个
    #   ├── tokenizer.json        # 分词规则
    #   └── tokenizer_config.json # 分词配置  ← AutoTokenizer 加载这个
    #
    # 【原因 3: 可以混搭】
    #   - 同一个 tokenizer 可以用于不同模型
    #   - 同一个模型可以用不同 tokenizer (虽然不推荐)
    #
    # 【加载代码】(见 main 函数)
    #   reward_model = AutoModel.from_pretrained(path)
    #   reward_tokenizer = AutoTokenizer.from_pretrained(path)  # 同一个 path!
    #
    # 【为什么不合并成一个对象？】
    #   - 显存优化: tokenizer 在 CPU，model 在 GPU
    #   - 并行处理: tokenizer 可以多线程，model 不行
    #   - 灵活性: 有时只需要 tokenizer 做预处理
    # ────────────────────────────────────────────────────────────────────
    
    【返回】
    - rewards: 奖励张量 [batch_size]
    """
    
    def reasoning_model_reward(rewards):
        """计算推理模型的格式奖励"""
        # 检查是否匹配推理格式
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"

        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        # 完整格式奖励
        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern:
                format_rewards.append(0.5)
            elif match_pattern2:
                format_rewards.append(0.5)
            else:
                format_rewards.append(0.0)
        rewards += torch.tensor(format_rewards, device=args.device)

        # 部分标签奖励 (防止奖励稀疏)
        def mark_num(text):
            reward = 0
            if text.count("<think>") == 1: reward += 0.25
            if text.count("</think>") == 1: reward += 0.25
            if text.count("<answer>") == 1: reward += 0.25
            if text.count("</answer>") == 1: reward += 0.25
            return reward

        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    # 初始化奖励
    rewards = torch.zeros(len(responses), device=args.device)

    # 格式奖励 (仅推理模型)
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    # 使用 Reward Model 计算奖励
    with torch.no_grad():
        reward_model_scores = []
        for prompt, response in zip(prompts, responses):
            # 从 ChatML 格式解析对话历史
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]

            # 构建完整对话
            tmp_chat = messages + [{"role": "assistant", "content": response}]
            score = reward_model.get_score(reward_tokenizer, tmp_chat)

            # 裁剪奖励范围
            scale = 3.0
            score = max(min(score, scale), -scale)

            # 推理模型: 额外评估 answer 内容
            if args.reasoning == 1:
                answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                    answer_score = max(min(answer_score, scale), -scale)
                    # 加权组合
                    score = score * 0.4 + answer_score * 0.6
                    
            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def ppo_train_epoch(epoch, loader, iters, old_actor_model, ref_model, actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step=0, wandb=None):
    """
    PPO 训练一个 epoch
    
    【PPO 训练流程】
    for each batch:
        1. 用 Actor 生成回复
        2. 用 Reward Model 计算奖励 R
        3. 用 Critic 预测价值 V(s)
        4. 计算优势 A = R - V(s)
        5. 计算策略损失 (PPO-Clip)
        6. 计算价值损失 (MSE)
        7. 计算 KL 惩罚
        8. 更新 Actor 和 Critic
        9. 定期更新 Old Actor
    
    【PPO-Clip 详解】
    r(θ) = π(a|s) / π_old(a|s)  # 新旧策略比率
    
    L_CLIP = min(
        r(θ) * A,                           # 无限制的策略梯度
        clip(r(θ), 1-ε, 1+ε) * A            # 限制更新幅度
    )
    
    这确保了即使 A 很大，策略也不会更新太多。
    
    【参数说明】
    - epoch: 当前 epoch
    - loader: 数据加载器
    - iters: 迭代次数
    - old_actor_model: 旧策略模型 (用于重要性采样)
    - ref_model: 参考模型 (用于 KL 惩罚)
    - actor_scheduler: Actor 学习率调度器
    - critic_scheduler: Critic 学习率调度器
    - reward_model: 奖励模型
    - reward_tokenizer: 奖励模型 tokenizer
    - start_step: 起始步数
    - wandb: 实验追踪
    """
    # 设置训练模式
    actor_model.train()
    critic_model.train()

    for step, batch in enumerate(loader, start=start_step + 1):
        
        # ==================== Step 1: 准备输入 ====================
        prompts = batch["prompt"]  # list[str], 长度 B
        
        # Token 化，左填充以便生成
        enc = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=args.max_seq_len, 
            padding_side="left"
        ).to(args.device)
        
        prompt_length = enc.input_ids.shape[1]  # prompt 的长度

        # ==================== Step 2: 生成回复 ====================
        # ────────────────────────────────────────────────────────────────────
        # 【为什么用 generate() 而不是 model()？】
        # ────────────────────────────────────────────────────────────────────
        #
        # 【model() = 一次前向传播】
        #   输入: "今天天气" (4个token)
        #   输出: logits [1, 4, vocab_size]  ← 只是概率分布，不是文字！
        #   
        #   适用场景: Pretrain/SFT/DPO
        #   因为: 训练数据已经有完整的输入+输出，只需要算 loss
        #   
        #   例如 SFT:
        #   输入 X = "问：1+1=？答：2"
        #   标签 Y = "：1+1=？答：2<eos>"
        #   model(X) → logits → CrossEntropy(logits, Y) → loss
        #
        # 【generate() = 自回归生成，循环调用 model()】
        #   输入: "今天天气" (4个token)
        #   
        #   循环开始:
        #     第1轮: model("今天天气") → 预测下一个token → "很"
        #     第2轮: model("今天天气很") → 预测下一个token → "好"
        #     第3轮: model("今天天气很好") → 预测下一个token → "，"
        #     ... 直到生成 <eos> 或达到 max_new_tokens
        #   
        #   输出: "今天天气很好，适合出门。" ← 真正的文字！
        #
        # 【PPO 为什么必须用 generate()？】
        #   因为 PPO 是强化学习，需要模型"自己生成"回复，然后被打分
        #   
        #   流程:
        #   prompt ──→ Actor.generate() ──→ 回复 ──→ Reward Model 打分
        #                    ↑
        #               这里必须真的生成出文字！
        #
        # 【对比总结】
        #   ┌───────────────┬──────────────────┬──────────────────────────┐
        #   │ 方法          │ 输出             │ 使用场景                 │
        #   ├───────────────┼──────────────────┼──────────────────────────┤
        #   │ model()       │ logits (概率)    │ 训练时算loss             │
        #   │ generate()    │ token序列 (文字) │ 推理时/PPO生成回复       │
        #   └───────────────┴──────────────────┴──────────────────────────┘
        # ────────────────────────────────────────────────────────────────────
        
        with torch.no_grad():
            # DDP 模型需要访问 .module
            model_for_gen = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            
            # 生成回复 (自回归，一个token一个token生成)
            gen_out = model_for_gen.generate(
                input_ids=enc.input_ids, 
                attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len,  # 最多生成多少个新token
                do_sample=True,                   # 采样 (有随机性)
                temperature=0.8,                  # 温度 (越高越随机)
                pad_token_id=tokenizer.pad_token_id, 
                eos_token_id=tokenizer.eos_token_id
            )  # [B, P+R]  P=prompt长度, R=生成的回复长度

        # 解码生成的回复
        responses_text = [
            tokenizer.decode(gen_out[i, prompt_length:], skip_special_tokens=True) 
            for i in range(len(prompts))
        ]
        
        # ==================== Step 3: 计算奖励 ====================
        rewards = calculate_rewards(prompts, responses_text, reward_model, reward_tokenizer)  # [B]

        # ==================== Step 4: Critic 预测价值 ====================
        # 创建 attention mask (非 padding 位置为 1)
        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]
        
        # Critic 前向传播，得到每个位置的价值
        values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)  # [B, P+R]
        
        # 取最后一个有效位置的价值作为整个序列的价值
        # 找到每个样本的最后一个非 padding 位置
        last_indices = (full_mask * torch.arange(full_mask.size(1), device=gen_out.device)).argmax(dim=1)
        
        # 提取对应位置的价值
        values = values_seq[torch.arange(values_seq.size(0), device=values_seq.device), last_indices]  # [B]
        
        # ==================== Step 5: 计算优势 ====================
        # A = R - V(s)
        # 如果实际奖励 > 预测价值 → 这个动作比预期好
        # detach() 防止梯度流向 Critic (这里只更新 Actor)
        advantages = rewards - values.detach()  # [B]

        # ==================== Step 6: Actor 前向传播 ====================
        # ────────────────────────────────────────────────────────────────────
        # 【为什么要再做一次 forward？不是已经 generate 过了吗？】
        # ────────────────────────────────────────────────────────────────────
        #
        # 【关键区别】
        #   generate():  在 torch.no_grad() 里，没有梯度！只是拿到 token 序列
        #   model():     有梯度！可以反向传播更新参数
        #
        # 【generate() 的问题】
        #   gen_out = model.generate(prompt)
        #   # gen_out = [token1, token2, token3, ...]  ← 只是一串数字
        #   # 没有 logits，没有梯度，没法算 loss，没法 backward！
        #
        # 【所以需要两步】
        #   Step 1: generate() ─→ 得到 token 序列 (无梯度)
        #   Step 2: model(tokens) ─→ 得到 logits (有梯度，可以算 loss)
        #
        # 【类比】
        #   generate() = 学生写作文 (产出文字)
        #   model()    = 老师批改，画红圈 (产出梯度，告诉哪里要改)
        #
        # 【完整流程图】
        #
        #   prompt ──→ generate() ──→ gen_out (token序列)
        #                               ↓
        #                          model(gen_out) ──→ logits (有梯度!)
        #                               ↓
        #                          计算 log_prob
        #                               ↓
        #                          计算 PPO loss
        #                               ↓
        #                          loss.backward() ← 这里才有反向传播！
        #                               ↓
        #                          optimizer.step() ← 更新参数
        #
        # 【backward 在哪里？】
        #   往下看 Step 9，有 scaler.scale(loss).backward()
        #   这行代码会把梯度从 loss 反向传播回 actor_model 的参数
        # ────────────────────────────────────────────────────────────────────
        
        with autocast_ctx:
            res = actor_model(input_ids=gen_out, attention_mask=full_mask)
            logits = res.logits  # [B, P+R, V]  ← 这个 logits 有梯度！
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
        
        # 计算每个 token 的 log 概率
        # labels = 目标 token (输入右移一位)
        labels = gen_out[:, 1:].clone()  # [B, P+R-1]
        
        # log_softmax 得到 log 概率分布
        # gather 提取目标 token 的 log 概率
        logp_tokens = F.log_softmax(logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)  # [B, P+R-1]
        
        # 只计算回复部分 (不包括 prompt)
        seq_len = gen_out.size(1) - 1
        resp_mask = torch.arange(seq_len, device=gen_out.device).unsqueeze(0) >= prompt_length - 1
        
        # 排除 padding 位置
        final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id))  # [B, P+R-1]
        
        # 对回复部分的 log 概率求和
        actor_logp = (logp_tokens * final_mask).sum(dim=1)  # [B]

        # ==================== Step 7: Old Actor 和 Ref Model 的 log 概率 ====================
        with torch.no_grad():
            # Old Actor (用于重要性采样)
            old_logits = old_actor_model(input_ids=gen_out, attention_mask=full_mask).logits
            old_logp_tokens = F.log_softmax(old_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
            old_logp = (old_logp_tokens * final_mask).sum(dim=1)  # [B]
            
            # Reference Model (用于 KL 惩罚)
            ref_logits = ref_model(input_ids=gen_out, attention_mask=full_mask).logits
            ref_logp_tokens = F.log_softmax(ref_logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
            ref_logp = (ref_logp_tokens * final_mask).sum(dim=1)  # [B]

        # ==================== Step 8: 计算 PPO 损失 ====================
        
        # KL 散度 (与 old actor)
        kl = (actor_logp - old_logp).mean()  # scalar
        
        # KL 散度 (与 reference model，用于惩罚)
        kl_ref = (actor_logp - ref_logp).mean()  # scalar
        
        # 重要性采样比率
        # ratio = π(a|s) / π_old(a|s) = exp(log π - log π_old)
        ratio = torch.exp(actor_logp - old_logp)  # [B]
        
        # PPO-Clip 损失
        # surr1: 无裁剪的策略梯度
        surr1 = ratio * advantages  # [B]
        
        # surr2: 裁剪后的策略梯度
        # clip 限制 ratio 在 [1-ε, 1+ε] 范围内
        surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages  # [B]
        
        # 取两者的最小值 (悲观估计)
        # 负号是因为我们要最大化目标，但优化器最小化损失
        policy_loss = -torch.min(surr1, surr2).mean()  # scalar
        
        # Critic 损失: MSE(V(s), R)
        # ────────────────────────────────────────────────────────────────────
        # 【MSE 是什么？均方误差 Mean Squared Error】
        # ────────────────────────────────────────────────────────────────────
        #
        # 【数学公式】
        #   MSE = (1/n) * Σ(预测值 - 真实值)²
        #       = (1/n) * Σ(V(s) - R)²
        #
        # 【为什么用平方？】
        #   1. 让正负误差都变正 (不然正负抵消)
        #   2. 放大大误差，缩小小误差 (对大错更敏感)
        #   3. 数学上好求导: d(x²)/dx = 2x
        #
        # 【数值例子】
        #   Critic 预测价值 V = [0.6, 0.8, 0.5]
        #   实际奖励      R = [0.8, 0.7, 0.9]
        #   误差          e = [-0.2, 0.1, -0.4]
        #   平方误差     e² = [0.04, 0.01, 0.16]
        #   MSE = (0.04 + 0.01 + 0.16) / 3 = 0.07
        #
        # 【Critic 的目标】
        #   让 MSE → 0，即让预测价值 V(s) 尽可能接近实际奖励 R
        #   这样 Advantage = R - V(s) 的估计才准确
        # ────────────────────────────────────────────────────────────────────
        value_loss = F.mse_loss(values, rewards)  # scalar
        
        # 总损失 = Actor 损失 + c1 * Critic 损失 + c2 * KL 惩罚 + MoE 辅助损失
        loss = (policy_loss + args.vf_coef * value_loss + args.kl_coef * kl_ref + aux_loss) / args.accumulation_steps
        
        # ────────────────────────────────────────────────────────────────────
        # 【loss.backward() 怎么知道更新谁？】
        # ────────────────────────────────────────────────────────────────────
        #
        # 【答案：PyTorch 自动追踪计算图！】
        #
        # 【计算图长这样】
        #
        #   actor_model.parameters() ──→ logits ──→ actor_logp ──→ ratio ──→ policy_loss ─┐
        #                                                                                  │
        #   critic_model.parameters() ──→ values ──→ value_loss ──────────────────────────→ loss
        #                                                                                  │
        #   ref_model.parameters() ──→ ref_logp ──→ kl_ref ────────────────────────────────┘
        #        ↑                                    (但 ref_model 在 no_grad 里，没梯度！)
        #   冻结，不更新
        #
        # 【backward() 做了什么？】
        #   1. 从 loss 出发，沿着计算图反向走
        #   2. 遇到 requires_grad=True 的参数，就计算梯度
        #   3. 把梯度存到 param.grad 里
        #
        # 【谁会被更新？】
        #   - actor_model: ✓ (policy_loss 经过它)
        #   - critic_model: ✓ (value_loss 经过它)
        #   - ref_model: ✗ (在 torch.no_grad() 里，没有梯度)
        #   - old_actor_model: ✗ (在 torch.no_grad() 里)
        #
        # 【optimizer.step() 才真正更新参数】
        #   actor_optimizer.step()   # 用 actor_model.parameters() 的梯度更新
        #   critic_optimizer.step()  # 用 critic_model.parameters() 的梯度更新
        # ────────────────────────────────────────────────────────────────────
        loss.backward()

        # ==================== Step 9: 参数更新 ====================
        if (step + 1) % args.accumulation_steps == 0:
            # 梯度裁剪
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            
            # 更新参数
            actor_optimizer.step()
            critic_optimizer.step()
            
            # 更新学习率
            actor_scheduler.step()
            critic_scheduler.step()
            
            # 清零梯度
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

        # ==================== Step 10: 日志记录 ====================
        if is_main_process():
            # 计算平均回复长度
            response_ids = gen_out[:, enc.input_ids.shape[1]:]
            is_eos = (response_ids == tokenizer.eos_token_id)
            eos_indices = torch.argmax(is_eos.int(), dim=1)
            has_eos = is_eos.any(dim=1)
            lengths = torch.where(has_eos, eos_indices + 1, torch.tensor(response_ids.shape[1], device=is_eos.device))
            avg_len = lengths.float().mean()

            actor_loss_val = policy_loss.item()
            critic_loss_val = value_loss.item()
            current_aux_loss = aux_loss.item()
            reward_val = rewards.mean().item()
            kl_val = kl.item()
            kl_ref_val = kl_ref.item()
            avg_len_val = avg_len.item()
            actor_lr = actor_optimizer.param_groups[0]['lr']
            critic_lr = critic_optimizer.param_groups[0]['lr']

            if wandb is not None:
                wandb.log({
                    "actor_loss": actor_loss_val,
                    "critic_loss": critic_loss_val,
                    "aux_loss": current_aux_loss,
                    "reward": reward_val,
                    "kl": kl_val,
                    "kl_ref": kl_ref_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                })

            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                   f"Actor Loss: {actor_loss_val:.4f}, Critic Loss: {critic_loss_val:.4f}, Aux Loss: {current_aux_loss:.4f}, "
                   f"Reward: {reward_val:.4f}, KL: {kl_val:.4f}, KL_ref: {kl_ref_val:.4f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.8f}, Critic LR: {critic_lr:.8f}")

        # ==================== Step 11: 更新 Old Actor ====================
        # 每隔一定步数，将当前 Actor 的权重复制到 Old Actor
        # 这样 Old Actor 跟踪"近期"的策略，用于重要性采样
        if (step + 1) % args.update_old_actor_freq == 0:
            state_dict = actor_model.module.state_dict() if isinstance(actor_model, DistributedDataParallel) else actor_model.state_dict()
            old_actor_model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})
            old_actor_model.to(args.device)

        # ==================== Step 12: 保存检查点 ====================
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            actor_state = actor_model.module.state_dict() if isinstance(actor_model, DistributedDataParallel) else actor_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)
            
            # 保存完整检查点 (包括 Critic)
            lm_checkpoint(
                lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer, 
                epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                scheduler=actor_scheduler, critic_model=critic_model, 
                critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler
            )
            actor_model.train()
            del actor_state

        # ==================== Step 13: 释放内存 ====================
        del enc, gen_out, responses_text, rewards, full_mask, values_seq, values, advantages
        del logits, labels, logp_tokens, final_mask, actor_logp, old_logits, old_logp, ref_logits, ref_logp
        del kl, kl_ref, ratio, surr1, surr2, policy_loss, value_loss, loss


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    
    # ----- 保存配置 -----
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    
    # ----- 训练配置 -----
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=8e-8, help="Critic学习率")
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
    
    # ----- PPO 特定配置 -----
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", help="RLAIF数据路径")
    parser.add_argument("--clip_epsilon", type=float, default=0.1, help="PPO裁剪参数 ε")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数 c1")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数 c2")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--update_old_actor_freq", type=int, default=4, help="更新old_actor_model的频率")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    
    # ----- 实验追踪 -----
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
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
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    base_weight = "reason" if args.reasoning == 1 else "full_sft"
    
    # Actor模型 (策略模型，需要训练)
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    
    # Old Actor模型 (用于重要性采样，定期更新)
    old_actor_model, _ = init_model(lm_config, base_weight, device=args.device)
    old_actor_model = old_actor_model.eval().requires_grad_(False)
    
    # Reference模型 (用于 KL 惩罚，冻结)
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    
    # Critic模型 (价值模型，需要训练)
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)  # 加载主干权重
    critic_model = critic_model.to(args.device)
    
    # Reward模型 (评估回复质量，冻结)
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    
    # 数据和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len))
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    # Actor 和 Critic 各自的优化器
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    
    # 学习率调度
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. DDP包模型 ==========
    if dist.is_initialized():
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
        old_actor_model.to(args.device)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        
        if epoch == start_epoch and start_step > 0:
            # 断点续训
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            ppo_train_epoch(epoch, loader, len(loader) + start_step + 1, old_actor_model, ref_model, 
                           actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step, wandb)
        else:
            # 从头开始
            loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(train_sampler is None), 
                              sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
            ppo_train_epoch(epoch, loader, len(loader), old_actor_model, ref_model, 
                           actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, 0, wandb)
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.destroy_process_group()
