"""
================================================================================
                    MiniMind SPO 训练脚本 (Self-Play Optimization)
================================================================================

【什么是 SPO】
SPO (自博弈优化) 是一种简化的强化学习方法:
- 比 PPO 更简单: 不需要 Critic 模型
- 比 GRPO 更简单: 不需要多次采样
- 使用自适应基线 (Adaptive Baseline) 替代 Critic

【核心思想】
1. 对每个 prompt 只生成一个回复 (而非 GRPO 的多个)
2. 使用历史统计数据维护一个"期望奖励"基线
3. 优势 = 当前奖励 - 基线
4. 用 Policy Gradient 更新

【SPO vs PPO vs GRPO】
┌─────────────────┬────────────┬────────────┬────────────┐
│                 │    PPO     │    GRPO    │    SPO     │
├─────────────────┼────────────┼────────────┼────────────┤
│ Critic 模型     │    需要    │   不需要   │   不需要   │
│ 每 prompt 采样  │    1个     │   N个(8)   │    1个     │
│ 基线估计        │ Critic V(s)│  组内均值  │  自适应EMA │
│ 计算复杂度      │    高      │    中      │    低      │
│ 内存占用        │    高      │    中      │    低      │
└─────────────────┴────────────┴────────────┴────────────┘

【自适应基线 (Adaptive Baseline)】
SPO 使用 Beta 分布来跟踪历史奖励分布:
- 维护 α 和 β 两个参数
- 基线 = α / (α + β)
- 每步更新: α = ρ*α + reward, β = ρ*β + (1-reward)
- ρ 是衰减因子，根据 KL 散度自适应调整

【衰减因子 ρ 的自适应】
- 当模型变化大时 (KL 大)，ρ 变小，基线更新更快
- 当模型稳定时 (KL 小)，ρ 保持较大，基线更稳定
- 公式: ρ = 2^(-KL / D_half)

【SPO 损失函数】
L = -E[log π(a|s) * (r - baseline)] + β * KL(π || π_ref)

其中:
- r: 当前回复的奖励
- baseline: 自适应基线估计
- π_ref: 参考模型
- β: KL 惩罚系数

【优势】
1. 计算效率高: 每 prompt 只生成一次
2. 内存占用低: 不需要 Critic
3. 实现简单: 核心逻辑清晰
4. 效果不错: 实践中表现良好

【使用方法】
python train_spo.py --reasoning 1 --learning_rate 1e-7
"""

import os
import sys

# 设置包名，确保相对导入正常工作
__package__ = "trainer"
# 将父目录添加到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import gc
import warnings
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model

# 忽略不重要的警告
warnings.filterwarnings('ignore')


class AutoAdaptiveValueTracker:
    """
    SPO 自适应价值追踪器
    
    【作用】
    跟踪历史奖励分布，提供稳定的基线估计。
    替代 PPO 中的 Critic 模型。
    
    【核心思想】
    使用 Beta 分布参数 (α, β) 来建模奖励分布:
    - 基线 = α / (α + β)
    - 这是 Beta 分布的期望值
    
    【自适应更新】
    衰减因子 ρ 根据 KL 散度自适应调整:
    - ρ = 2^(-KL / D_half)
    - KL 大时 ρ 小: 模型变化大，基线更新快
    - KL 小时 ρ 大: 模型稳定，基线更稳定
    
    【参数说明】
    - rho_mode: 衰减模式 ('kl' 或 'constant')
    - rho_const: 常数衰减因子
    - D_half: ρ 减半时的 KL 值
    - clip_lower/upper: ρ 的裁剪范围
    """
    def __init__(self, rho_mode='kl', rho_const=0.9, D_half=0.06, clip_lower=0.5, clip_upper=0.96):
        self.rho_mode = rho_mode
        self.rho_const = rho_const
        self.D_half = D_half
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper
        
        # 初始化 Beta 分布参数
        # N_init 确保初始基线 = clip_lower
        N_init = 1.0 / (1.0 - self.clip_lower)
        self.alpha = 0.5 * N_init  # 成功次数
        self.beta = 0.5 * N_init   # 失败次数
        
        # 上一步的平均 log 概率 (用于计算 KL)
        self.old_mean_logprob = None

    def get_baselines(self, batch_size):
        """
        获取当前基线估计
        
        【计算】
        baseline = α / (α + β)
        这是 Beta 分布的期望值
        
        【参数】
        - batch_size: 批量大小
        
        【返回】
        - baselines: [batch_size] 的基线张量
        """
        baseline = self.alpha / (self.alpha + self.beta)
        return torch.full((batch_size,), baseline, dtype=torch.float32)

    def compute_rho(self, cur_mean_logprob):
        """
        计算自适应衰减因子 ρ
        
        【公式】
        ρ = 2^(-KL / D_half)
        
        【解释】
        - KL = |old_logprob - cur_logprob|: 策略变化程度
        - D_half: ρ 减半需要的 KL 值
        - KL 大时 ρ 小: 策略变化大，需要更快更新基线
        - KL 小时 ρ 大: 策略稳定，基线保持稳定
        
        【参数】
        - cur_mean_logprob: 当前的平均 log 概率
        
        【返回】
        - rho: 衰减因子 ∈ [clip_lower, clip_upper]
        """
        if self.rho_mode == 'constant':
            return self.rho_const
        
        if self.old_mean_logprob is None:
            return self.rho_const
        
        # 计算 KL 散度近似值
        kl = abs(self.old_mean_logprob - cur_mean_logprob)
        
        # 指数衰减: ρ = 2^(-KL / D_half)
        rho = 2 ** (-kl / self.D_half)
        
        # 裁剪到合理范围
        return max(min(rho, self.clip_upper), self.clip_lower)

    def update(self, rewards, cur_logprobs=None, response_masks=None):
        """
        更新 Beta 分布参数
        
        【更新公式】
        α' = ρ * α + normalized_reward
        β' = ρ * β + (1 - normalized_reward)
        
        【归一化】
        奖励从 [-scale, scale] 归一化到 [0, 1]:
        normalized = (reward + scale) / (2 * scale)
        
        【参数】
        - rewards: 当前 batch 的奖励 [batch_size]
        - cur_logprobs: 当前的 log 概率 (用于计算 KL)
        - response_masks: 回复掩码
        
        【返回】
        - rho: 使用的衰减因子
        """
        # 计算自适应 ρ
        if cur_logprobs is not None and response_masks is not None:
            mean_logprob = ((cur_logprobs * response_masks).sum() / response_masks.sum()).item()
            rho = self.compute_rho(mean_logprob)
            self.old_mean_logprob = mean_logprob
        else:
            rho = self.rho_const

        # 归一化奖励到 [0, 1]
        scale = 3.0
        normalized_rewards = (rewards + scale) / (2 * scale)
        avg_normalized_reward = normalized_rewards.mean().item()
        
        # 更新 Beta 分布参数
        self.alpha = rho * self.alpha + avg_normalized_reward
        self.beta = rho * self.beta + (1 - avg_normalized_reward)
        
        return rho


def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """
    计算回复的奖励分数
    
    【奖励组成】
    1. Reward Model 分数: 使用外部模型评估回复质量
    2. 格式奖励 (reasoning=1): 正确使用推理格式给额外奖励
    
    【推理格式奖励】
    - 正确格式 <think>...</think><answer>...</answer>: +0.5
    - 每个正确的标签: +0.25
    
    【参数】
    - prompts: 输入提示列表
    - responses: 生成的回复列表
    - reward_model: 奖励模型
    - reward_tokenizer: 奖励模型的 tokenizer
    
    【返回】
    - rewards: [batch_size] 的奖励张量
    """
    def reasoning_model_reward(rewards):
        """计算推理格式奖励"""
        # 完整格式匹配
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        # 完整格式奖励
        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern or match_pattern2:
                format_rewards.append(0.5)  # 完整格式 +0.5
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
    
    # 添加格式奖励 (如果是推理模型)
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)

    # 使用 Reward Model 计算奖励
    with torch.no_grad():
        reward_model_scores = []
        scale = 3.0

        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            # 解析对话历史
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]

            # 构建完整对话
            tmp_chat = messages + [{"role": "assistant", "content": response}]
            score = reward_model.get_score(reward_tokenizer, tmp_chat)
            
            # 裁剪到 [-scale, scale]
            score = max(min(score, scale), -scale)

            # 对推理模型，额外评估 <answer> 内容
            if args.reasoning == 1:
                answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                    answer_score = max(min(answer_score, scale), -scale)
                    # 加权组合: 40% 完整回复 + 60% 答案内容
                    score = score * 0.4 + answer_score * 0.6

            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def spo_train_epoch(epoch, loader, iters, ref_model, reward_model, reward_tokenizer, value_tracker, start_step=0, wandb=None):
    """
    SPO 训练一个 epoch
    
    【SPO 训练流程】
    for each batch:
        1. 生成回复: response = model.generate(prompt)
        2. 计算奖励: reward = reward_model(prompt, response)
        3. 获取基线: baseline = value_tracker.get_baselines()
        4. 计算优势: advantage = reward - baseline
        5. 计算 KL 惩罚: kl = ref_logprob - cur_logprob
        6. 计算损失: loss = -logprob * advantage + β * kl
        7. 更新策略: optimizer.step()
        8. 更新基线: value_tracker.update(reward)
    
    【与 GRPO 的区别】
    - GRPO: 每个 prompt 生成 N 个回复，优势 = (r - group_mean) / group_std
    - SPO: 每个 prompt 生成 1 个回复，优势 = r - adaptive_baseline
    
    【参数说明】
    - epoch: 当前 epoch
    - loader: 数据加载器
    - iters: 迭代次数
    - ref_model: 参考模型 (冻结)
    - reward_model: 奖励模型
    - reward_tokenizer: 奖励模型 tokenizer
    - value_tracker: 自适应基线追踪器
    - start_step: 起始步数
    - wandb: 实验追踪
    """
    for step, batch in enumerate(loader, start=start_step + 1):
        # ==================== 准备输入 ====================
        prompts = batch['prompt']  # list[str], 长度 B
        
        # Token 化 prompt
        prompt_inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            return_token_type_ids=False,
            padding_side="left",  # 左填充，让生成从右边开始
            add_special_tokens=False
        ).to(args.device)
        
        # 截断到最大长度
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # ==================== 生成回复 ====================
        with torch.no_grad():
            # DDP 模型需要访问 .module
            model_for_gen = model.module if isinstance(model, DistributedDataParallel) else model
            outputs = model_for_gen.generate(
                **prompt_inputs, 
                max_new_tokens=args.max_gen_len, 
                do_sample=True, 
                temperature=0.8,
                num_return_sequences=1,  # SPO 每 prompt 只生成一个
                pad_token_id=tokenizer.pad_token_id
            )

        # 提取生成的部分 (去掉 prompt)
        completion_ids = outputs[:, prompt_inputs["input_ids"].size(1):]

        # ==================== 计算 log 概率 ====================
        def get_per_token_logps(mdl, input_ids, n_keep):
            """计算每个 token 的 log 概率"""
            input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
            logits = mdl(input_ids, logits_to_keep=n_keep + 1).logits[:, :-1, :]
            per_token_logps = []
            for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
                ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
                per_token_logps.append(
                    torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
                )
            return torch.stack(per_token_logps)

        # 策略模型的 log 概率
        with autocast_ctx:
            per_token_logps = get_per_token_logps(model, outputs, completion_ids.size(1))
            res = model(outputs) if lm_config.use_moe else None
            aux_loss = res.aux_loss if res is not None else torch.tensor(0.0, device=args.device)
        
        # 参考模型的 log 概率 (用于 KL 惩罚)
        with torch.no_grad():
            ref_per_token_logps = get_per_token_logps(ref_model, outputs, completion_ids.size(1))

        # ==================== 计算奖励 ====================
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        rewards = calculate_rewards(prompts, completions, reward_model, reward_tokenizer).to(args.device)

        # ==================== 计算基线和优势 ====================
        baselines = value_tracker.get_baselines(len(prompts)).to(args.device)

        scale = 3.0
        # 反归一化基线到原始奖励尺度 [-3, 3]
        unnormalized_baselines = baselines * (2 * scale) - scale
        
        # 优势 = 奖励 - 基线
        advantages = rewards - unnormalized_baselines
        
        # 裁剪优势防止梯度爆炸
        advantages = advantages.clamp(-5.0, 5.0)

        # ==================== 构建回复掩码 ====================
        # 找到 EOS token 位置
        is_eos = completion_ids == tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        
        # 创建掩码: EOS 之前(包含)为 1
        completion_mask = (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()

        # ==================== 计算 KL 散度 ====================
        kl_div = ref_per_token_logps - per_token_logps
        # 使用近似 KL: E[exp(diff) - diff - 1]
        per_token_kl = torch.exp(kl_div) - kl_div - 1

        # ==================== 计算 SPO 损失 ====================
        # L = -logprob * advantage + β * kl
        per_token_loss = -per_token_logps * advantages.unsqueeze(1) + args.beta * per_token_kl
        
        # 平均每个序列的损失
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        # 总损失 (加上 MoE 辅助损失)
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        # ==================== 更新基线 ====================
        response_masks = completion_mask.float()
        rho = value_tracker.update(rewards, per_token_logps.detach(), response_masks)

        # ==================== 参数更新 ====================
        if (step + 1) % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ==================== 日志记录 ====================
        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            kl_val = ((per_token_kl * completion_mask).sum() / (completion_mask.sum() + 1e-8)).item()
            avg_baseline_val = baselines.mean().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Actor Loss: {policy_loss_val:.4f}, Aux Loss: {current_aux_loss:.4f}, Reward: {avg_reward_val:.4f}, '
                   f'Baseline: {avg_baseline_val:.4f}, KL: {kl_val:.4f}, Rho: {rho:.4f}, '
                   f'Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')

            if wandb and is_main_process():
                wandb.log({
                    "policy_loss": policy_loss_val,
                    "aux_loss": current_aux_loss,
                    "reward": avg_reward_val,
                    "kl": kl_val,
                    "rho": float(rho),
                    "baseline": avg_baseline_val,
                    "advantages_mean": advantages.mean().item(),
                    "learning_rate": current_lr
                })

        # ==================== 保存检查点 ====================
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        # 及时释放内存
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, advantages, completion_mask, baselines, response_masks


if __name__ == "__main__":
    # ==================== 参数解析 ====================
    parser = argparse.ArgumentParser(description="MiniMind SPO (Self-Play Optimization)")
    
    # 保存配置
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='spo', type=str, help="保存权重的前缀名")
    
    # 训练配置
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-7, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    
    # 模型配置
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=66, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成的最大长度")
    
    # 数据和超参数
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", help="RLAIF数据路径")
    parser.add_argument("--beta", type=float, default=0.02, help="KL惩罚系数")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    
    # 实验追踪
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-SPO", help="wandb项目名")
    
    args = parser.parse_args()

    # ==================== 1. 初始化环境 ====================
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ==================== 2. 配置模型参数 ====================
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len + args.max_gen_len,
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
        wandb_run_name = f"MiniMind-SPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ==================== 5. 初始化模型 ====================
    base_weight = "reason" if args.reasoning == 1 else "full_sft"
    
    # Policy 模型 (需要训练)
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    
    # Reference 模型 (冻结，用于 KL 惩罚)
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    
    # Reward 模型 (冻结，用于评估回复质量)
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    
    # Value Tracker (自适应基线)
    value_tracker = AutoAdaptiveValueTracker(
        rho_mode='kl', 
        rho_const=0.9, 
        D_half=0.06, 
        clip_lower=0.5, 
        clip_upper=0.96
    )
    
    # ==================== 6. 数据和优化器 ====================
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 学习率调度
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    
    # ==================== 7. 从ckp恢复 ====================
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ==================== 8. DDP ====================
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ==================== 9. 训练 ====================
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        
        if epoch == start_epoch and start_step > 0:
            batch_sampler = SkipBatchSampler(train_sampler or range(len(train_ds)), args.batch_size, start_step + 1)
            loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            spo_train_epoch(epoch, loader, len(loader) + start_step + 1, ref_model, reward_model, reward_tokenizer, value_tracker, start_step, wandb)
        else:
            loader = DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True,
                              drop_last=False, shuffle=(train_sampler is None),
                              num_workers=args.num_workers, sampler=train_sampler)
            spo_train_epoch(epoch, loader, len(loader), ref_model, reward_model, reward_tokenizer, value_tracker, 0, wandb)
    
    # ==================== 10. 清理 ====================
    if dist.is_initialized():
        dist.destroy_process_group()
