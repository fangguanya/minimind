"""
UnrealEngine 代码助手训练主脚本
================================

这是一个一站式训练脚本，包含：
1. 数据准备
2. 预训练（Pretrain）
3. 监督微调（SFT）
4. 评估测试

使用方法：
python train_ue_assistant.py --ue_source_path "D:/UnrealEngine/Engine/Source" --stage all
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# 添加项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))


def run_command(cmd: str, description: str = ""):
    """运行命令"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"命令: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, cwd=str(PROJECT_ROOT))
    
    if result.returncode != 0:
        print(f"❌ 命令执行失败: {description}")
        return False
    
    print(f"✅ {description} 完成!")
    return True


def prepare_data(args):
    """准备数据"""
    print("\n" + "="*60)
    print("📦 Step 1: 准备训练数据")
    print("="*60)
    
    # 1.1 准备Pretrain数据
    if not os.path.exists(f"{PROJECT_ROOT}/dataset/ue_pretrain.jsonl") or args.force_regenerate:
        cmd = (
            f"python scripts/prepare_ue_pretrain_data.py "
            f"--ue_source_path \"{args.ue_source_path}\" "
            f"--output_path dataset/ue_pretrain.jsonl "
            f"--chunk_size {args.chunk_size} "
            f"--max_file_size {args.max_file_size}"
        )
        if args.max_pretrain_samples:
            cmd += f" --max_samples {args.max_pretrain_samples}"
        
        run_command(cmd, "生成Pretrain数据集")
    else:
        print("📁 Pretrain数据集已存在，跳过生成")
    
    # 1.2 准备SFT数据
    if not os.path.exists(f"{PROJECT_ROOT}/dataset/ue_sft.jsonl") or args.force_regenerate:
        cmd = (
            f"python scripts/prepare_ue_sft_data.py "
            f"--ue_source_path \"{args.ue_source_path}\" "
            f"--output_path dataset/ue_sft.jsonl"
        )
        if args.max_sft_files:
            cmd += f" --max_files {args.max_sft_files}"
        
        run_command(cmd, "生成SFT数据集")
    else:
        print("📁 SFT数据集已存在，跳过生成")
    
    # 1.3 合并通用SFT数据（可选）
    if args.include_general_sft and os.path.exists(f"{PROJECT_ROOT}/dataset/sft_mini_512.jsonl"):
        print("📎 将合并通用SFT数据以保持通用对话能力")
        merge_datasets(
            [f"{PROJECT_ROOT}/dataset/ue_sft.jsonl", 
             f"{PROJECT_ROOT}/dataset/sft_mini_512.jsonl"],
            f"{PROJECT_ROOT}/dataset/ue_sft_merged.jsonl"
        )


def merge_datasets(input_files: list, output_file: str):
    """合并多个数据集"""
    import json
    
    all_data = []
    for f in input_files:
        if os.path.exists(f):
            with open(f, 'r', encoding='utf-8') as fp:
                for line in fp:
                    if line.strip():
                        all_data.append(json.loads(line))
    
    import random
    random.shuffle(all_data)
    
    with open(output_file, 'w', encoding='utf-8') as fp:
        for item in all_data:
            fp.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 合并完成: {len(all_data)} 条数据 -> {output_file}")


def run_pretrain(args):
    """运行预训练"""
    print("\n" + "="*60)
    print("🎓 Step 2: 预训练 (Pretrain)")
    print("="*60)
    
    # 检查数据
    pretrain_data = f"{PROJECT_ROOT}/dataset/ue_pretrain_chunked.jsonl"
    if not os.path.exists(pretrain_data):
        pretrain_data = f"{PROJECT_ROOT}/dataset/ue_pretrain.jsonl"
    
    if not os.path.exists(pretrain_data):
        print("❌ 预训练数据不存在，请先运行数据准备阶段")
        return False
    
    cmd = (
        f"python trainer/train_pretrain.py "
        f"--data_path {pretrain_data} "
        f"--hidden_size {args.hidden_size} "
        f"--num_hidden_layers {args.num_hidden_layers} "
        f"--epochs {args.pretrain_epochs} "
        f"--batch_size {args.batch_size} "
        f"--learning_rate {args.pretrain_lr} "
        f"--max_seq_len {args.max_seq_len} "
        f"--save_weight ue_pretrain "
        f"--log_interval 50 "
        f"--save_interval 500"
    )
    
    if args.use_wandb:
        cmd += " --use_wandb --wandb_project MiniMind-UE-Pretrain"
    
    return run_command(cmd, "UE代码预训练")


def run_sft(args):
    """运行SFT微调"""
    print("\n" + "="*60)
    print("💬 Step 3: 监督微调 (SFT)")
    print("="*60)
    
    # 选择SFT数据
    if args.include_general_sft and os.path.exists(f"{PROJECT_ROOT}/dataset/ue_sft_merged.jsonl"):
        sft_data = f"{PROJECT_ROOT}/dataset/ue_sft_merged.jsonl"
    else:
        sft_data = f"{PROJECT_ROOT}/dataset/ue_sft.jsonl"
    
    if not os.path.exists(sft_data):
        print("❌ SFT数据不存在，请先运行数据准备阶段")
        return False
    
    cmd = (
        f"python trainer/train_full_sft.py "
        f"--data_path {sft_data} "
        f"--hidden_size {args.hidden_size} "
        f"--num_hidden_layers {args.num_hidden_layers} "
        f"--epochs {args.sft_epochs} "
        f"--batch_size {args.batch_size} "
        f"--learning_rate {args.sft_lr} "
        f"--max_seq_len {args.max_seq_len} "
        f"--from_weight ue_pretrain "
        f"--save_weight ue_sft "
        f"--log_interval 50 "
        f"--save_interval 500"
    )
    
    if args.use_wandb:
        cmd += " --use_wandb --wandb_project MiniMind-UE-SFT"
    
    return run_command(cmd, "UE助手SFT微调")


def run_eval(args):
    """评估模型"""
    print("\n" + "="*60)
    print("🧪 Step 4: 模型评估")
    print("="*60)
    
    # 测试问题
    test_questions = [
        "什么是AActor类？",
        "UActorComponent和AActor有什么区别？",
        "如何在UE中实现Tick功能？",
        "UPROPERTY宏有哪些常用参数？",
        "UE中如何创建定时器？",
        "什么是UE的反射系统？",
        "ACharacter类在哪个文件中定义？",
    ]
    
    print("\n📝 测试问题列表:")
    for i, q in enumerate(test_questions, 1):
        print(f"  {i}. {q}")
    
    print("\n启动交互式测试...")
    
    cmd = (
        f"python eval_llm.py "
        f"--load_from model "
        f"--weight ue_sft "
        f"--hidden_size {args.hidden_size} "
        f"--num_hidden_layers {args.num_hidden_layers}"
    )
    
    return run_command(cmd, "模型评估")


def print_training_plan(args):
    """打印训练计划"""
    print("\n" + "="*60)
    print("📋 UE代码助手训练计划")
    print("="*60)
    
    print(f"""
🔧 训练配置:
    - UE源代码路径: {args.ue_source_path}
    - 模型隐藏层维度: {args.hidden_size}
    - 模型层数: {args.num_hidden_layers}
    - 最大序列长度: {args.max_seq_len}
    - Batch Size: {args.batch_size}
    
📊 训练阶段:
    Stage 1: 数据准备
        - 提取UE源代码 -> ue_pretrain.jsonl
        - 生成QA对 -> ue_sft.jsonl
        
    Stage 2: 预训练 ({args.pretrain_epochs} epochs, lr={args.pretrain_lr})
        - 学习UE代码风格和知识
        - 输出: out/ue_pretrain_{args.hidden_size}.pth
        
    Stage 3: SFT微调 ({args.sft_epochs} epochs, lr={args.sft_lr})
        - 学习问答对话格式
        - 输出: out/ue_sft_{args.hidden_size}.pth
        
    Stage 4: 评估测试
        - 交互式问答测试
        
💡 预计资源消耗:
    - 显存: ~4-8GB (根据模型大小)
    - 预训练时间: 取决于数据量
    - SFT时间: 通常较快
""")


def main():
    parser = argparse.ArgumentParser(description="UE代码助手一站式训练脚本")
    
    # 数据相关
    parser.add_argument('--ue_source_path', type=str, required=True,
                        help="UE源代码目录路径 (例如: D:/UnrealEngine/Engine/Source)")
    parser.add_argument('--force_regenerate', action='store_true',
                        help="强制重新生成数据集")
    parser.add_argument('--max_pretrain_samples', type=int, default=None,
                        help="最大预训练样本数")
    parser.add_argument('--max_sft_files', type=int, default=None,
                        help="最大SFT处理文件数")
    parser.add_argument('--chunk_size', type=int, default=512,
                        help="代码块大小")
    parser.add_argument('--max_file_size', type=int, default=100,
                        help="最大文件大小(KB)")
    parser.add_argument('--include_general_sft', action='store_true',
                        help="包含通用SFT数据以保持对话能力")
    
    # 模型相关
    parser.add_argument('--hidden_size', type=int, default=768,
                        help="隐藏层维度 (512=Small, 768=Base)")
    parser.add_argument('--num_hidden_layers', type=int, default=16,
                        help="隐藏层数量 (8=Small, 16=Base)")
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help="最大序列长度")
    
    # 训练相关
    parser.add_argument('--pretrain_epochs', type=int, default=2,
                        help="预训练轮数")
    parser.add_argument('--sft_epochs', type=int, default=3,
                        help="SFT轮数")
    parser.add_argument('--pretrain_lr', type=float, default=5e-4,
                        help="预训练学习率")
    parser.add_argument('--sft_lr', type=float, default=1e-5,
                        help="SFT学习率")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch大小")
    parser.add_argument('--use_wandb', action='store_true',
                        help="使用WandB记录训练")
    
    # 执行阶段
    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', 'data', 'pretrain', 'sft', 'eval', 'plan'],
                        help="执行阶段")
    
    args = parser.parse_args()
    
    # 检查路径
    if not os.path.exists(args.ue_source_path) and args.stage not in ['plan', 'eval']:
        print(f"❌ UE源代码路径不存在: {args.ue_source_path}")
        print("\n请确保路径正确，例如:")
        print("  - Windows: D:/UnrealEngine/Engine/Source")
        print("  - Linux: /home/user/UnrealEngine/Engine/Source")
        return
    
    # 打印训练计划
    print_training_plan(args)
    
    if args.stage == 'plan':
        return
    
    # 执行各阶段
    stages_to_run = {
        'all': ['data', 'pretrain', 'sft', 'eval'],
        'data': ['data'],
        'pretrain': ['pretrain'],
        'sft': ['sft'],
        'eval': ['eval'],
    }
    
    stages = stages_to_run[args.stage]
    
    if 'data' in stages:
        prepare_data(args)
    
    if 'pretrain' in stages:
        if not run_pretrain(args):
            print("预训练失败，停止后续阶段")
            return
    
    if 'sft' in stages:
        if not run_sft(args):
            print("SFT失败，停止后续阶段")
            return
    
    if 'eval' in stages:
        run_eval(args)
    
    print("\n" + "="*60)
    print("🎉 所有阶段完成!")
    print("="*60)
    print(f"""
📁 生成的文件:
    - dataset/ue_pretrain.jsonl (预训练数据)
    - dataset/ue_sft.jsonl (SFT数据)
    - out/ue_pretrain_{args.hidden_size}.pth (预训练模型)
    - out/ue_sft_{args.hidden_size}.pth (最终模型)

🚀 使用模型:
    python eval_llm.py --weight ue_sft --hidden_size {args.hidden_size}
    
💡 进一步优化建议:
    1. 增加更多高质量手动QA对
    2. 使用LoRA微调特定领域
    3. 考虑结合RAG提升检索准确性
""")


if __name__ == '__main__':
    main()
