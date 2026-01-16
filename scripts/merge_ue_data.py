"""
合并数据集脚本
- 合并Pretrain: UE代码 + 通用知识
- 合并SFT: UE问答 + 通用对话
"""
import os
import sys
import json
import random
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def merge_pretrain_datasets(ue_sample_ratio=0.3):
    """
    合并预训练数据集
    Args:
        ue_sample_ratio: UE数据采样比例 (0.3表示采样30%的UE数据，避免数据太大)
    """
    print("\n" + "="*50)
    print("合并Pretrain数据集")
    print("="*50)
    
    ue_pretrain = PROJECT_ROOT / "dataset" / "ue_pretrain.jsonl"
    general_pretrain = PROJECT_ROOT / "dataset" / "pretrain_hq.jsonl"
    output_file = PROJECT_ROOT / "dataset" / "ue_pretrain_merged.jsonl"
    
    all_data = []
    
    # 1. 加载通用预训练数据 (全部)
    if general_pretrain.exists():
        print(f"加载通用预训练数据: {general_pretrain}")
        with open(general_pretrain, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line))
        print(f"  -> {len(all_data)} 条通用知识")
    else:
        print(f"[错误] 通用预训练数据不存在: {general_pretrain}")
        return None
    
    general_count = len(all_data)
    
    # 2. 加载UE预训练数据 (采样)
    if ue_pretrain.exists():
        print(f"加载UE代码数据: {ue_pretrain}")
        ue_data = []
        with open(ue_pretrain, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    ue_data.append(json.loads(line))
        
        # 采样
        sample_size = int(len(ue_data) * ue_sample_ratio)
        ue_sampled = random.sample(ue_data, sample_size)
        all_data.extend(ue_sampled)
        print(f"  -> {len(ue_data)} 条UE代码，采样 {sample_size} 条 ({int(ue_sample_ratio*100)}%)")
    else:
        print(f"[警告] UE预训练数据不存在: {ue_pretrain}")
    
    # 3. 打乱并保存
    print(f"\n合并后共 {len(all_data)} 条数据")
    random.shuffle(all_data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"保存至: {output_file}")
    print(f"\n数据构成:")
    print(f"  - 通用知识: {general_count} 条")
    print(f"  - UE代码: {len(all_data)-general_count} 条")
    
    return output_file


def merge_sft_datasets():
    """合并SFT数据集"""
    print("\n" + "="*50)
    print("合并SFT数据集")
    print("="*50)
    
    ue_sft = PROJECT_ROOT / "dataset" / "ue_sft.jsonl"
    general_sft = PROJECT_ROOT / "dataset" / "sft_2048.jsonl"
    output_file = PROJECT_ROOT / "dataset" / "ue_sft_merged.jsonl"
    
    all_data = []
    
    # 1. 加载UE SFT数据
    if ue_sft.exists():
        print(f"加载UE SFT数据: {ue_sft}")
        with open(ue_sft, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line))
        print(f"  -> {len(all_data)} 条UE问答")
    else:
        print(f"[警告] UE SFT数据不存在: {ue_sft}")
    
    ue_count = len(all_data)
    
    # 2. 加载通用SFT数据
    if general_sft.exists():
        print(f"加载通用SFT数据: {general_sft}")
        general_count = 0
        with open(general_sft, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line))
                    general_count += 1
        print(f"  -> {general_count} 条通用对话")
    else:
        print(f"[错误] 通用SFT数据不存在: {general_sft}")
        print("请先下载 sft_2048.jsonl:")
        print("  https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files")
        return None
    
    # 3. 打乱顺序
    print(f"\n合并后共 {len(all_data)} 条数据")
    random.shuffle(all_data)
    
    # 4. 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"保存至: {output_file}")
    print(f"\n数据构成:")
    print(f"  - UE问答: {ue_count} 条")
    print(f"  - 通用对话: {len(all_data)-ue_count} 条")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="合并数据集")
    parser.add_argument('--type', choices=['pretrain', 'sft', 'all'], default='all',
                        help="合并类型")
    parser.add_argument('--ue_ratio', type=float, default=0.3,
                        help="UE预训练数据采样比例 (默认0.3)")
    args = parser.parse_args()
    
    if args.type in ['pretrain', 'all']:
        merge_pretrain_datasets(args.ue_ratio)
    
    if args.type in ['sft', 'all']:
        merge_sft_datasets()
    
    print("\n" + "="*50)
    print("✅ 数据合并完成！")
    print("="*50)


if __name__ == '__main__':
    main()
