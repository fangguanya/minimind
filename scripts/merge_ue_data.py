"""
合并数据集脚本
- 合并Pretrain: UE代码 + 通用知识
- 合并SFT: UE问答 + 通用对话
支持选择不同的数据集版本
"""
import os
import sys
import json
import random
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# 数据集配置
PRETRAIN_DATASETS = {
    'hq': 'pretrain_hq.jsonl',      # 高质量预训练数据 (~140万条)
}

SFT_DATASETS = {
    'mini': 'sft_mini_512.jsonl',   # 精简版 (~120万条, 1.2GB)
    '512': 'sft_512.jsonl',         # 完整512版 (~680万条, 7.2GB)
    '1024': 'sft_1024.jsonl',       # 1024版 (~420万条, 5.3GB)
    '2048': 'sft_2048.jsonl',       # 2048版 (~540万条, 8.5GB)
}


def merge_pretrain_datasets(pretrain_dataset='hq', ue_sample_ratio=0.3):
    """
    合并预训练数据集
    Args:
        pretrain_dataset: 预训练数据集版本 (hq)
        ue_sample_ratio: UE数据采样比例 (0.3表示采样30%的UE数据)
    """
    print("\n" + "="*50)
    print("合并Pretrain数据集")
    print("="*50)
    
    ue_pretrain = PROJECT_ROOT / "dataset" / "ue_pretrain.jsonl"
    
    # 选择预训练数据集
    if pretrain_dataset not in PRETRAIN_DATASETS:
        print(f"[错误] 未知的预训练数据集: {pretrain_dataset}")
        print(f"可选: {list(PRETRAIN_DATASETS.keys())}")
        return None
    
    general_pretrain = PROJECT_ROOT / "dataset" / PRETRAIN_DATASETS[pretrain_dataset]
    output_file = PROJECT_ROOT / "dataset" / "ue_pretrain_merged.jsonl"
    
    print(f"使用预训练数据集: {pretrain_dataset} -> {PRETRAIN_DATASETS[pretrain_dataset]}")
    
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


def merge_sft_datasets(sft_dataset='mini', ue_repeat=1, general_sample_ratio=1.0):
    """
    合并SFT数据集
    Args:
        sft_dataset: SFT数据集版本 (mini, 512, 1024, 2048)
        ue_repeat: UE数据重复次数 (增加UE数据权重)
        general_sample_ratio: 通用数据采样比例 (减少通用数据)
    """
    print("\n" + "="*50)
    print("合并SFT数据集")
    print("="*50)
    
    ue_sft = PROJECT_ROOT / "dataset" / "ue_sft.jsonl"
    
    # 选择SFT数据集
    if sft_dataset not in SFT_DATASETS:
        print(f"[错误] 未知的SFT数据集: {sft_dataset}")
        print(f"可选: {list(SFT_DATASETS.keys())}")
        return None
    
    general_sft = PROJECT_ROOT / "dataset" / SFT_DATASETS[sft_dataset]
    output_file = PROJECT_ROOT / "dataset" / "ue_sft_merged.jsonl"
    
    print(f"使用SFT数据集: {sft_dataset} -> {SFT_DATASETS[sft_dataset]}")
    print(f"UE数据重复: {ue_repeat}x, 通用数据采样: {int(general_sample_ratio*100)}%")
    
    all_data = []
    ue_data = []
    
    # 1. 加载UE SFT数据
    if ue_sft.exists():
        print(f"加载UE SFT数据: {ue_sft}")
        with open(ue_sft, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    ue_data.append(json.loads(line))
        print(f"  -> {len(ue_data)} 条UE问答")
        
        # 重复UE数据以增加权重
        for _ in range(ue_repeat):
            all_data.extend(ue_data)
        print(f"  -> 重复 {ue_repeat}x 后共 {len(all_data)} 条")
    else:
        print(f"[警告] UE SFT数据不存在: {ue_sft}")
    
    ue_count = len(all_data)
    
    # 2. 加载通用SFT数据 (可采样)
    if general_sft.exists():
        print(f"加载通用SFT数据: {general_sft}")
        general_data = []
        with open(general_sft, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    general_data.append(json.loads(line))
        
        # 采样通用数据
        if general_sample_ratio < 1.0:
            sample_size = int(len(general_data) * general_sample_ratio)
            general_data = random.sample(general_data, sample_size)
        
        all_data.extend(general_data)
        print(f"  -> {len(general_data)} 条通用对话 (采样 {int(general_sample_ratio*100)}%)")
    else:
        print(f"[错误] 通用SFT数据不存在: {general_sft}")
        print(f"请先下载 {SFT_DATASETS[sft_dataset]}:")
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
    parser.add_argument('--sft_dataset', choices=['mini', '512', '1024', '2048'], default='mini',
                        help="SFT数据集版本: mini=精简版(120万), 512=完整版(680万), 1024=(420万), 2048=(540万)")
    parser.add_argument('--pretrain_dataset', choices=['hq'], default='hq',
                        help="预训练数据集版本: hq=高质量(140万)")
    parser.add_argument('--ue_repeat', type=int, default=1,
                        help="UE SFT数据重复次数 (增加UE数据权重, 默认1)")
    parser.add_argument('--general_sample', type=float, default=1.0,
                        help="通用SFT数据采样比例 (0.1=10%%, 默认1.0=100%%)")
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("数据集合并配置")
    print("="*50)
    print(f"  Pretrain数据集: {args.pretrain_dataset}")
    print(f"  SFT数据集: {args.sft_dataset}")
    print(f"  UE Pretrain采样: {args.ue_ratio}")
    print(f"  UE SFT重复: {args.ue_repeat}x")
    print(f"  通用SFT采样: {int(args.general_sample*100)}%")
    
    if args.type in ['pretrain', 'all']:
        merge_pretrain_datasets(args.pretrain_dataset, args.ue_ratio)
    
    if args.type in ['sft', 'all']:
        merge_sft_datasets(args.sft_dataset, args.ue_repeat, args.general_sample)
    
    print("\n" + "="*50)
    print("✅ 数据合并完成！")
    print("="*50)


if __name__ == '__main__':
    main()
