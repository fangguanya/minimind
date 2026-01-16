"""
UnrealEngine 源代码预处理脚本 - 生成Pretrain数据集
=================================================

功能：
1. 遍历UE源代码目录，提取.h/.cpp/.cs等文件
2. 清洗和格式化代码
3. 生成pretrain格式的jsonl文件

使用方法：
python prepare_ue_pretrain_data.py --ue_source_path "D:/UnrealEngine/Engine/Source" --output_path "../dataset/ue_pretrain.jsonl"
"""

import os
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional
import hashlib


class UECodeExtractor:
    """UnrealEngine代码提取器"""
    
    # 支持的代码文件扩展名
    CODE_EXTENSIONS = {'.h', '.hpp', '.cpp', '.c', '.cs', '.inl', '.ush', '.usf'}
    
    # 需要跳过的目录
    SKIP_DIRS = {
        'ThirdParty', 'Intermediate', 'Binaries', 'Saved', 
        'DerivedDataCache', '.git', '.vs', '__pycache__',
        'Documentation', 'Extras'
    }
        
    def __init__(self, ue_source_path: str, max_file_size: int = 100 * 1024 * 1024):
        self.ue_source_path = Path(ue_source_path)
        self.max_file_size = max_file_size  # 默认100KB
        self.seen_hashes = set()  # 用于去重
        
    def should_skip_dir(self, dir_path: Path) -> bool:
        """检查是否应跳过该目录"""
        for skip in self.SKIP_DIRS:
            if skip in dir_path.parts:
                return True
        return False
    
    def extract_file_info(self, file_path: Path) -> Optional[dict]:
        """提取单个文件的信息"""
        try:
            # 检查文件大小
            if file_path.stat().st_size > self.max_file_size:
                return None
            
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # 内容去重
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in self.seen_hashes:
                return None
            self.seen_hashes.add(content_hash)
            
            # 清洗代码
            cleaned_content = self.clean_code(content)
            
            # 过滤太短的文件
            if len(cleaned_content.strip()) < 100:
                return None
            
            # 获取相对路径
            try:
                rel_path = file_path.relative_to(self.ue_source_path)
            except ValueError:
                rel_path = file_path.name
            
            return {
                'file_path': str(rel_path),
                'content': cleaned_content,
                'extension': file_path.suffix,
            }
            
        except Exception as e:
            return None
    
    def clean_code(self, content: str) -> str:
        """清洗代码内容"""
        # 移除版权声明头部（通常很长且重复）
        lines = content.split('\n')
        
        # 跳过开头的版权声明块
        start_idx = 0
        in_comment = False
        for i, line in enumerate(lines[:50]):  # 只检查前50行
            stripped = line.strip()
            if stripped.startswith('/*'):
                in_comment = True
            if in_comment:
                if '*/' in stripped:
                    in_comment = False
                    start_idx = i + 1
                continue
            if stripped.startswith('//') and ('Copyright' in line or 'License' in line):
                start_idx = i + 1
                continue
            if stripped and not stripped.startswith('//'):
                break
        
        content = '\n'.join(lines[start_idx:])
        
        # 压缩多余的空行（超过2个连续空行压缩为2个）
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # 移除行尾空白
        content = '\n'.join(line.rstrip() for line in content.split('\n'))
        
        return content.strip()
    
    def extract_class_info(self, content: str) -> List[str]:
        """提取类/结构体信息"""
        # 匹配UE风格的类定义
        patterns = [
            r'(UCLASS\([^)]*\)\s*class\s+\w*_API\s+(\w+))',  # UCLASS
            r'(USTRUCT\([^)]*\)\s*struct\s+\w*_API\s+(\w+))',  # USTRUCT  
            r'(UENUM\([^)]*\)\s*enum\s+(?:class\s+)?(\w+))',  # UENUM
            r'(class\s+\w*_API\s+(\w+)\s*(?::|{))',  # 普通类
        ]
        
        classes = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            classes.extend([m[1] if isinstance(m, tuple) else m for m in matches])
        
        return list(set(classes))
    
    def collect_files(self) -> List[Path]:
        """收集所有代码文件"""
        files = []
        
        print(f"扫描目录: {self.ue_source_path}")
        
        for root, dirs, filenames in os.walk(self.ue_source_path):
            root_path = Path(root)
            
            # 跳过不需要的目录
            if self.should_skip_dir(root_path):
                dirs.clear()
                continue
            
            # 过滤子目录
            dirs[:] = [d for d in dirs if d not in self.SKIP_DIRS]
            
            for filename in filenames:
                file_path = root_path / filename
                if file_path.suffix.lower() in self.CODE_EXTENSIONS:
                    files.append(file_path)
        
        print(f"找到 {len(files)} 个代码文件")
        return files
    
    def generate_pretrain_data(self, output_path: str, max_samples: int = None):
        """生成预训练数据"""
        files = self.collect_files()
        
        if max_samples:
            files = files[:max_samples]
        
        samples = []
        
        with tqdm(files, desc="处理文件") as pbar:
            for file_path in pbar:
                info = self.extract_file_info(file_path)
                if info:
                    # 构造pretrain格式的文本
                    # 包含文件路径信息，帮助模型学习文件组织结构
                    text = f"// File: {info['file_path']}\n{info['content']}"
                    
                    samples.append({
                        'text': text
                    })
                    pbar.set_postfix({'samples': len(samples)})
        
        # 保存为jsonl格式
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"\n生成完成！共 {len(samples)} 条数据")
        print(f"保存至: {output_path}")
        
        # 统计信息
        total_chars = sum(len(s['text']) for s in samples)
        print(f"总字符数: {total_chars:,}")
        print(f"估算tokens: {total_chars // 4:,} (按4字符/token估算)")
        
        return samples


def generate_code_chunks(samples: List[dict], max_length: int = 512) -> List[dict]:
    """
    将长代码文件切分为适合训练的chunks
    
    Args:
        samples: 原始样本列表
        max_length: 每个chunk的最大字符长度
    """
    chunks = []
    
    for sample in tqdm(samples, desc="切分chunks"):
        text = sample['text']
        
        if len(text) <= max_length:
            chunks.append({'text': text})
        else:
            # 按函数/类边界切分
            # 简单策略：按空行分割
            parts = re.split(r'\n\n+', text)
            
            current_chunk = ""
            for part in parts:
                if len(current_chunk) + len(part) + 2 <= max_length:
                    current_chunk += part + "\n\n"
                else:
                    if current_chunk.strip():
                        chunks.append({'text': current_chunk.strip()})
                    current_chunk = part + "\n\n"
            
            if current_chunk.strip():
                chunks.append({'text': current_chunk.strip()})
    
    return chunks


def main():
    parser = argparse.ArgumentParser(description="UE源代码预处理 - 生成Pretrain数据集")
    parser.add_argument('--ue_source_path', type=str, required=True,
                        help="UE源代码目录路径 (例如: D:/UnrealEngine/Engine/Source)")
    parser.add_argument('--output_path', type=str, default='../dataset/ue_pretrain.jsonl',
                        help="输出文件路径")
    parser.add_argument('--max_file_size', type=int, default=100,
                        help="最大文件大小(KB)")
    parser.add_argument('--max_samples', type=int, default=None,
                        help="最大样本数(用于测试)")
    parser.add_argument('--chunk_size', type=int, default=512,
                        help="每个训练样本的最大字符数")
    parser.add_argument('--no_chunk', action='store_true',
                        help="不进行chunk切分")
    
    args = parser.parse_args()
    
    # 检查路径
    if not os.path.exists(args.ue_source_path):
        print(f"错误: 路径不存在 - {args.ue_source_path}")
        return
    
    # 提取代码
    extractor = UECodeExtractor(
        args.ue_source_path,
        max_file_size=args.max_file_size * 1024
    )
    
    samples = extractor.generate_pretrain_data(
        args.output_path,
        max_samples=args.max_samples
    )
    
    # 可选：切分为小chunks
    if not args.no_chunk and samples:
        print(f"\n切分为 {args.chunk_size} 字符的chunks...")
        chunks = generate_code_chunks(samples, max_length=args.chunk_size)
        
        chunk_output = args.output_path.replace('.jsonl', '_chunked.jsonl')
        with open(chunk_output, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        print(f"Chunks保存至: {chunk_output}")
        print(f"共 {len(chunks)} 个chunks")


if __name__ == '__main__':
    main()
