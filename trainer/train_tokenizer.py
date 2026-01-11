"""
================================================================================
                    MiniMind 分词器训练脚本 (Tokenizer Training)
================================================================================

【什么是分词器 (Tokenizer)】
分词器将文本转换为模型能理解的数字 (token ID):
- "你好世界" → [1234, 5678] (示例)
- 是 LLM 的"翻译器"

【为什么需要训练分词器】
1. 通用分词器可能不适合特定领域
2. 中文处理需要专门优化
3. 可以控制词表大小

【本脚本使用 SentencePiece + BPE】

BPE (Byte Pair Encoding) 算法:
1. 从单个字符开始
2. 统计相邻字符对的频率
3. 合并最频繁的字符对
4. 重复直到达到目标词表大小

例如:
初始: ['l', 'o', 'w', 'e', 'r']
→ 合并 'l'+'o' → ['lo', 'w', 'e', 'r']
→ 合并 'lo'+'w' → ['low', 'e', 'r']
...

【输出格式】
训练完成后生成两个文件:
1. tokenizer.model: SentencePiece 模型文件
2. tokenizer.json: HuggingFace 格式的 JSON 配置

【词表大小选择】
- 小词表 (6400): 适合小模型，推理快
- 中词表 (16000): 平衡选择
- 大词表 (32000+): 适合大模型，OOV 少

【特殊 Token】
- [PAD]: 填充 token，用于对齐序列长度
- [UNK]: 未知 token，处理词表外的词
- [BOS]/[EOS]: 序列开始/结束标记
- <|im_start|>/<|im_end|>: ChatML 格式的对话标记

【使用方法】
python train_tokenizer.py --data_path ../dataset/pretrain_hq.jsonl --vocab_size 6400
"""

import os
import sys

# 设置包名，确保相对导入正常工作
__package__ = "trainer"
# 将父目录添加到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import sentencepiece as spm
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from datasets import load_dataset

from trainer.trainer_utils import Logger


def train(data_path, out_path, vocab_size=6400):
    """
    训练 SentencePiece BPE 分词器
    
    【训练流程】
    1. 加载训练数据 (JSONL 格式)
    2. 提取所有文本
    3. 保存为临时文件
    4. 调用 SentencePiece 训练
    5. 转换为 HuggingFace 格式
    
    【参数】
    - data_path: JSONL 数据文件路径 (每行包含 'text' 字段)
    - out_path: 输出目录
    - vocab_size: 词表大小
    
    【SentencePiece 配置说明】
    - model_type='bpe': 使用 BPE 算法
    - character_coverage=0.9995: 覆盖 99.95% 的字符
    - user_defined_symbols: 预定义的特殊符号
    - pad_id=0: padding token ID
    - unk_id=1: unknown token ID
    - bos_id=2: begin-of-sequence ID
    - eos_id=3: end-of-sequence ID
    """
    Logger(f'开始训练分词器: 数据={data_path}, 词表大小={vocab_size}')
    
    # ==================== 1. 加载和准备数据 ====================
    Logger('正在加载数据...')
    # 使用 HuggingFace datasets 加载 JSONL 数据
    samples = load_dataset('json', data_files=data_path, split='train')
    
    # 将所有文本合并并保存到临时文件
    # SentencePiece 需要从文件读取训练数据
    temp_file = os.path.join(out_path, 'temp_train.txt')
    os.makedirs(out_path, exist_ok=True)
    
    with open(temp_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            text = str(sample.get('text', ''))
            if text.strip():  # 跳过空文本
                f.write(text + '\n')
    
    Logger(f'数据准备完成，共 {len(samples)} 条样本')
    
    # ==================== 2. 定义特殊 Token ====================
    # 这些是 ChatML 格式需要的特殊标记
    # <|im_start|>user\n ... <|im_end|>\n
    special_tokens = [
        '<|im_start|>',  # 对话轮次开始
        '<|im_end|>',    # 对话轮次结束
    ]
    
    # ==================== 3. 训练 SentencePiece 模型 ====================
    Logger('正在训练 SentencePiece 模型...')
    
    model_prefix = os.path.join(out_path, 'tokenizer')
    
    # SentencePiece 训练参数
    # 详细文档: https://github.com/google/sentencepiece
    spm.SentencePieceTrainer.train(
        input=temp_file,                    # 训练数据文件
        model_prefix=model_prefix,          # 输出模型前缀
        vocab_size=vocab_size,              # 词表大小
        model_type='bpe',                   # 使用 BPE 算法
        character_coverage=0.9995,          # 字符覆盖率 (中文需要高覆盖)
        user_defined_symbols=special_tokens,# 预定义特殊符号
        pad_id=0,                           # <pad> token ID
        unk_id=1,                           # <unk> token ID
        bos_id=2,                           # <s> (bos) token ID
        eos_id=3,                           # </s> (eos) token ID
        max_sentence_length=8192,           # 最大句子长度
        num_threads=os.cpu_count(),         # 使用所有 CPU 核心
        train_extremely_large_corpus=False  # 如果数据量大，设为 True
    )
    
    Logger('SentencePiece 模型训练完成')
    
    # ==================== 4. 转换为 HuggingFace 格式 ====================
    Logger('正在转换为 HuggingFace tokenizers 格式...')
    
    # 加载训练好的 SentencePiece 模型
    sp_model = spm.SentencePieceProcessor()
    sp_model.load(model_prefix + '.model')
    
    # 提取词表 (token -> ID 映射)
    vocab = {}
    for i in range(sp_model.get_piece_size()):
        token = sp_model.id_to_piece(i)
        vocab[token] = i
    
    # 提取 BPE merges (合并规则)
    # merges 记录了 BPE 训练过程中的合并顺序
    merges = []
    for i in range(sp_model.get_piece_size()):
        piece = sp_model.id_to_piece(i)
        # 跳过特殊 token 和单字符
        if piece.startswith('<') or piece.startswith('▁') or len(piece) <= 1:
            continue
        # 对于多字符 piece，找到合并点
        # 这是一个简化的实现，实际 merge 顺序由 SentencePiece 内部决定
        # 这里我们按字符顺序拆分
        for j in range(1, len(piece)):
            left = piece[:j]
            right = piece[j:]
            if left in vocab and right in vocab:
                merges.append((left, right))
                break
    
    # 创建 HuggingFace BPE tokenizer
    tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges))
    
    # 设置 pre-tokenizer (分词前处理)
    # Metaspace: 用 ▁ 替换空格，用于标记词边界
    tokenizer.pre_tokenizer = Metaspace()
    tokenizer.decoder = MetaspaceDecoder()
    
    # 保存为 JSON 格式
    tokenizer_json_path = os.path.join(out_path, 'tokenizer.json')
    tokenizer.save(tokenizer_json_path)
    
    Logger(f'分词器训练完成！')
    Logger(f'  - SentencePiece 模型: {model_prefix}.model')
    Logger(f'  - HuggingFace 格式: {tokenizer_json_path}')
    Logger(f'  - 词表大小: {sp_model.get_piece_size()}')
    
    # ==================== 5. 清理临时文件 ====================
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    # ==================== 6. 验证分词器 ====================
    Logger('验证分词器...')
    test_text = "你好，这是一个测试文本。Hello, this is a test."
    
    # SentencePiece 编码
    sp_encoded = sp_model.encode(test_text, out_type=str)
    Logger(f'  测试文本: "{test_text}"')
    Logger(f'  SentencePiece 分词: {sp_encoded}')
    Logger(f'  Token 数量: {len(sp_encoded)}')
    
    return model_prefix + '.model', tokenizer_json_path


if __name__ == "__main__":
    # ==================== 参数解析 ====================
    parser = argparse.ArgumentParser(description="Train a SentencePiece BPE tokenizer")
    
    # 数据路径
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="../dataset/pretrain_hq.jsonl",
        help="JSONL 训练数据路径，每行包含 'text' 字段"
    )
    
    # 输出目录
    parser.add_argument(
        "--out_path", 
        type=str, 
        default="../model",
        help="输出目录，保存 tokenizer.model 和 tokenizer.json"
    )
    
    # 词表大小
    parser.add_argument(
        "--vocab_size", 
        type=int, 
        default=6400,
        help="词表大小 (推荐: 6400 小模型, 16000 中模型, 32000+ 大模型)"
    )
    
    args = parser.parse_args()
    
    # 开始训练
    train(
        data_path=args.data_path,
        out_path=args.out_path,
        vocab_size=args.vocab_size
    )
