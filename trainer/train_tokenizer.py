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

================================================================================
【BPE 的数学依据】为什么这样做是对的？
================================================================================

【信息论基础】

BPE 本质上是一种数据压缩算法，基于以下数学原理:

1. 【香农信息熵】
   H = -Σ p(x) × log₂(p(x))
   
   高频出现的符号应该用更短的编码表示
   → BPE 把高频字符对合并成一个符号，减少总 token 数

2. 【最小描述长度 (MDL)】
   总成本 = 词表成本 + 编码成本
   
   - 词表越大 → 词表成本↑，但编码成本↓ (每个词更短)
   - 词表越小 → 词表成本↓，但编码成本↑ (需要更多 token)
   
   BPE 在两者之间找平衡点

【为什么合并最频繁的？】

假设文本中 "th" 出现 1000 次，"zq" 出现 1 次:

合并 "th":
  - 词表 +1 个符号
  - 但文本中减少 1000 个 token！
  - 净收益: 很大

合并 "zq":
  - 词表 +1 个符号
  - 只减少 1 个 token
  - 净收益: 几乎为零

================================================================================
【你的疑问】合并后原字符还保留，不是浪费 ID 吗？
================================================================================

【疑问】
"合并了 'l'+'o'='lo'，但 'l' 和 'o' 还在词表里，这不是浪费了 2 个 ID？"

【回答】不是浪费，而是必须保留！

────────────────────────────────────────────────────────────────────────────────
【原因1】必须能表示所有可能的输入
────────────────────────────────────────────────────────────────────────────────

假设只保留合并后的 "lo"，删除 'l' 和 'o':

输入 "hello" → "hel" + "lo" ✓ 可以
输入 "lion"  → "l" + "ion"  ✗ 'l' 单独出现，无法表示！

【必须保留原字符】因为它们可能单独出现或与其他字符组合

────────────────────────────────────────────────────────────────────────────────
【原因2】分词是贪婪匹配，不是只用合并后的
────────────────────────────────────────────────────────────────────────────────

BPE 分词时的规则:
1. 尝试匹配最长的已知 token
2. 如果匹配不上，回退到更短的

例如词表: {'low', 'lo', 'l', 'o', 'w', 'e', 'r'}

分词 "lower":
  → 尝试匹配 "lower" → 不在词表 ✗
  → 尝试匹配 "lowe" → 不在词表 ✗
  → 尝试匹配 "low" → 在词表 ✓ → 取出 "low"
  → 剩余 "er" → 尝试匹配 "er" → 不在词表
  → 尝试匹配 "e" → 在词表 ✓ → 取出 "e"
  → 剩余 "r" → 在词表 ✓ → 取出 "r"
  → 结果: ["low", "e", "r"]

分词 "owl":
  → "owl" 不在 → "ow" 不在 → "o" 在 ✓
  → "wl" 不在 → "w" 在 ✓
  → "l" 在 ✓
  → 结果: ["o", "w", "l"]  ← 这里必须用原字符！

────────────────────────────────────────────────────────────────────────────────
【原因3】实际上很划算！
────────────────────────────────────────────────────────────────────────────────

【成本分析】

假设:
  - 原始字符数: 256 (ASCII) 或 ~5000 (常用中文)
  - 目标词表: 32000

合并操作:
  - 每次合并增加 1 个新 token
  - 从 256 到 32000，需要 ~31744 次合并

【实际效果】

文本 "the the the the" (16 字符):
  - 字符级: 16 个 token
  - BPE后: 4 个 token ("the" × 4)
  - 压缩率: 4倍！

词表中虽然同时有 't', 'h', 'e', 'th', 'the':
  - 高频词 "the" 用 1 个 token
  - 罕见词 "thx" 用 ['th', 'x'] 或 ['t', 'h', 'x']
  - 保证了灵活性 + 效率

────────────────────────────────────────────────────────────────────────────────
【类比】
────────────────────────────────────────────────────────────────────────────────

想象你是快递员，有不同大小的箱子:
  - 小箱 (单字符): 能装任何东西
  - 中箱 (2-3字符): 装常见组合更高效
  - 大箱 (完整词): 装高频词最高效

你不能只留大箱:
  - 大箱装不下的怎么办？
  - 必须保留小箱作为"兜底"

BPE 词表 = 各种大小的箱子
  - 高频词 → 用大箱 (效率高)
  - 罕见词 → 用小箱组合 (灵活性)
  - 未知词 → 回退到字符/字节 (兜底)

================================================================================

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
    
    # ════════════════════════════════════════════════════════════════════
    # 【为什么要遍历两轮？】两个循环的目的完全不同！
    # ════════════════════════════════════════════════════════════════════
    #
    # 【第1轮】构建 vocab 字典
    # 目的: 建立 "token文本 → ID" 的映射表
    # 结果: vocab = {"▁the": 0, "▁a": 1, "ing": 2, ...}
    #
    # 【第2轮】提取 merge 规则
    # 目的: 找出每个多字符 token 是由哪两个部分合并来的
    # 结果: merges = [("i", "ng"), ("t", "h"), ...]
    #
    # 【为什么不能合并成一轮？】
    # 第2轮需要查询 `if left in vocab`，所以必须先完成第1轮建好 vocab！
    #
    # ────────────────────────────────────────────────────────────────────
    # 【HuggingFace Tokenizer 需要两样东西】
    # ────────────────────────────────────────────────────────────────────
    #
    # 1. vocab: 词表，知道有哪些 token 及其 ID
    # 2. merges: 合并规则，知道分词时如何将字符组合成 token
    #
    # 例如分词 "thing":
    #   - 查 merges: 发现 ("t","h") 是一条规则 → "th"
    #   - 查 merges: 发现 ("i","ng") 是一条规则 → "ing"
    #   - 查 merges: 发现 ("th","ing") 是一条规则 → "thing"
    #   - 查 vocab: "thing" 的 ID 是 1234
    #   - 结果: [1234]
    #
    # ════════════════════════════════════════════════════════════════════
    
    # ==================== 第1轮: 构建词表 ====================
    # 提取词表 (token -> ID 映射)
    vocab = {}
    for i in range(sp_model.get_piece_size()):
        token = sp_model.id_to_piece(i)
        vocab[token] = i
    # 现在 vocab 已完整，可以用于第2轮的查询
    
    # ==================== 第2轮: 提取合并规则 ====================
    # 提取 BPE merges (合并规则)
    # merges 记录了 BPE 训练过程中的合并顺序
    merges = []
    for i in range(sp_model.get_piece_size()):
        piece = sp_model.id_to_piece(i)
        # 跳过特殊 token 和单字符 (单字符不是合并产生的)
        if piece.startswith('<') or piece.startswith('▁') or len(piece) <= 1:
            continue
        # 对于多字符 piece，找到合并点
        # 例如 "ing" 可能是 "i"+"ng" 或 "in"+"g"
        # 这是一个简化的实现，实际 merge 顺序由 SentencePiece 内部决定
        # 这里我们按字符顺序拆分，找到第一个有效的拆分点
        for j in range(1, len(piece)):
            left = piece[:j]    # 例如 "i"
            right = piece[j:]   # 例如 "ng"
            # 只有当 left 和 right 都在词表中时，这才是一个有效的合并
            if left in vocab and right in vocab:  # ← 这里需要查询第1轮建好的 vocab！
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
