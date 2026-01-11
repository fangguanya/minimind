"""
================================================================================
                    MiniMind 模型格式转换脚本
================================================================================

【什么是这个脚本】
将 MiniMind 的 PyTorch 权重转换为其他格式:
1. HuggingFace Transformers 格式 (用于上传 Hub)
2. ONNX 格式 (用于推理优化)
3. (可扩展) TensorRT, CoreML 等

【为什么需要格式转换】
1. HuggingFace Hub 发布: 需要标准 Transformers 格式
2. 推理优化: ONNX 可以用 ONNX Runtime 加速
3. 跨平台部署: 不同框架需要不同格式

【HuggingFace 格式】
标准 Transformers 模型目录结构:
├── config.json           # 模型配置
├── pytorch_model.bin     # 权重 (或 model.safetensors)
├── tokenizer.json        # 分词器
├── tokenizer_config.json # 分词器配置
└── special_tokens_map.json

【ONNX 格式】
Open Neural Network Exchange 格式:
- 跨框架兼容 (PyTorch, TensorFlow, etc.)
- 支持 ONNX Runtime 推理优化
- 可进一步转换为 TensorRT

【使用方法】
# 转换为 HuggingFace 格式
python convert_model.py --format huggingface --model_weight full_sft --output_dir ./minimind-hf

# 转换为 ONNX 格式
python convert_model.py --format onnx --model_weight full_sft --output_dir ./minimind-onnx
"""

import os
import sys
import json
import shutil
import argparse

# 将父目录添加到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoTokenizer

from model.model_minimind import MiniMindForCausalLM, MiniMindConfig
from model.model_lora import apply_lora, load_lora


def convert_to_huggingface(model, tokenizer, lm_config, output_dir: str):
    """
    转换为 HuggingFace Transformers 格式
    
    【转换步骤】
    1. 创建输出目录
    2. 保存模型权重
    3. 保存模型配置
    4. 复制分词器文件
    5. 创建 README
    
    【注意事项】
    - 权重使用 float16 节省空间
    - 配置需要包含模型架构信息
    - 分词器需要完整复制
    
    【参数】
    - model: MiniMind 模型实例
    - tokenizer: 分词器
    - lm_config: 模型配置
    - output_dir: 输出目录
    """
    print(f"转换为 HuggingFace 格式: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== 1. 保存模型权重 ====================
    # 获取 state_dict
    state_dict = model.state_dict()
    
    # 转换为 float16 节省空间
    state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
    
    # 保存为 pytorch_model.bin
    model_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(state_dict, model_path)
    print(f"  ✓ 权重保存到: {model_path}")
    
    # ==================== 2. 保存模型配置 ====================
    # 创建 HuggingFace 风格的配置
    config_dict = {
        "architectures": ["MiniMindForCausalLM"],
        "model_type": "minimind",
        "vocab_size": lm_config.vocab_size,
        "hidden_size": lm_config.hidden_size,
        "num_hidden_layers": lm_config.num_hidden_layers,
        "num_attention_heads": lm_config.num_attention_heads,
        "num_key_value_heads": lm_config.num_key_value_heads,
        "intermediate_size": lm_config.intermediate_size,
        "max_position_embeddings": lm_config.max_seq_len,
        "rms_norm_eps": lm_config.rms_norm_eps,
        "rope_theta": lm_config.rope_theta,
        "use_moe": lm_config.use_moe,
        "torch_dtype": "float16",
        "transformers_version": "4.35.0"
    }
    
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 配置保存到: {config_path}")
    
    # ==================== 3. 复制分词器文件 ====================
    tokenizer_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
    
    # 复制 tokenizer.json
    src_tokenizer = os.path.join(tokenizer_dir, 'tokenizer.json')
    if os.path.exists(src_tokenizer):
        shutil.copy(src_tokenizer, os.path.join(output_dir, 'tokenizer.json'))
        print(f"  ✓ 复制 tokenizer.json")
    
    # 复制 tokenizer_config.json
    src_config = os.path.join(tokenizer_dir, 'tokenizer_config.json')
    if os.path.exists(src_config):
        shutil.copy(src_config, os.path.join(output_dir, 'tokenizer_config.json'))
        print(f"  ✓ 复制 tokenizer_config.json")
    
    # ==================== 4. 创建 README ====================
    readme_content = f"""---
language:
- zh
license: apache-2.0
tags:
- llm
- chinese
- minimind
---

# MiniMind

轻量级中文语言模型

## 模型信息

- **参数量**: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M
- **隐藏层维度**: {lm_config.hidden_size}
- **层数**: {lm_config.num_hidden_layers}
- **注意力头数**: {lm_config.num_attention_heads}
- **词表大小**: {lm_config.vocab_size}

## 使用方法

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("path/to/model", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("path/to/model", trust_remote_code=True)

input_text = "你好"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""
    
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"  ✓ 创建 README.md")
    
    print(f"\n转换完成! 目录: {output_dir}")


def convert_to_onnx(model, tokenizer, lm_config, output_dir: str, opset_version: int = 14):
    """
    转换为 ONNX 格式
    
    【ONNX 简介】
    Open Neural Network Exchange (ONNX):
    - 开放的神经网络交换格式
    - 支持跨框架互操作
    - 可用 ONNX Runtime 推理
    
    【转换步骤】
    1. 准备示例输入
    2. 使用 torch.onnx.export 导出
    3. 验证 ONNX 模型
    
    【参数】
    - model: MiniMind 模型实例
    - tokenizer: 分词器
    - lm_config: 模型配置
    - output_dir: 输出目录
    - opset_version: ONNX opset 版本
    """
    print(f"转换为 ONNX 格式: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置为评估模式
    model.eval()
    
    # ==================== 1. 准备示例输入 ====================
    # ONNX 导出需要一个示例输入来追踪计算图
    batch_size = 1
    seq_length = 32
    
    # 创建示例输入
    dummy_input = torch.randint(
        0, lm_config.vocab_size, 
        (batch_size, seq_length), 
        dtype=torch.long
    )
    
    # 如果模型在 GPU 上，输入也需要在 GPU
    if next(model.parameters()).is_cuda:
        dummy_input = dummy_input.cuda()
    
    # ==================== 2. 导出 ONNX ====================
    onnx_path = os.path.join(output_dir, "model.onnx")
    
    # 定义动态轴 (允许不同的 batch size 和序列长度)
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size', 1: 'sequence_length'}
    }
    
    print("  正在导出 ONNX (这可能需要几分钟)...")
    
    # 导出
    torch.onnx.export(
        model,                      # 模型
        dummy_input,                # 示例输入
        onnx_path,                  # 输出路径
        export_params=True,         # 导出参数
        opset_version=opset_version,# ONNX opset 版本
        do_constant_folding=True,   # 常量折叠优化
        input_names=['input_ids'],  # 输入名称
        output_names=['logits'],    # 输出名称
        dynamic_axes=dynamic_axes   # 动态轴
    )
    
    print(f"  ✓ ONNX 模型保存到: {onnx_path}")
    
    # ==================== 3. 验证 ONNX 模型 ====================
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("  ✓ ONNX 模型验证通过")
    except ImportError:
        print("  ! 跳过验证 (需要安装 onnx: pip install onnx)")
    except Exception as e:
        print(f"  ! ONNX 验证警告: {e}")
    
    # ==================== 4. 保存配置和分词器 ====================
    # 保存配置
    config_dict = {
        "vocab_size": lm_config.vocab_size,
        "hidden_size": lm_config.hidden_size,
        "num_hidden_layers": lm_config.num_hidden_layers,
        "num_attention_heads": lm_config.num_attention_heads,
        "max_seq_len": lm_config.max_seq_len
    }
    
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2)
    print(f"  ✓ 配置保存到: {config_path}")
    
    # 复制分词器
    tokenizer_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
    src_tokenizer = os.path.join(tokenizer_dir, 'tokenizer.json')
    if os.path.exists(src_tokenizer):
        shutil.copy(src_tokenizer, os.path.join(output_dir, 'tokenizer.json'))
        print(f"  ✓ 复制 tokenizer.json")
    
    print(f"\n转换完成! 目录: {output_dir}")
    print("\n使用 ONNX Runtime 推理示例:")
    print("""
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx")
input_ids = np.array([[1, 2, 3, 4]], dtype=np.int64)
outputs = session.run(None, {"input_ids": input_ids})
logits = outputs[0]
""")


def merge_lora_weights(model, lora_path: str):
    """
    合并 LoRA 权重到基础模型
    
    【LoRA 合并原理】
    原始: output = W·x + B·A·x
    合并: W' = W + B·A
    合并后: output = W'·x
    
    【为什么要合并】
    1. 推理时不需要 LoRA 层，减少计算
    2. 导出时模型更简洁
    3. 部署更方便
    
    【参数】
    - model: 已应用 LoRA 的模型
    - lora_path: LoRA 权重路径
    """
    print(f"正在合并 LoRA 权重: {lora_path}")
    
    # 加载 LoRA 权重
    load_lora(model, lora_path)
    
    # 遍历所有模块，合并 LoRA
    for name, module in model.named_modules():
        if hasattr(module, 'lora') and module.lora is not None:
            # 获取 LoRA 输出
            # ΔW = B @ A
            lora_weight = module.lora.lora_B @ module.lora.lora_A
            
            # 合并到原始权重
            # W' = W + ΔW
            module.weight.data += lora_weight
            
            # 移除 LoRA 模块
            module.lora = None
    
    print("  ✓ LoRA 权重已合并")


def main():
    """主函数"""
    # ==================== 参数解析 ====================
    parser = argparse.ArgumentParser(description="MiniMind 模型格式转换")
    
    # 转换格式
    parser.add_argument(
        "--format", 
        type=str, 
        required=True,
        choices=["huggingface", "onnx"],
        help="目标格式: huggingface 或 onnx"
    )
    
    # 输出目录
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./converted_model",
        help="输出目录"
    )
    
    # 模型配置
    parser.add_argument("--device", type=str, default="cpu", help="加载设备")
    parser.add_argument("--hidden_size", type=int, default=512, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="隐藏层数量")
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1], help="是否使用MoE")
    
    # 权重配置
    parser.add_argument("--model_weight", type=str, default="full_sft", help="基础权重名称")
    parser.add_argument("--lora_weight", type=str, default=None, help="LoRA权重名称（可选，会合并到基础模型）")
    
    # ONNX 特定参数
    parser.add_argument("--opset_version", type=int, default=14, help="ONNX opset版本")
    
    args = parser.parse_args()
    
    # ==================== 加载模型 ====================
    print("正在加载模型...")
    
    # 创建配置
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained('../model', trust_remote_code=True)
    
    # 初始化模型
    model = MiniMindForCausalLM(lm_config)
    
    # 加载权重
    moe_suffix = '_moe' if lm_config.use_moe else ''
    weight_path = f'../out/{args.model_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(weight_path, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    print(f"  ✓ 加载权重: {weight_path}")
    
    # 如果有 LoRA，应用并合并
    if args.lora_weight:
        apply_lora(model)
        lora_path = f'../out/lora/{args.lora_weight}_{lm_config.hidden_size}.pth'
        merge_lora_weights(model, lora_path)
    
    # 移到设备
    model = model.to(args.device)
    model.eval()
    
    print(f"  ✓ 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # ==================== 执行转换 ====================
    if args.format == "huggingface":
        convert_to_huggingface(model, tokenizer, lm_config, args.output_dir)
    elif args.format == "onnx":
        convert_to_onnx(model, tokenizer, lm_config, args.output_dir, args.opset_version)


if __name__ == "__main__":
    main()
