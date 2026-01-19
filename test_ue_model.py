"""
测试 UE 专业助手模型
"""
import os
import sys
import torch

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from transformers import AutoTokenizer

def test_model(model_name="ue_sft_pure"):
    device = "cuda:0"
    hidden_size = 512
    num_layers = 8
    
    # 加载模型
    print(f"加载模型: {model_name}...")
    lm_config = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_layers)
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT_DIR, "model"))
    model = MiniMindForCausalLM(lm_config).to(device)
    
    weight_path = os.path.join(ROOT_DIR, f"out/{model_name}_{hidden_size}.pth")
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"Model Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 测试问题
    test_questions = [
        "什么是UObject类？",
        "UMovieSceneWidgetMaterialTrack关联哪些类型？",
        "AActor有哪些主要函数？",
        "什么是UGameplayTask类？",
        "FVector是什么？",
    ]
    
    print("\n" + "="*60)
    print("开始测试")
    print("="*60)
    
    for question in test_questions:
        # 构建对话格式
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        print(f"\n💬 {question}")
        print(f"🤖 {response[:500]}")
        print("-"*60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ue_sft_pure", help="模型名称")
    args = parser.parse_args()
    test_model(args.model)
