"""
简化版 UE 专业助手训练脚本
只用纯 UE 数据训练，确保模型保存
"""
import os
import sys
import time
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from contextlib import nullcontext

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)  # 确保工作目录正确
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr
from transformers import AutoTokenizer

def train():
    # ========== 配置 ==========
    device = "cuda:0"
    epochs = 10          # 增加到10轮
    batch_size = 16      # 减小batch适应更大模型
    learning_rate = 5e-5 # 大模型用更小学习率
    max_seq_len = 512
    hidden_size = 768    # 增大模型
    num_layers = 16      # 增加层数
    log_interval = 50
    save_path = "out/ue_sft_pure_768.pth"
    
    # ========== 模型配置 ==========
    lm_config = MiniMindConfig(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        use_moe=False
    )
    
    # ========== 加载模型 ==========
    print("加载预训练模型...")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(ROOT_DIR, "model"))
    model = MiniMindForCausalLM(lm_config).to(device)
    
    # 加载预训练权重 (如果有的话)
    weight_path = os.path.join(ROOT_DIR, f"out/ue_pretrain_{hidden_size}.pth")
    if os.path.exists(weight_path):
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"加载预训练权重: {weight_path}")
    else:
        # 768模型没有预训练，直接从头训练SFT
        print(f"未找到预训练权重，直接进行SFT训练（768模型）")
    
    print(f"Model Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # ========== 数据集 ==========
    print("加载数据集...")
    train_ds = SFTDataset(
        jsonl_path=os.path.join(ROOT_DIR, "dataset/ue_sft.jsonl"),
        tokenizer=tokenizer,
        max_length=max_seq_len
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    print(f"数据集大小: {len(train_ds)} 条")
    print(f"每 epoch 步数: {len(train_loader)} 步")
    
    # ========== 优化器和混合精度 ==========
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scaler = GradScaler('cuda')
    autocast_ctx = autocast('cuda', dtype=torch.bfloat16)
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    
    total_steps = epochs * len(train_loader)
    global_step = 0
    
    # ========== 训练循环 ==========
    print(f"\n开始训练 {epochs} epochs...")
    model.train()
    
    for epoch in range(epochs):
        start_time = time.time()
        
        for step, (X, Y, loss_mask) in enumerate(train_loader):
            X = X.to(device)
            Y = Y.to(device)
            loss_mask = loss_mask.to(device)
            
            # 学习率调度
            lr = get_lr(global_step, total_steps, learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            # 前向传播
            with autocast_ctx:
                res = model(X)
                loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())
                loss = (loss * loss_mask).sum() / loss_mask.sum()
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            global_step += 1
            
            # 日志
            if (step + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (step + 1) * len(train_loader) - elapsed
                print(f"Epoch {epoch+1}/{epochs} Step {step+1}/{len(train_loader)} "
                      f"Loss: {loss.item():.4f} LR: {lr:.6f} ETA: {eta/60:.1f}min", flush=True)
        
        print(f"Epoch {epoch+1} 完成, 耗时: {(time.time()-start_time)/60:.1f}min", flush=True)
    
    # ========== 保存模型 ==========
    print(f"\n保存模型到 {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    state_dict = {k: v.half().cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, save_path)
    print(f"✅ 模型已保存! 大小: {os.path.getsize(save_path) / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    train()
