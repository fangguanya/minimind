"""
================================================================================
                    MiniMind Web Demo (Gradio 界面)
================================================================================

【什么是这个脚本】
这是一个基于 Gradio 的 Web 演示界面:
- 提供直观的聊天界面
- 支持参数调节
- 实时流式输出

【Gradio 简介】
Gradio 是一个快速创建 ML 演示界面的库:
- 几行代码创建 Web UI
- 自动处理前后端通信
- 支持各种输入输出组件

【功能特点】
1. 聊天界面: 类似 ChatGPT 的对话框
2. 参数调节: 温度、top_p、max_tokens
3. 流式输出: 实时显示生成过程
4. 清除历史: 一键清除对话

【使用方法】
启动服务:
    python web_demo.py --port 7860 --model_weight full_sft

然后在浏览器访问: http://localhost:7860

【界面布局】
┌─────────────────────────────────────────┐
│                 MiniMind Chat           │
├─────────────────────────────────────────┤
│  ┌─────────────────────────────────┐    │
│  │ 用户: 你好                      │    │
│  │ 助手: 你好！有什么可以帮你的？  │    │
│  │ ...                              │    │
│  └─────────────────────────────────┘    │
├─────────────────────────────────────────┤
│  输入框: [________________] [发送]      │
├─────────────────────────────────────────┤
│  参数: Temperature [0.7] Max Tokens [512]│
└─────────────────────────────────────────┘
"""

import os
import sys

# 将父目录添加到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import gradio as gr
from transformers import AutoTokenizer

from model.model_minimind import MiniMindForCausalLM, MiniMindConfig
from model.model_lora import apply_lora, load_lora


# ==================== 全局变量 ====================
model = None
tokenizer = None
args = None


def generate_response(message: str, history: list, temperature: float = 0.7, 
                      top_p: float = 0.9, max_tokens: int = 512):
    """
    生成对话回复 (流式)
    
    【流程】
    1. 将对话历史和当前消息组装成 messages 列表
    2. 使用 chat_template 格式化
    3. 逐 token 生成并 yield
    
    【参数】
    - message: 用户当前输入
    - history: 对话历史 [(user_msg, bot_msg), ...]
    - temperature: 采样温度
    - top_p: nucleus 采样参数
    - max_tokens: 最大生成长度
    
    【Yields】
    - 逐步生成的回复文本
    """
    # ==================== 1. 组装对话历史 ====================
    messages = []
    
    # 添加系统提示 (可选)
    # messages.append({"role": "system", "content": "你是一个有用的助手。"})
    
    # 添加历史对话
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if bot_msg:  # bot_msg 可能为 None (正在生成时)
            messages.append({"role": "assistant", "content": bot_msg})
    
    # 添加当前用户消息
    messages.append({"role": "user", "content": message})
    
    # ==================== 2. 使用 chat_template 格式化 ====================
    # add_generation_prompt=True 会添加 <|im_start|>assistant\n
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # ==================== 3. Token 化 ====================
    input_ids = tokenizer(
        prompt, 
        return_tensors="pt", 
        add_special_tokens=False
    ).input_ids.to(args.device)
    
    # ==================== 4. 流式生成 ====================
    generated_text = ""
    past_key_values = None
    current_input_ids = input_ids
    
    for _ in range(max_tokens):
        with torch.no_grad():
            # 前向传播
            outputs = model(
                current_input_ids,
                past_key_values=past_key_values,
                use_cache=True  # 使用 KV 缓存加速
            )
            
            # 获取最后一个位置的 logits
            next_token_logits = outputs.logits[:, -1, :]
            
            # 采样策略
            if temperature > 0:
                # 应用温度
                next_token_logits = next_token_logits / temperature
                
                # Top-p (nucleus) 采样
                # 只保留累积概率达到 top_p 的 token
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 找到累积概率超过 top_p 的位置
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # 将这些位置的 logits 设为负无穷
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
                
                # 从修改后的分布中采样
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # 贪婪解码 (temperature=0)
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 检查是否是结束 token
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # 解码当前 token
            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            generated_text += token_text
            
            # Yield 当前生成的文本 (流式输出)
            yield generated_text
            
            # 更新状态
            current_input_ids = next_token
            past_key_values = outputs.past_key_values
    
    # 返回最终结果
    yield generated_text


def clear_history():
    """清除对话历史"""
    return [], ""


def init_model_from_args(args_input):
    """
    根据命令行参数初始化模型
    
    【加载流程】
    1. 创建模型配置
    2. 加载 tokenizer
    3. 初始化模型并加载权重
    4. (可选) 应用 LoRA
    5. 设置为评估模式
    """
    global model, tokenizer, args
    args = args_input
    
    # 1. 创建模型配置
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=bool(args.use_moe)
    )
    
    # 2. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained('../model', trust_remote_code=True)
    
    # 3. 初始化模型
    model = MiniMindForCausalLM(lm_config)
    
    # 4. 加载权重
    moe_suffix = '_moe' if lm_config.use_moe else ''
    weight_path = f'../out/{args.model_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(weight_path, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    
    # 5. (可选) 加载 LoRA
    if args.lora_weight:
        apply_lora(model)
        lora_path = f'../out/lora/{args.lora_weight}_{lm_config.hidden_size}.pth'
        load_lora(model, lora_path)
        print(f"LoRA 权重已加载: {lora_path}")
    
    # 6. 移到设备并设为评估模式
    model = model.to(args.device)
    model.eval()
    
    print(f"模型加载完成: {weight_path}")


def create_demo():
    """
    创建 Gradio 演示界面
    
    【界面组件】
    - Chatbot: 对话显示区域
    - Textbox: 用户输入框
    - Button: 发送和清除按钮
    - Slider: 参数调节滑块
    """
    
    # 创建界面
    with gr.Blocks(title="MiniMind Chat", theme=gr.themes.Soft()) as demo:
        # 标题
        gr.Markdown("""
        # 🧠 MiniMind Chat
        轻量级中文语言模型对话演示
        """)
        
        # 对话区域
        chatbot = gr.Chatbot(
            height=500,
            bubble_full_width=False,
            avatar_images=(None, "https://em-content.zobj.net/source/apple/354/robot_1f916.png")
        )
        
        # 输入区域
        with gr.Row():
            msg = gr.Textbox(
                placeholder="输入你的问题...",
                show_label=False,
                container=False,
                scale=8
            )
            submit_btn = gr.Button("发送", variant="primary", scale=1)
            clear_btn = gr.Button("清除", scale=1)
        
        # 参数调节区域
        with gr.Accordion("⚙️ 高级参数", open=False):
            with gr.Row():
                temperature = gr.Slider(
                    minimum=0.0, 
                    maximum=2.0, 
                    value=0.7, 
                    step=0.1, 
                    label="Temperature (温度)",
                    info="越高越随机，越低越确定"
                )
                top_p = gr.Slider(
                    minimum=0.0, 
                    maximum=1.0, 
                    value=0.9, 
                    step=0.1, 
                    label="Top-P",
                    info="Nucleus 采样参数"
                )
                max_tokens = gr.Slider(
                    minimum=64, 
                    maximum=2048, 
                    value=512, 
                    step=64, 
                    label="Max Tokens",
                    info="最大生成长度"
                )
        
        # 使用示例
        gr.Examples(
            examples=[
                "你好，请介绍一下你自己",
                "请解释一下什么是机器学习",
                "帮我写一首关于春天的诗",
                "1+1等于多少？请详细解释"
            ],
            inputs=msg
        )
        
        # ==================== 事件绑定 ====================
        
        def user_submit(message, history):
            """用户提交消息"""
            if not message.strip():
                return "", history
            # 添加用户消息到历史 (bot 回复先设为 None)
            history = history + [[message, None]]
            return "", history
        
        def bot_response(history, temperature, top_p, max_tokens):
            """生成机器人回复"""
            if not history:
                return history
            
            # 获取最后一条用户消息
            user_message = history[-1][0]
            
            # 流式生成回复
            for response in generate_response(
                user_message, 
                history[:-1],  # 历史不包括当前这条
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            ):
                history[-1][1] = response
                yield history
        
        # 提交按钮事件
        submit_btn.click(
            user_submit,
            [msg, chatbot],
            [msg, chatbot],
            queue=False
        ).then(
            bot_response,
            [chatbot, temperature, top_p, max_tokens],
            chatbot
        )
        
        # 回车提交
        msg.submit(
            user_submit,
            [msg, chatbot],
            [msg, chatbot],
            queue=False
        ).then(
            bot_response,
            [chatbot, temperature, top_p, max_tokens],
            chatbot
        )
        
        # 清除按钮事件
        clear_btn.click(
            lambda: ([], ""),
            None,
            [chatbot, msg]
        )
    
    return demo


if __name__ == "__main__":
    # ==================== 参数解析 ====================
    parser = argparse.ArgumentParser(description="MiniMind Web Demo")
    
    # 服务器配置
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=7860, help="服务器端口")
    parser.add_argument("--share", action="store_true", help="是否创建公共链接")
    
    # 模型配置
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="推理设备")
    parser.add_argument("--hidden_size", type=int, default=512, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", type=int, default=8, help="隐藏层数量")
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1], help="是否使用MoE")
    
    # 权重配置
    parser.add_argument("--model_weight", type=str, default="full_sft", help="基础权重名称")
    parser.add_argument("--lora_weight", type=str, default=None, help="LoRA权重名称（可选）")
    
    args = parser.parse_args()
    
    # 初始化模型
    print("正在加载模型...")
    init_model_from_args(args)
    
    # 创建并启动 demo
    print(f"启动 Web Demo: http://{args.host}:{args.port}")
    demo = create_demo()
    demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )
