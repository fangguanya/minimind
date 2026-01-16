"""
================================================================================
                    MiniMind OpenAI 兼容 API 服务器
================================================================================

【什么是这个脚本】
这个脚本将 MiniMind 模型包装成 OpenAI 兼容的 REST API:
- 支持 /v1/chat/completions 端点
- 支持流式和非流式响应
- 可以用 OpenAI SDK 直接调用

【为什么需要这个】
1. OpenAI API 是行业标准
2. 方便集成到各种应用中
3. 支持流式输出，用户体验好

【API 规范】
遵循 OpenAI Chat Completions API:
- 请求体: {"model": "...", "messages": [...], "stream": bool, ...}
- 响应体: {"choices": [{"message": {"content": "..."}}], ...}
- 流式响应: Server-Sent Events (SSE) 格式

【使用方法】
启动服务器:
    python serve_openai_api.py --port 8000 --model_weight full_sft

调用示例 (Python):
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")
    response = client.chat.completions.create(
        model="minimind",
        messages=[{"role": "user", "content": "你好"}],
        stream=True
    )
    for chunk in response:
        print(chunk.choices[0].delta.content, end="")

【端点说明】
- GET /: 健康检查
- POST /v1/chat/completions: 对话补全 (主要端点)

【流式响应格式 (SSE)】
data: {"choices": [{"delta": {"content": "你"}}]}
data: {"choices": [{"delta": {"content": "好"}}]}
data: [DONE]
"""

import os
import sys

# 将父目录添加到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import json
import uuid
import argparse
import uvicorn
import torch
from transformers import AutoTokenizer
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union

from model.model_minimind import MiniMindForCausalLM, MiniMindConfig
from model.model_lora import apply_lora, load_lora


# ==================== FastAPI 应用 ====================
app = FastAPI(
    title="MiniMind OpenAI Compatible API",
    description="OpenAI 兼容的 MiniMind 推理 API",
    version="1.0"
)


# ==================== 请求/响应模型定义 ====================
class ChatMessage(BaseModel):
    """
    单条对话消息
    
    【字段说明】
    - role: 角色 (system/user/assistant)
    - content: 消息内容
    """
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """
    对话补全请求
    
    【字段说明】
    - model: 模型名称 (可选，这里只有一个模型)
    - messages: 对话历史
    - stream: 是否流式输出
    - max_tokens: 最大生成长度
    - temperature: 采样温度 (越高越随机)
    - top_p: nucleus 采样参数
    """
    model: Optional[str] = "minimind"
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


class ChatCompletionResponse(BaseModel):
    """
    对话补全响应 (非流式)
    
    【字段说明】
    - id: 请求 ID
    - object: 对象类型 ("chat.completion")
    - created: 创建时间戳
    - model: 模型名称
    - choices: 生成结果列表
    - usage: token 使用统计
    """
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "minimind"
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None


class ChatCompletionChunk(BaseModel):
    """
    流式响应的单个 chunk
    
    【与非流式的区别】
    - object: "chat.completion.chunk"
    - choices 中使用 "delta" 而非 "message"
    """
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str = "minimind"
    choices: List[Dict[str, Any]]


# ==================== 全局变量 ====================
model = None
tokenizer = None
args = None


def generate_response(messages: List[ChatMessage], max_tokens: int = 2048, 
                      temperature: float = 0.7, top_p: float = 0.9):
    """
    生成非流式响应
    
    【流程】
    1. 将对话历史格式化为 ChatML 格式
    2. 调用模型生成回复
    3. 解码并返回文本
    
    【参数】
    - messages: 对话历史
    - max_tokens: 最大生成 token 数
    - temperature: 采样温度
    - top_p: nucleus 采样参数
    
    【返回】
    - response_text: 生成的回复文本
    """
    # 格式化对话历史
    formatted_messages = [{"role": m.role, "content": m.content} for m in messages]
    
    # 使用 chat_template 将对话转换为模型输入
    # add_generation_prompt=True 会添加 <|im_start|>assistant\n 提示模型开始生成
    prompt = tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Token 化
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
    
    # 生成回复
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 提取生成的部分 (去掉 prompt)
    generated_ids = output[0][input_ids.shape[1]:]
    
    # 解码
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response_text


async def generate_stream(messages: List[ChatMessage], max_tokens: int = 2048,
                          temperature: float = 0.7, top_p: float = 0.9):
    """
    生成流式响应 (Generator)
    
    【流式生成原理】
    不是一次生成所有 token，而是:
    1. 每生成一个 token，立即 yield
    2. 客户端可以立即显示，不需要等待完整回复
    3. 用户体验更好 (像打字一样)
    
    【SSE 格式】
    每个 chunk 格式为:
    data: {"choices": [{"delta": {"content": "你"}}]}
    
    最后发送:
    data: [DONE]
    
    【参数】
    - messages: 对话历史
    - max_tokens: 最大生成 token 数
    - temperature: 采样温度
    - top_p: nucleus 采样参数
    
    【Yields】
    - SSE 格式的字符串
    """
    # 生成唯一 ID
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created_time = int(time.time())
    
    # 格式化对话历史
    formatted_messages = [{"role": m.role, "content": m.content} for m in messages]
    
    # 使用 chat_template 转换
    prompt = tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Token 化
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
    
    # 流式生成
    # 使用 generate 的 iterator 模式
    generated_ids = []
    past_key_values = None
    current_input_ids = input_ids
    
    for _ in range(max_tokens):
        with torch.no_grad():
            # 前向传播
            outputs = model(
                current_input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            # 获取下一个 token 的 logits
            next_token_logits = outputs.logits[:, -1, :]
            
            # 应用温度和 top_p 采样
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                
                # Top-p (nucleus) 采样
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float('-inf')
                
                # 采样
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # 贪婪解码
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 检查是否生成结束
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # 解码当前 token
            token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            
            # 构造 SSE chunk
            chunk = ChatCompletionChunk(
                id=request_id,
                created=created_time,
                choices=[{
                    "index": 0,
                    "delta": {"content": token_text},
                    "finish_reason": None
                }]
            )
            yield f"data: {json.dumps(chunk.dict())}\n\n"
            
            # 更新状态
            generated_ids.append(next_token.item())
            current_input_ids = next_token
            past_key_values = outputs.past_key_values
    
    # 发送结束标记
    final_chunk = ChatCompletionChunk(
        id=request_id,
        created=created_time,
        choices=[{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    )
    yield f"data: {json.dumps(final_chunk.dict())}\n\n"
    yield "data: [DONE]\n\n"


# ==================== API 端点 ====================

@app.get("/")
async def root():
    """
    健康检查端点
    
    用于检测服务是否正常运行
    """
    return {"status": "ok", "model": "minimind", "message": "MiniMind API is running"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    对话补全端点 (OpenAI 兼容)
    
    【请求示例】
    {
        "model": "minimind",
        "messages": [
            {"role": "system", "content": "你是一个助手"},
            {"role": "user", "content": "你好"}
        ],
        "stream": false,
        "max_tokens": 512,
        "temperature": 0.7
    }
    
    【非流式响应示例】
    {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "minimind",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "你好！有什么可以帮助你的？"},
            "finish_reason": "stop"
        }]
    }
    
    【流式响应】
    data: {"id":"...","choices":[{"delta":{"content":"你"}}]}
    data: {"id":"...","choices":[{"delta":{"content":"好"}}]}
    data: [DONE]
    """
    try:
        if request.stream:
            # 流式响应
            return StreamingResponse(
                generate_stream(
                    request.messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p
                ),
                media_type="text/event-stream"
            )
        else:
            # 非流式响应
            response_text = generate_response(
                request.messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            )
            
            return ChatCompletionResponse(
                choices=[{
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop"
                }]
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def init_model_from_args(args):
    """
    根据命令行参数初始化模型
    
    【加载流程】
    1. 创建模型配置
    2. 初始化空模型
    3. 加载预训练权重
    4. (可选) 应用 LoRA 权重
    5. 切换到评估模式
    """
    global model, tokenizer
    
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
    
    # 6. 移到设备并设为评估模式
    model = model.to(args.device)
    model.eval()
    
    print(f"模型加载完成: {weight_path}")
    if args.lora_weight:
        print(f"LoRA 权重: {lora_path}")


if __name__ == "__main__":
    # ==================== 参数解析 ====================
    parser = argparse.ArgumentParser(description="MiniMind OpenAI Compatible API Server")
    
    # 服务器配置
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    
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
    print(f"服务器启动: http://{args.host}:{args.port}")
    
    # 启动服务器
    uvicorn.run(app, host=args.host, port=args.port)
