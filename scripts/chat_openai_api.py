"""
================================================================================
                    MiniMind OpenAI API 客户端
================================================================================

【什么是这个脚本】
这是一个调用 MiniMind API 服务的客户端示例:
- 演示如何使用 OpenAI SDK 调用 MiniMind
- 支持流式和非流式调用
- 可作为集成 MiniMind 的参考代码

【使用场景】
1. 测试 serve_openai_api.py 服务是否正常
2. 作为调用模板，集成到其他应用
3. 学习如何使用 OpenAI SDK

【使用方法】
1. 首先启动服务:
   python serve_openai_api.py --port 8000

2. 然后运行客户端:
   python chat_openai_api.py --stream

【OpenAI SDK 简介】
openai 是 OpenAI 官方提供的 Python SDK:
- 支持所有 OpenAI API
- 可以通过 base_url 切换到兼容服务
- 支持同步和异步调用
"""

import os
import sys
import argparse

# 尝试导入 OpenAI SDK
try:
    from openai import OpenAI
except ImportError:
    print("请先安装 openai: pip install openai")
    sys.exit(1)


def chat_non_stream(client, messages: list, model: str = "minimind"):
    """
    非流式对话调用
    
    【流程】
    1. 调用 chat.completions.create
    2. 等待完整响应返回
    3. 打印结果
    
    【特点】
    - 简单直接
    - 需要等待完整生成
    - 适合短回复或后台处理
    
    【参数】
    - client: OpenAI 客户端实例
    - messages: 对话历史 [{"role": "user", "content": "..."}]
    - model: 模型名称
    
    【返回】
    - response.choices[0].message.content: 生成的回复
    """
    print("=" * 50)
    print("非流式调用")
    print("=" * 50)
    
    # 调用 API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,  # 非流式
        max_tokens=512,
        temperature=0.7
    )
    
    # 提取回复
    content = response.choices[0].message.content
    print(f"助手: {content}")
    
    return content


def chat_stream(client, messages: list, model: str = "minimind"):
    """
    流式对话调用
    
    【流程】
    1. 调用 chat.completions.create (stream=True)
    2. 返回一个迭代器
    3. 逐 chunk 处理并打印
    
    【特点】
    - 实时显示生成过程
    - 用户体验好 (像打字)
    - 适合交互式应用
    
    【SSE 响应格式】
    每个 chunk 包含:
    - choices[0].delta.content: 新生成的内容片段
    - 或 choices[0].finish_reason: 结束原因
    
    【参数】
    - client: OpenAI 客户端实例
    - messages: 对话历史
    - model: 模型名称
    
    【返回】
    - 完整的生成文本
    """
    print("=" * 50)
    print("流式调用")
    print("=" * 50)
    
    print("助手: ", end="", flush=True)
    
    # 调用 API (流式)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,  # 流式
        max_tokens=512,
        temperature=0.7
    )
    
    # 收集完整回复
    full_response = ""
    
    # 遍历 stream
    for chunk in response:
        # 检查是否有新内容
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)  # 实时打印
            full_response += content
        
        # 检查是否结束
        if chunk.choices[0].finish_reason:
            break
    
    print()  # 换行
    return full_response


def interactive_chat(client, model: str = "minimind", stream: bool = True):
    """
    交互式对话
    
    【流程】
    1. 循环读取用户输入
    2. 调用 API 生成回复
    3. 保存对话历史
    4. 特殊命令: 'quit' 退出, 'clear' 清除历史
    
    【参数】
    - client: OpenAI 客户端实例
    - model: 模型名称
    - stream: 是否使用流式输出
    """
    print("=" * 50)
    print("MiniMind 交互式对话")
    print("输入 'quit' 退出, 'clear' 清除历史")
    print("=" * 50)
    
    # 对话历史
    messages = []
    
    while True:
        # 读取用户输入
        try:
            user_input = input("\n你: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见!")
            break
        
        # 处理特殊命令
        if user_input.lower() == 'quit':
            print("再见!")
            break
        elif user_input.lower() == 'clear':
            messages = []
            print("对话历史已清除")
            continue
        elif not user_input:
            continue
        
        # 添加用户消息
        messages.append({"role": "user", "content": user_input})
        
        # 调用 API
        if stream:
            response = chat_stream(client, messages, model)
        else:
            response = chat_non_stream(client, messages, model)
        
        # 添加助手回复到历史
        messages.append({"role": "assistant", "content": response})


def main():
    """
    主函数
    
    【流程】
    1. 解析命令行参数
    2. 创建 OpenAI 客户端
    3. 根据模式运行对话
    """
    # ==================== 参数解析 ====================
    parser = argparse.ArgumentParser(description="MiniMind OpenAI API 客户端")
    
    # 服务器配置
    parser.add_argument(
        "--api_url", 
        type=str, 
        default="http://localhost:8000/v1",
        help="API 服务器地址 (默认: http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--api_key", 
        type=str, 
        default="none",
        help="API 密钥 (MiniMind 不需要，填任意值)"
    )
    
    # 模型配置
    parser.add_argument(
        "--model", 
        type=str, 
        default="minimind",
        help="模型名称"
    )
    
    # 调用模式
    parser.add_argument(
        "--stream", 
        action="store_true",
        help="使用流式输出"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="交互式对话模式"
    )
    
    # 单次问题
    parser.add_argument(
        "--question", 
        type=str, 
        default=None,
        help="单次问题 (不进入交互模式)"
    )
    
    args = parser.parse_args()
    
    # ==================== 创建客户端 ====================
    # 通过修改 base_url，OpenAI SDK 可以连接到任何兼容服务
    client = OpenAI(
        base_url=args.api_url,
        api_key=args.api_key  # MiniMind 不验证 API key
    )
    
    print(f"连接到: {args.api_url}")
    
    # ==================== 运行对话 ====================
    if args.interactive:
        # 交互式模式
        interactive_chat(client, args.model, args.stream)
    elif args.question:
        # 单次问答模式
        messages = [{"role": "user", "content": args.question}]
        if args.stream:
            chat_stream(client, messages, args.model)
        else:
            chat_non_stream(client, messages, args.model)
    else:
        # 默认演示
        print("\n演示调用...")
        messages = [
            {"role": "user", "content": "你好，请介绍一下你自己"}
        ]
        
        if args.stream:
            chat_stream(client, messages, args.model)
        else:
            chat_non_stream(client, messages, args.model)
        
        print("\n" + "=" * 50)
        print("提示: 使用 --interactive 进入交互模式")
        print("      使用 --question '问题' 进行单次问答")


if __name__ == "__main__":
    main()
