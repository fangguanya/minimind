"""
UnrealEngine RAG增强服务
========================

由于小模型难以精确记忆大量代码细节，本模块提供RAG（检索增强生成）功能：
1. 构建UE代码向量索引
2. 根据用户问题检索相关代码
3. 将检索结果注入到Prompt中，增强LLM回答能力

依赖安装：
pip install sentence-transformers faiss-cpu

使用方法：
1. 构建索引: python ue_rag_server.py --build_index --ue_source_path "D:/UnrealEngine/Engine/Source"
2. 启动服务: python ue_rag_server.py --serve
"""

import os
import sys
import json
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))


@dataclass
class CodeChunk:
    """代码块"""
    file_path: str
    content: str
    chunk_type: str  # 'class', 'function', 'header', 'general'
    class_name: Optional[str] = None
    function_name: Optional[str] = None


class UECodeIndexer:
    """UE代码索引器"""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self.embedder = None
        self.index = None
        self.chunks: List[CodeChunk] = []
        
    def _init_embedder(self):
        """延迟初始化embedding模型"""
        if self.embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"加载Embedding模型: {self.embedding_model_name}")
                self.embedder = SentenceTransformer(self.embedding_model_name)
            except ImportError:
                print("请先安装: pip install sentence-transformers")
                raise
    
    def extract_code_chunks(self, ue_source_path: str, max_files: int = None) -> List[CodeChunk]:
        """从UE源码提取代码块"""
        chunks = []
        source_path = Path(ue_source_path)
        
        skip_dirs = {'ThirdParty', 'Intermediate', 'Binaries', '.git'}
        
        header_files = []
        for root, dirs, files in os.walk(source_path):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for f in files:
                if f.endswith('.h'):
                    header_files.append(Path(root) / f)
        
        if max_files:
            header_files = header_files[:max_files]
        
        print(f"处理 {len(header_files)} 个头文件...")
        
        from tqdm import tqdm
        for file_path in tqdm(header_files, desc="提取代码块"):
            try:
                chunks.extend(self._extract_from_file(file_path, source_path))
            except Exception as e:
                continue
        
        print(f"共提取 {len(chunks)} 个代码块")
        return chunks
    
    def _extract_from_file(self, file_path: Path, base_path: Path) -> List[CodeChunk]:
        """从单个文件提取代码块"""
        chunks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            return chunks
        
        try:
            rel_path = str(file_path.relative_to(base_path))
        except ValueError:
            rel_path = file_path.name
        
        # 提取UCLASS定义
        uclass_pattern = r'(UCLASS\([^)]*\)\s*class\s+(?:\w+_API\s+)?(\w+)[^{]*\{[^}]*(?:\{[^}]*\}[^}]*)*\})'
        for match in re.finditer(uclass_pattern, content, re.DOTALL):
            class_code = match.group(0)[:2000]  # 限制长度
            class_name = match.group(2)
            
            chunks.append(CodeChunk(
                file_path=rel_path,
                content=class_code,
                chunk_type='class',
                class_name=class_name
            ))
        
        # 提取UFUNCTION定义
        ufunc_pattern = r'(UFUNCTION\([^)]*\)[^;{]+(?:\{[^}]*\}|;))'
        for match in re.finditer(ufunc_pattern, content):
            func_code = match.group(0)
            # 提取函数名
            name_match = re.search(r'(\w+)\s*\(', func_code)
            func_name = name_match.group(1) if name_match else None
            
            chunks.append(CodeChunk(
                file_path=rel_path,
                content=func_code,
                chunk_type='function',
                function_name=func_name
            ))
        
        # 如果文件较小，整体作为一个chunk
        if len(content) < 3000 and not chunks:
            chunks.append(CodeChunk(
                file_path=rel_path,
                content=content[:2000],
                chunk_type='header'
            ))
        
        return chunks
    
    def build_index(self, chunks: List[CodeChunk]):
        """构建向量索引"""
        self._init_embedder()
        self.chunks = chunks
        
        try:
            import faiss
            import numpy as np
        except ImportError:
            print("请先安装: pip install faiss-cpu")
            raise
        
        # 生成文本用于embedding
        texts = []
        for chunk in chunks:
            # 组合文件路径、类名/函数名和代码内容
            text = f"File: {chunk.file_path}\n"
            if chunk.class_name:
                text += f"Class: {chunk.class_name}\n"
            if chunk.function_name:
                text += f"Function: {chunk.function_name}\n"
            text += chunk.content[:500]  # 限制embedding的文本长度
            texts.append(text)
        
        print("生成Embeddings...")
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # 构建FAISS索引
        print("构建FAISS索引...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
        
        # 归一化（用于余弦相似度）
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        print(f"索引构建完成！共 {self.index.ntotal} 个向量")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[CodeChunk, float]]:
        """搜索相关代码"""
        self._init_embedder()
        
        if self.index is None:
            raise ValueError("索引未构建，请先调用build_index()")
        
        import numpy as np
        import faiss
        
        # 编码查询
        query_embedding = self.embedder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # 搜索
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def save(self, path: str):
        """保存索引"""
        import faiss
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # 保存FAISS索引
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # 保存chunks
        with open(path / "chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        print(f"索引已保存至: {path}")
    
    def load(self, path: str):
        """加载索引"""
        import faiss
        
        path = Path(path)
        
        # 加载FAISS索引
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        # 加载chunks
        with open(path / "chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)
        
        print(f"索引已加载！共 {len(self.chunks)} 个代码块")


class UERAGChatBot:
    """RAG增强的UE问答机器人"""
    
    def __init__(self, indexer: UECodeIndexer, model_path: str = None, 
                 hidden_size: int = 512, num_hidden_layers: int = 8):
        self.indexer = indexer
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.device = 'cuda' if self._check_cuda() else 'cpu'
        
    def _check_cuda(self):
        """检查CUDA是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
        
    def _init_model(self):
        """初始化LLM模型"""
        if self.model is not None:
            return True
        
        import torch
        from transformers import AutoTokenizer
        
        # 添加项目路径
        model_dir = PROJECT_ROOT / "model"
        sys.path.insert(0, str(PROJECT_ROOT))
        
        try:
            from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
        except ImportError:
            print("[错误] 无法导入MiniMind模型，请确保在minimind项目目录下运行")
            return False
        
        # 加载tokenizer
        tokenizer_path = PROJECT_ROOT / "model"
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        
        # 查找模型权重
        weight_path = None
        if self.model_path and os.path.exists(self.model_path):
            weight_path = self.model_path
        else:
            # 自动查找ue_sft权重
            default_weight = PROJECT_ROOT / "out" / f"ue_sft_{self.hidden_size}.pth"
            if default_weight.exists():
                weight_path = str(default_weight)
            else:
                # 尝试full_sft权重
                fallback_weight = PROJECT_ROOT / "out" / f"full_sft_{self.hidden_size}.pth"
                if fallback_weight.exists():
                    weight_path = str(fallback_weight)
        
        if not weight_path:
            print("[警告] 未找到模型权重，仅使用RAG检索功能")
            print(f"  尝试查找: out/ue_sft_{self.hidden_size}.pth")
            return False
        
        print(f"加载MiniMind模型: {weight_path}")
        
        # 创建模型
        config = MiniMindConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers
        )
        self.model = MiniMindForCausalLM(config)
        
        # 加载权重
        state_dict = torch.load(weight_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.eval().to(self.device)
        
        print(f"模型加载完成！设备: {self.device}")
        return True
    
    def build_prompt_with_context(self, query: str, top_k: int = 3) -> Tuple[str, List[CodeChunk]]:
        """构建带上下文的Prompt"""
        # 检索相关代码
        results = self.indexer.search(query, top_k=top_k)
        
        # 构建上下文
        context_parts = []
        retrieved_chunks = []
        
        for chunk, score in results:
            context = f"【相关代码 - {chunk.file_path}】\n"
            if chunk.class_name:
                context += f"类名: {chunk.class_name}\n"
            if chunk.function_name:
                context += f"函数: {chunk.function_name}\n"
            context += f"```cpp\n{chunk.content[:800]}\n```\n"
            
            context_parts.append(context)
            retrieved_chunks.append(chunk)
        
        context_str = "\n".join(context_parts)
        
        prompt = f"""你是一个UnrealEngine代码助手。请根据以下参考代码回答用户的问题。

【参考代码】
{context_str}

【用户问题】
{query}

【回答要求】
1. 如果参考代码能回答问题，请引用相关的类名、函数名和文件路径
2. 提供清晰的解释和代码示例
3. 如果参考代码不足以回答，请说明并给出一般性建议

【回答】"""
        
        return prompt, retrieved_chunks
    
    def chat(self, query: str, top_k: int = 3) -> Dict:
        """问答接口"""
        prompt, chunks = self.build_prompt_with_context(query, top_k)
        
        response = {
            "query": query,
            "retrieved_chunks": [
                {
                    "file_path": c.file_path,
                    "class_name": c.class_name,
                    "function_name": c.function_name,
                    "content_preview": c.content[:200] + "..."
                }
                for c in chunks
            ],
            "prompt": prompt,
            "answer": None
        }
        
        # 尝试初始化模型
        model_loaded = self._init_model()
        
        # 如果有模型，生成回答
        if model_loaded and self.model is not None:
            import torch
            
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            answer = self.tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
            response["answer"] = answer
        else:
            # 没有模型时，基于检索结果生成简单回答
            answer_parts = ["根据检索到的UE代码信息：\n"]
            for i, chunk in enumerate(chunks, 1):
                answer_parts.append(f"\n**{i}. {chunk.file_path}**")
                if chunk.class_name:
                    answer_parts.append(f"\n   类名: `{chunk.class_name}`")
                if chunk.function_name:
                    answer_parts.append(f"\n   函数: `{chunk.function_name}`")
                answer_parts.append(f"\n   代码预览:\n```cpp\n{chunk.content[:300]}...\n```")
            
            response["answer"] = "".join(answer_parts)
        
        return response


def build_index(args):
    """构建索引"""
    indexer = UECodeIndexer()
    
    # 提取代码块
    chunks = indexer.extract_code_chunks(
        args.ue_source_path,
        max_files=args.max_files
    )
    
    # 构建索引
    indexer.build_index(chunks)
    
    # 保存
    indexer.save(args.index_path)


def serve_rag(args):
    """启动RAG服务"""
    indexer = UECodeIndexer()
    indexer.load(args.index_path)
    
    chatbot = UERAGChatBot(
        indexer, 
        model_path=args.model_path,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers
    )
    
    print("\n" + "="*60)
    print("🚀 UE RAG问答服务已启动")
    print("="*60)
    print(f"模型配置: hidden_size={args.hidden_size}, layers={args.num_hidden_layers}")
    print("输入问题进行查询，输入 'quit' 退出")
    print("="*60 + "\n")
    
    while True:
        try:
            query = input("💬 问题: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            if not query:
                continue
            
            print("\n🔍 检索相关代码...")
            result = chatbot.chat(query, top_k=args.top_k)
            
            print("\n📚 检索到的相关代码:")
            print("-" * 40)
            for i, chunk in enumerate(result["retrieved_chunks"], 1):
                print(f"  [{i}] {chunk['file_path']}")
                if chunk['class_name']:
                    print(f"      类名: {chunk['class_name']}")
                if chunk['function_name']:
                    print(f"      函数: {chunk['function_name']}")
            print("-" * 40)
            
            print(f"\n🤖 回答:")
            print(result['answer'])
            print("\n" + "="*60 + "\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[错误] {e}")
            continue
    
    print("\n👋 再见！")


def main():
    parser = argparse.ArgumentParser(description="UE代码RAG服务")
    
    subparsers = parser.add_subparsers(dest='command')
    
    # 构建索引
    build_parser = subparsers.add_parser('build', help='构建代码索引')
    build_parser.add_argument('--ue_source_path', type=str, required=True,
                              help="UE源代码路径")
    build_parser.add_argument('--index_path', type=str, default='./ue_index',
                              help="索引保存路径")
    build_parser.add_argument('--max_files', type=int, default=None,
                              help="最大处理文件数")
    
    # 启动服务
    serve_parser = subparsers.add_parser('serve', help='启动RAG服务')
    serve_parser.add_argument('--index_path', type=str, default='./ue_index',
                              help="索引路径")
    serve_parser.add_argument('--model_path', type=str, default=None,
                              help="LLM模型权重路径（.pth文件，可选）")
    serve_parser.add_argument('--hidden_size', type=int, default=512,
                              help="模型隐藏层维度 (512=Small, 768=Base)")
    serve_parser.add_argument('--num_hidden_layers', type=int, default=8,
                              help="模型层数 (8=Small, 16=Base)")
    serve_parser.add_argument('--top_k', type=int, default=5,
                              help="检索结果数量")
    
    args = parser.parse_args()
    
    if args.command == 'build':
        build_index(args)
    elif args.command == 'serve':
        serve_rag(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
