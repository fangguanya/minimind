@echo off
chcp 65001 >nul
REM ====================================================
REM MiniMind-UE RAG增强问答 (推荐)
REM ====================================================

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║       MiniMind-UE RAG增强问答服务 (推荐)                 ║
echo ╠══════════════════════════════════════════════════════════╣
echo ║  模式: 向量检索 + 大语言模型                             ║
echo ║  优点: 精确的文件路径、类名、函数名、代码片段            ║
echo ║  原理: 先检索相关代码，再由LLM组织回答                   ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

cd /d %~dp0

if not exist "ue_index\index.faiss" (
    echo [错误] RAG索引不存在: ue_index\
    echo 请先运行 train_ue_full.bat 进行训练
    pause
    exit /b 1
)

echo 提示: 输入问题后按回车，输入 quit 退出
echo.
echo 示例问题:
echo   - 什么是AActor类？
echo   - BeginPlay在哪个文件定义？
echo   - 如何实现Tick功能？
echo   - UCharacterMovementComponent有哪些函数？
echo.

cd scripts
python ue_rag_server.py serve --index_path ../ue_index --top_k 5

pause
