@echo off
chcp 65001 >nul
REM ====================================================
REM MiniMind-UE 完整训练脚本 (包含RAG)
REM ====================================================
REM 包含: 数据准备 → Pretrain → SFT → RAG索引 → 问答服务
REM ====================================================

REM ========== 配置区域 (请根据你的环境修改) ==========
SET UE_SOURCE_PATH=E:\Matrix_UE\Unreal_Matrix\Engine\Source
SET HIDDEN_SIZE=512
SET NUM_LAYERS=8
SET PRETRAIN_EPOCHS=2
SET SFT_EPOCHS=3
SET BATCH_SIZE=16
SET MAX_SEQ_LEN=512
REM ===================================================

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║     MiniMind-UE: UnrealEngine 代码助手完整训练流程        ║
echo ╠══════════════════════════════════════════════════════════╣
echo ║  Step 1: 准备Pretrain数据 (UE代码 → 纯文本)              ║
echo ║  Step 2: 准备SFT数据 (UE代码 → 问答对)                   ║
echo ║  Step 3: 预训练 (学习UE代码知识)                         ║
echo ║  Step 4: SFT微调 (学习问答对话)                          ║
echo ║  Step 5: 构建RAG索引 (代码向量检索)                      ║
echo ║  Step 6: 启动问答服务 (LLM + RAG)                        ║
echo ╚══════════════════════════════════════════════════════════╝
echo.
echo 配置信息:
echo   UE引擎路径: %UE_SOURCE_PATH%
echo   模型维度: %HIDDEN_SIZE%, 层数: %NUM_LAYERS%
echo   Pretrain轮数: %PRETRAIN_EPOCHS%, SFT轮数: %SFT_EPOCHS%
echo.

REM 检查路径
if not exist "%UE_SOURCE_PATH%" (
    echo [错误] UE源码路径不存在: %UE_SOURCE_PATH%
    echo 请编辑此脚本，设置正确的 UE_SOURCE_PATH
    pause
    exit /b 1
)

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请确保Python已安装并添加到PATH
    pause
    exit /b 1
)

echo.
echo ══════════════════════════════════════════════════════════
echo [Step 1/6] 准备Pretrain数据...
echo ══════════════════════════════════════════════════════════
cd /d %~dp0
cd scripts

REM 先生成UE代码数据
if not exist "..\dataset\ue_pretrain.jsonl" (
    python prepare_ue_pretrain_data.py --ue_source_path "%UE_SOURCE_PATH%" --output_path "../dataset/ue_pretrain.jsonl" --chunk_size %MAX_SEQ_LEN% --max_file_size 200
    if errorlevel 1 (
        echo [错误] UE Pretrain数据生成失败
        pause
        exit /b 1
    )
)

REM 检查通用预训练数据
if not exist "..\dataset\pretrain_hq.jsonl" (
    echo [错误] 缺少通用预训练数据: dataset\pretrain_hq.jsonl
    echo 请从以下地址下载:
    echo   https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files
    pause
    exit /b 1
)

REM 合并UE代码 + 通用知识
if exist "..\dataset\ue_pretrain_merged.jsonl" (
    echo [跳过] 合并Pretrain数据已存在: dataset\ue_pretrain_merged.jsonl
) else (
    echo 合并UE代码 + 通用知识数据...
    python merge_ue_data.py --type pretrain --ue_ratio 0.3
    if errorlevel 1 (
        echo [错误] 预训练数据合并失败
        pause
        exit /b 1
    )
)

echo.
echo ══════════════════════════════════════════════════════════
echo [Step 2/6] 准备SFT数据...
echo ══════════════════════════════════════════════════════════

REM 先生成UE专业SFT数据
if not exist "..\dataset\ue_sft.jsonl" (
    python prepare_ue_sft_data.py --ue_source_path "%UE_SOURCE_PATH%" --output_path "../dataset/ue_sft.jsonl"
    if errorlevel 1 (
        echo [错误] UE SFT数据生成失败
        pause
        exit /b 1
    )
)

REM 检查通用SFT数据是否存在
if not exist "..\dataset\sft_mini_512.jsonl" (
    echo [错误] 缺少通用SFT数据: dataset\sft_mini_512.jsonl
    echo 请从以下地址下载:
    echo   https://www.modelscope.cn/datasets/gongjy/minimind_dataset/files
    pause
    exit /b 1
)

REM 合并UE数据和通用数据
if exist "..\dataset\ue_sft_merged.jsonl" (
    echo [跳过] 合并SFT数据已存在: dataset\ue_sft_merged.jsonl
) else (
    echo 合并UE专业数据 + 通用对话数据...
    python merge_ue_data.py
    if errorlevel 1 (
        echo [错误] 数据合并失败
        pause
        exit /b 1
    )
)

echo.
echo ══════════════════════════════════════════════════════════
echo [Step 3/6] 预训练 (Pretrain)...
echo ══════════════════════════════════════════════════════════
cd /d %~dp0
cd trainer

REM 检查是否已有预训练权重
if exist "..\out\ue_pretrain_%HIDDEN_SIZE%.pth" (
    echo [跳过] 预训练权重已存在: out\ue_pretrain_%HIDDEN_SIZE%.pth
    echo 如需重新训练，请删除该文件
) else (
    echo 开始预训练 - 使用合并数据: UE代码 + 通用知识...
    echo 数据: dataset\ue_pretrain_merged.jsonl
    python train_pretrain.py --data_path ../dataset/ue_pretrain_merged.jsonl --hidden_size %HIDDEN_SIZE% --num_hidden_layers %NUM_LAYERS% --epochs %PRETRAIN_EPOCHS% --batch_size %BATCH_SIZE% --learning_rate 5e-4 --max_seq_len %MAX_SEQ_LEN% --save_weight ue_pretrain --log_interval 100 --save_interval 500
    if errorlevel 1 (
        echo [错误] 预训练失败
        pause
        exit /b 1
    )
)

echo.
echo ══════════════════════════════════════════════════════════
echo [Step 4/6] SFT微调...
echo ══════════════════════════════════════════════════════════

if exist "..\out\ue_sft_%HIDDEN_SIZE%.pth" (
    echo [跳过] SFT权重已存在: out\ue_sft_%HIDDEN_SIZE%.pth
    echo 如需重新训练，请删除该文件
) else (
    echo 开始SFT微调 - 使用合并数据: UE专业 + 通用对话...
    echo 数据: dataset\ue_sft_merged.jsonl
    python train_full_sft.py --data_path ../dataset/ue_sft_merged.jsonl --hidden_size %HIDDEN_SIZE% --num_hidden_layers %NUM_LAYERS% --epochs %SFT_EPOCHS% --batch_size %BATCH_SIZE% --learning_rate 1e-5 --max_seq_len %MAX_SEQ_LEN% --from_weight ue_pretrain --save_weight ue_sft --log_interval 100 --save_interval 500
    if errorlevel 1 (
        echo [错误] SFT微调失败
        pause
        exit /b 1
    )
)

echo.
echo ══════════════════════════════════════════════════════════
echo [Step 5/6] 构建RAG索引...
echo ══════════════════════════════════════════════════════════
cd /d %~dp0
cd scripts

REM 检查依赖
python -c "import sentence_transformers; import faiss" >nul 2>&1
if errorlevel 1 (
    echo [安装] 正在安装RAG依赖...
    pip install sentence-transformers faiss-cpu -i https://mirrors.aliyun.com/pypi/simple
)

if exist "..\ue_index\index.faiss" (
    echo [跳过] RAG索引已存在: ue_index\
    echo 如需重新构建，请删除ue_index目录
) else (
    echo 正在构建RAG向量索引，预计耗时10-30分钟...
    python ue_rag_server.py build --ue_source_path "%UE_SOURCE_PATH%" --index_path ../ue_index
    if errorlevel 1 (
        echo [警告] RAG索引构建失败，但不影响基础模型使用
    )
)

echo.
echo ══════════════════════════════════════════════════════════
echo [Step 6/6] 完成！
echo ══════════════════════════════════════════════════════════
echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║                    训练完成！                            ║
echo ╠══════════════════════════════════════════════════════════╣
echo ║  生成的文件:                                             ║
echo ║    - dataset\ue_pretrain.jsonl (预训练数据)              ║
echo ║    - dataset\ue_sft.jsonl (SFT数据)                      ║
echo ║    - out\ue_pretrain_%HIDDEN_SIZE%.pth (预训练模型)              ║
echo ║    - out\ue_sft_%HIDDEN_SIZE%.pth (最终模型)                     ║
echo ║    - ue_index\ (RAG向量索引)                             ║
echo ╠══════════════════════════════════════════════════════════╣
echo ║  使用方式:                                               ║
echo ║    方式1: 纯LLM问答                                      ║
echo ║      run_ue_chat.bat                                     ║
echo ║                                                          ║
echo ║    方式2: LLM + RAG问答 (推荐，更精确)                   ║
echo ║      run_ue_rag.bat                                      ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

echo 是否立即启动问答服务? (1=纯LLM, 2=LLM+RAG, 其他=退出)
set /p choice=请选择: 

if "%choice%"=="1" (
    echo.
    echo 启动纯LLM问答服务...
    cd /d %~dp0
    python eval_llm.py --load_from model --weight ue_sft --hidden_size %HIDDEN_SIZE% --num_hidden_layers %NUM_LAYERS%
) else if "%choice%"=="2" (
    echo.
    echo 启动LLM+RAG问答服务...
    cd /d %~dp0
    cd scripts
    python ue_rag_server.py serve --index_path ../ue_index --top_k 5
) else (
    echo.
    echo 训练完成！你可以稍后运行问答服务。
)

pause
