@echo off
chcp 65001 >nul
REM ====================================================
REM MiniMind-UE 纯LLM问答
REM ====================================================

SET HIDDEN_SIZE=512
SET NUM_LAYERS=8

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║          MiniMind-UE 纯LLM问答服务                       ║
echo ╠══════════════════════════════════════════════════════════╣
echo ║  模式: 纯大语言模型回答                                  ║
echo ║  优点: 响应快速，回答自然                                ║
echo ║  缺点: 可能不够精确，文件路径可能编造                    ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

cd /d %~dp0

REM 优先使用纯UE训练的模型
if exist "out\ue_sft_pure_%HIDDEN_SIZE%.pth" (
    SET MODEL_WEIGHT=ue_sft_pure
) else if exist "out\ue_sft_%HIDDEN_SIZE%.pth" (
    SET MODEL_WEIGHT=ue_sft
) else (
    echo [错误] 模型权重不存在
    echo 请先运行 train_ue_full.bat 或 train_ue_simple.py 进行训练
    pause
    exit /b 1
)

echo 使用模型: %MODEL_WEIGHT%_%HIDDEN_SIZE%.pth
echo 提示: 输入问题后按回车，输入 quit 退出
echo.

python eval_llm.py --load_from model --weight %MODEL_WEIGHT% --hidden_size %HIDDEN_SIZE% --num_hidden_layers %NUM_LAYERS%

pause
