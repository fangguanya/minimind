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

if not exist "out\ue_sft_%HIDDEN_SIZE%.pth" (
    echo [错误] 模型权重不存在: out\ue_sft_%HIDDEN_SIZE%.pth
    echo 请先运行 train_ue_full.bat 进行训练
    pause
    exit /b 1
)

echo 提示: 输入问题后按回车，输入 quit 退出
echo.

::python eval_llm.py --load_from model --weight ue_sft --hidden_size %HIDDEN_SIZE% --num_hidden_layers %NUM_LAYERS%
python eval_llm.py --load_from model --weight ue_pretrain --hidden_size %HIDDEN_SIZE% --num_hidden_layers %NUM_LAYERS%

pause
