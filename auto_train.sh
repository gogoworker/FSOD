#!/bin/bash

# 配置项
TARGET_PID=746910
YOUR_TRAIN_CMD=" python train.py configs/moe/voc/split1/moe_prototype_r101_c4_8xb4_voc-split1_base-training.py"  # 你的训练命令


echo "等待 PID $TARGET_PID 结束..."
while kill -0 $TARGET_PID >/dev/null 2>&1; do
    sleep 600  # 每 600 秒检查一次
done

echo "目标进程已结束，启动训练任务..."
$YOUR_TRAIN_CMD