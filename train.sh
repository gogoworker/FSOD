#!/bin/bash
if python train.py configs/moe/voc/split1/moe_prototype_r101_c4_8xb4_voc-split1_base-training.py; then
    python train.py configs/moe/voc/split1/moe_prototype_r101_c4_8xb4_voc-split1_1shot-fine-tuning.py
    python train.py configs/moe/voc/split1/moe_prototype_r101_c4_8xb4_voc-split1_2shot-fine-tuning.py
    python train.py configs/moe/voc/split1/moe_prototype_r101_c4_8xb4_voc-split1_3shot-fine-tuning.py
    python train.py configs/moe/voc/split1/moe_prototype_r101_c4_8xb4_voc-split1_5shot-fine-tuning.py
    python train.py configs/moe/voc/split1/moe_prototype_r101_c4_8xb4_voc-split1_10shot-fine-tuning.py
else
    echo "base_train_error"
    exit 1
fi