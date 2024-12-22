#!/bin/bash

# 定义要运行的模型类型
model_types=('TCTRec' 'SASRec' 'GRU4Rec' 'BSARec')

# 定义数据集名称
datasets=('ml-100k' 'ml-1m' 'ml-20m')

# 循环遍历每个模型类型和数据集，并运行 main.py
for model_type in "${model_types[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo "Testing model: $model_type on dataset: $dataset"
        python main.py --model_type "$model_type" --load_model "$model_type" --data_name "$dataset" --do_eval
    done
done