#!/bin/bash

steps=($(seq 800 400 7500))
dataset_types=("summary_15x1000biographies" "summary_15x1000capitals")
temps=(0.6)

for step in "${steps[@]}"; do
  for dataset_type in "${dataset_types[@]}"; do
    for temp in "${temps[@]}"; do
      echo "======================================================"
      echo "Running: step=$step, dataset_type=$dataset_type, temp=$temp"
      start_time=$(date +%s)

      python generate_input_response_data_1000.py \
        -step "$step" \
        -ft_dataset_type "$dataset_type" \
        -temperature "$temp"

      end_time=$(date +%s)
      duration=$((end_time - start_time))
      echo "Finished: step=$step, dataset_type=$dataset_type, temp=$temp"
      echo "Duration: ${duration}s"
      echo "======================================================"
      echo ""
    done
  done
done
