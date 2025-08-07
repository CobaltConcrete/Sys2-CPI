#!/bin/bash

steps=(20 40 60 80 100 120 140 159)
dataset_types=("cheating_context" "misconception_rewrite" "misconception_detect")

for step in "${steps[@]}"; do
  for dataset_type in "${dataset_types[@]}"; do
    echo "=============================================="
    echo "Running: step=$step, dataset_type=$dataset_type"
    start_time=$(date +%s)

    python generate_input_response_data.py -step "$step" -ft_dataset_type "$dataset_type"

    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "Finished: step=$step, dataset_type=$dataset_type"
    echo "Duration: ${duration}s"
    echo "=============================================="
    echo ""
  done
done
