#!/bin/bash

steps=(200 400 600 800 1000 1200 1400 1584)
dataset_types=("paraphrase" "implication" "qa")

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
