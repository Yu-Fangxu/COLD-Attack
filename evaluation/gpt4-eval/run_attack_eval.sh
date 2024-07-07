#!/bin/bash

export OPENAI_API_KEY=sk-aa
# Replace OPENAI_API_KEY with your open-ai key

# Set the folder containing your model output files
MODEL_OUTPUT_FOLDER=$1

# Set the folder where you'd like to save evaluation outputs
EVAL_OUTPUT_FOLDER=$2

# Array of filenames
# Loop through each filename and run the commands
for filename in "$MODEL_OUTPUT_FOLDER"/processed_*.json; do
  echo "Processing $filename ..."

  python evaluate.py  \
      --mode "score_safety" \
      --subtask "attack" \
      --model "gpt-4-0314" \
      --first_file "${filename}" \
      --output_file "${filename}-score_multi.gpt4.json"

  echo "Done with $filename."
done
