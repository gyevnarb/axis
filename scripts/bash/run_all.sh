#!/bin/bash

# Check if the model argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <model> [complexity]"
  echo "Example: $0 llama70b 3"
  exit 1
fi

# Define variables
MODEL=$1  # Model is passed as a command-line argument
COMPLEXITY=${2:-2}  # Complexity is optional, default is 2
SCENARIOS=$(seq 0 9)  # Scenarios 0 to 9
FEATURES='["add_macro_actions", "add_layout", "add_observations"]'
LOG_FILE="run_success.log"

# Clear the log file
echo "Run success log for generate.py" > $LOG_FILE

cd /pvc/axs

# Iterate over scenarios
for SCENARIO in $SCENARIOS; do
  echo "Running scenario $SCENARIO with model $MODEL and complexity $COMPLEXITY..."
  uv run python /pvc/axs/scripts/python/generate.py \
    --scenario $SCENARIO \
    --model $MODEL \
    --complexity $COMPLEXITY \
    --features "$FEATURES" \
    --n-max 10 \
    run

  # Check if the command succeeded
  if [ $? -eq 0 ]; then
    echo "Scenario $SCENARIO: SUCCESS" >> $LOG_FILE
  else
    echo "Scenario $SCENARIO: FAILED" >> $LOG_FILE
  fi
done

echo "All scenarios processed. Check $LOG_FILE for success details."