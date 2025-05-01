#!/bin/bash

# Check whether .env file exists in current directory
if [ ! -f ".env" ]; then
  echo ".env file not found. Please ensure you are in the correct directory."
  exit 1
fi

source .env

# Check if the model argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <model> [--use-interrogation|--no-interrogation] [--use-context|--no-context]"
  echo "Example: $0 llama70b 3 --no-interrogation --use-context"
  exit 1
fi

# Define variables
MODEL=$1  # Model is passed as a command-line argument

# Parse optional flags
USE_INTERROGATION=true
USE_CONTEXT=true

# Shift past the first argument (model)
shift

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --use-interrogation) USE_INTERROGATION=true ;;
    --no-interrogation) USE_INTERROGATION=false ;;
    --use-context) USE_CONTEXT=true ;;
    --no-context) USE_CONTEXT=false ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

# Adjust command-line arguments for generate.py
INTERROGATION_FLAG=""
CONTEXT_FLAG=""
N_MAX=10  # Default value for n_max
if [ "$USE_INTERROGATION" = false ]; then
  INTERROGATION_FLAG="--no-interrogation"
  N_MAX=0  # Set n_max to 0 if no interrogation is used
fi
if [ "$USE_CONTEXT" = false ]; then
  CONTEXT_FLAG="--no-context"
fi

# Set complexity and features based on the model
if [ "$MODEL" = "llama70b" ] || [ "$MODEL" = "qwen72b" ]; then
  COMPLEXITY=1
  FEATURES='["add_macro_actions", "add_actions"]'
else
  COMPLEXITY=2
  FEATURES='["add_macro_actions", "add_observations"]'
fi

SCENARIOS=$(seq 0 9)  # Scenarios 0 to 9
LOG_FILE="run_success.log"

# Clear the log file
echo "Run success log for generate.py" > $LOG_FILE


# Iterate over scenarios
for SCENARIO in $SCENARIOS; do
  COMMAND="uv run python scripts/python/generate.py --scenario $SCENARIO --model $MODEL --complexity $COMPLEXITY --features "$FEATURES" --n-max $N_MAX $INTERROGATION_FLAG $CONTEXT_FLAG run"
  echo "Running command: $COMMAND"

  uv run python scripts/python/generate.py \
    --scenario $SCENARIO \
    --model $MODEL \
    --complexity $COMPLEXITY \
    --features "$FEATURES" \
    --n-max $N_MAX \
    $INTERROGATION_FLAG \
    $CONTEXT_FLAG \
    run

  # Check if the command succeeded
  if [ $? -eq 0 ]; then
    echo "Scenario $SCENARIO: SUCCESS" >> $LOG_FILE
    echo "Command: $COMMAND" >> $LOG_FILE
  else
    echo "Scenario $SCENARIO: FAILED" >> $LOG_FILE
    echo "Command: $COMMAND" >> $LOG_FILE
  fi

done

echo "All scenarios processed. Check $LOG_FILE for success details."