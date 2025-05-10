#!/bin/bash

# Check whether .env file exists in current directory
if [ ! -f ".env" ]; then
  echo ".env file not found. Please ensure you are in the correct directory."
  exit 1
fi

source .env

# Check if the model argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <genmodel> [--use-features|--no-features] [--use-interrogation|--no-interrogation] [--use-context|--no-context] [--explanation-kind <final|all>] [--override]"
  echo "Example: $0 llama70b 3 --no-interrogation --use-context --explanation-kind final --override"
  exit 1
fi

# Define variables
MODEL=$1  # Model is passed as a command-line argument
EXPLANATION_KIND="final"  # Default explanation kind

# Parse optional flags
USE_INTERROGATION=true
USE_CONTEXT=true
USE_FEATURES=true
OVERRIDE=false

# Shift past the first argument (model)
shift

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --use-interrogation) USE_INTERROGATION=true ;;
    --no-interrogation) USE_INTERROGATION=false ;;
    --use-context) USE_CONTEXT=true ;;
    --no-context) USE_CONTEXT=false ;;
    --use-features) USE_FEATURES=true ;;
    --no-features) USE_FEATURES=false ;;
    --explanation-kind) shift; EXPLANATION_KIND=$1 ;;  # Set explanation kind
    --override) OVERRIDE=true ;;  # Add override flag
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

SCENARIOS=$(seq 0 9)  # Scenarios 0 to 9
LOG_FILE="eval_success.log"

# Check if the log file exists
if [ -f "$LOG_FILE" ]; then
  echo "Appending to existing log file: $LOG_FILE"
  echo "----- Appending new run details -----" >> $LOG_FILE
else
  echo "Run success log for generate.py" > $LOG_FILE
fi

# Iterate over scenarios
for SCENARIO in $SCENARIOS; do
  RESULTS_FILE="${MODEL}"  # Base results file name

  # Append flags to the results file name
  if [ "$USE_FEATURES" = true ]; then
    RESULTS_FILE+="_features"
  fi
  if [ "$USE_INTERROGATION" = true ]; then
    RESULTS_FILE+="_interrogation"
  fi
  if [ "$USE_CONTEXT" = true ]; then
    RESULTS_FILE+="_context"
  fi
  RESULTS_FILE+=".pkl"  # Add file extension

  # Check if the results file exists
  if [ ! -f "output/igp2/scenario${SCENARIO}/results/$RESULTS_FILE" ]; then
    echo "Results file not found: output/igp2/scenario${SCENARIO}/results/$RESULTS_FILE"
    continue
  fi

  COMMAND="uv run python scripts/python/evaluate.py --scenario $SCENARIO --model claude35 --results-file $RESULTS_FILE --explanation-kind $EXPLANATION_KIND"
  if [ "$OVERRIDE" = true ]; then
    COMMAND+=" --override"  # Append override flag
  fi
  echo "Running command: $COMMAND"

  $COMMAND

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