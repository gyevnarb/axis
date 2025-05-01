#!/bin/sh

# Go to working directory
cd /pvc/axs
source .env


# Install uv
if ! command -v uv &> /dev/null
then
    echo "uv could not be found"
    echo "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "uv is already installed"
fi

source $HOME/.local/bin/env

# Sync environment
# Check whether .venv already exists if so delete it and uv snyc
echo ".venv already exists"
echo "Deleting .venv"
rm -rf .venv
echo "Syncing virtual environment"
uv sync -U --all-extras


# Check if the model argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <model> [complexity]"
  echo "Example: $0 llama70b 3"
  exit 1
fi


# Define variables
MODEL=$1  # Model is passed as a command-line argument

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