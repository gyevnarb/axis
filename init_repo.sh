#!/bin/sh

# Install uv
if ! command -v uv &> /dev/null
then
    echo "uv could not be found"
    echo "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "uv is already installed"
fi

# add binaries to PATH if they aren't added yet
# affix colons on either side of $PATH to simplify matching
case ":${PATH}:" in
    *:"$HOME/.local/bin":*)
        ;;
    *)
        # Prepending path in case a system-installed binary needs to be overridden
        export PATH="$HOME/.local/bin:$PATH"
        ;;
esac

# Ask user if they want to sync
SYNC_ENV="no"
echo -n "Do you want to sync the virtual environment? (y/N): "
read answer
if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
    SYNC_ENV="yes"
fi

if [ "$SYNC_ENV" = "yes" ]; then
    # Sync environment
    # Check whether .venv already exists if so delete it and uv snyc
    if [ -d ".venv" ]
    then
        echo ".venv already exists"
        echo "Deleting .venv"
        rm -rf .venv
    else
        echo ".venv does not exist"
        echo "Syncing environment"
    fi
    echo "Syncing virtual environment"
    uv sync --all-extras
else
    echo "Skipping uv sync..."
fi

# Login to huggingface-hub
uv run huggingface-cli login