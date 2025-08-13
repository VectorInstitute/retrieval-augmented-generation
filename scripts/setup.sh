#!/bin/bash

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

if [ -d ".venv" ]; then
    echo "Virtual environment already exists."
else
    echo "Creating virtual environment..."
    uv venv .venv
fi

source .venv/bin/activate
uv sync --dev

echo "Virtual environment activated and dependencies synced."

# Install Jupyter kernel
uv run ipython kernel install --user --name=retrieval-augmented-generation --display-name "RAG"
echo "Jupyter kernel installed."

# Start Jupyter lab in the background
echo "Starting Jupyter lab..."
nohup uv run jupyter lab --no-browser --port=8888 --ip=0.0.0.0 --ServerApp.token='' & > jupyter.log 2>&1 &
