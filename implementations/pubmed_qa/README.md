# RAG Bootcamp: Evaluation Techniques

This notebook presents a RAG workflow for the [PubMed QA](https://pubmedqa.github.io/) task using [LlamaIndex](https://www.llamaindex.ai/). The code is written in a configurable fashion, giving you the flexibility to edit the RAG configuration and observe the change in output/responses. It serves as a good template for various RAG workflows.

## Requirements

* Python 3.10+
* [OpenAI](https://platform.openai.com/) API Key 
* [Weaviate](https://console.weaviate.io/) cluster URL and API key

## Setup Instructions

Follow these steps to set up your environment for the RAG Bootcamp notebooks:

1. **Install [uv](https://github.com/astral-sh/uv):**

    ```bash
    pip install uv
    ```

2. **Install dependencies using uv:**

    ```bash
    uv sync --dev
    ```

3. **(Optional) Install GPU dependencies:**

    If you are on a Linux machine with GPU support, you can install the GPU dependency group:

    ```bash
    uv sync --dev --group gpu
    ```

4. **Configure environment variables:**

    Copy the example environment file and update it with your settings:

    ```bash
    cp .env.example .env
    ```

    Update the value for `OPENAI_API_KEY`, `WEAVIATE_API_KEY`, `WEAVIATE_CLUSTER_URL` env variables.

5. **Create and activate a virtual environment using uv:**

    ```bash
    uv venv .venv
    source .venv/bin/activate
    ```

6. **Install the Jupyter kernel:**

    ```bash
    uv run ipython kernel install --user --name=rag-bootcamp  
    ```

7. **Start Jupyter Lab with environment variables loaded:**

    ```bash
    uv run --env-file .env jupyter lab
    ```
