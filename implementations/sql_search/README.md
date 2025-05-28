# RAG Bootcamp: SQL Search

This is a reference implementations for Vector Institute's **RAG (Retrieval-Augmented Generation) Bootcamp**, taking place from Nov 2024 to Jan 2025. Popular LLMs like OpenAI's GPT-4o and Meta's Llama-3 are very good at natural language and sounding like humans, but their knowledge is limited by the data they were trained on. 

In this demo, we answer natural language questions with information from SQL structured relational data. This uses a financial dataset from a Portugese banking instituation, [available on Kaggle](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets), stored in a sqlite3 database.

## Requirements

* Python 3.10+
* Kaggle username and API token ([how to get it](https://www.kaggle.com/docs/api))
* [OpenAI](https://platform.openai.com/) API Key 

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

    Update the value for `OPENAI_API_KEY`, `KAGGLE_USERNAME` and `KAGGLE_KEY` env variables.

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
