# RAG Bootcamp: Document Search

This is a reference implementations for Vector Institute's **RAG (Retrieval-Augmented Generation) Bootcamp**, taking place from Nov 2024 to Jan 2025. Popular LLMs like OpenAI's GPT-4o and Meta's Llama-3 are very good at natural language and sounding like humans, but their knowledge is limited by the data they were trained on. 

In this demo, we use a collection of unstructured documents to answer domain-specific questions, like: *How many AI scholarships did Vector Institute award in 2022?*. We augment a Llama-3 LLM request with an unstructured PDF document search to get the correct answer: **Vector Institute awarded 109 AI scholarships in 2022**

## Requirements

* Python 3.10+
* [OpenAI](https://platform.openai.com/) API Key 
* [Cohere](https://dashboard.cohere.com/api-keys) API Key

## Setup Instructions

Follow these steps to set up your environment for the RAG Bootcamp notebooks:

1. **Install [uv](https://github.com/astral-sh/uv):**

    ```bash
    pip install uv
    ```

2. **Create and activate a virtual environment using uv:**

    ```bash
    uv venv .venv
    source .venv/bin/activate
    ```
3. **Install dependencies using uv:**

    ```bash
    uv sync --dev
    ```

4. **(Optional) Install GPU dependencies:**

    If you are on a Linux machine with GPU support, you can install the GPU dependency group:

    ```bash
    uv sync --dev --group gpu
    ```

5. **Configure environment variables:**

    Copy the example environment file and update it with your settings:

    ```bash
    cp .env.example .env
    ```

    Update the value for `OPENAI_API_KEY` and `COHERE_API_KEY` env variables.

6. **Install the Jupyter kernel:**

    ```bash
    uv run ipython kernel install --user --name=rag-bootcamp  
    ```

7. **Start Jupyter Lab with environment variables loaded:**

    ```bash
    uv run --env-file .env jupyter lab
    ```