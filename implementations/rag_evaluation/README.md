# RAG Bootcamp: Evaluation Techniques

This is a reference implementations for Vector Institute's **RAG (Retrieval-Augmented Generation) Bootcamp**, taking place from Nov 2024 to Jan 2025. Popular LLMs like OpenAI's GPT-4o and Meta's Llama-3 are very good at natural language and sounding like humans, but their knowledge is limited by the data they were trained on.

When implementing **RAG (Retrieval-Augmented Generation)** workflows, it is very difficult to understand if you are getting high quality results. There are many different techniques to evaluate RAG workflows, as described in many online sources such as https://weaviate.io/blog/rag-evaluation. In these reference implementations, we focus on the [Ragas](https://docs.ragas.io/en/stable/) framework.

There are many different metrics to determine if a RAG workflow is producing high "quality" results. The Ragas framework defines these metrics at https://docs.ragas.io/en/latest/concepts/metrics/index.html. In these examples, we focus on the following three:

- **Faithfulness:** The generated answer is regarded as *faithful* if all the claims that are made in the answer can be inferred from the given context(s).
- **Answer Correctness:** Was the generated answer correct? Was it complete?
- **Context Precision:** Did our retriever return good results that matched the question it was being asked?

## Requirements

* Python 3.10+
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

    Update the value for `OPENAI_API_KEY` env variable.

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