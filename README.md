# RAG Bootcamp

This is a collection of reference implementations for Vector Institute's **RAG (Retrieval-Augmented Generation) Bootcamp**, that took place from Nov 2024 to Jan 2025. It demonstrates some of the common methodologies used in RAG workflows (data ingestion, chunks, embeddings, vector databases, sparse/dense retrieval, reranking) using the popular Python [LangChain](https://python.langchain.com/docs/get_started/introduction) and [LlamaIndex](https://docs.llamaindex.ai/en/stable/) libraries.

## Reference Implementations

This repository includes several reference implementations showing different approaches and methodologies related to Retrieval-Augmented Generation.

- [**Web Search**](https://github.com/VectorInstitute/rag_bootcamp/tree/main/web_search): Popular LLMs like OpenAI's GPT-4o and Meta's Llama-3 are very good at processing natural language, but their knowledge is limited by the data they were trained on. As of November 2024, neither service can correctly answer the question "Who won the 2024 World Series of Baseball?"
- [**Document Search**](https://github.com/VectorInstitute/rag_bootcamp/tree/main/document_search): Use a collection of unstructured documents to answer domain-specific questions, like: "How many AI scholarships did Vector Institute award in 2022?"
- [**SQL Search**](https://github.com/VectorInstitute/rag_bootcamp/tree/main/sql_search): Answer natural language questions with information from structured relational data. This demo uses a financial dataset from a Portugese banking instituation, [available on Kaggle](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets)
- [**Cloud Search**](https://github.com/VectorInstitute/rag_bootcamp/tree/main/cloud_search): Retrieve information from data in a cloud service, in this example AWS S3 storage
- [**PubMed QA**](https://github.com/VectorInstitute/rag_bootcamp/tree/main/pubmed_qa): A full pipeline on the [PubMed](https://pubmed.ncbi.nlm.nih.gov/download/) dataset demonstrating ingestion, embeddings, vector index/storage, retrieval, reranking, with a focus on evaluation metrics.
- [**RAG Evaluation**](https://github.com/VectorInstitute/rag_bootcamp/tree/main/rag_evaluation): RAG evaluation techniques based on the [Ragas](https://github.com/explodinggradients/ragas) framework. Focuses on evaluation "test sets" and how to use these to determine how well a RAG pipeline is actually working.

## Requirements

* Python 3.10+

## Git Repostory

Start by cloning this git repository to a local folder:

```
git clone https://github.com/VectorInstitute/rag-bootcamp
```

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
    # Edit .env and add all required environment variables
    ```

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

You are now ready to use the RAG Bootcamp notebooks!

## Opening Notebooks in Google Colab

Each notebook in this repository includes an **"Open in Colab"** badge at the top. To run a notebook in Google Colab:

1. Navigate to the desired notebook in the GitHub repository.
2. Click the **"Open in Colab"** badge at the top of the notebook.
3. The notebook will open in Google Colab, where you can run the code interactively or make a copy to make changes to it.
4. Select T4 GPU as the runtime type.

> **Note:**  
> Some features (such as access to local files or environment variables) may require additional configuration or may not be fully supported in Colab. For best results, review any instructions provided in the notebook itself.