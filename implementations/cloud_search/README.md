# RAG Bootcamp: Cloud Search

This is a reference implementations for Vector Institute's **RAG (Retrieval-Augmented Generation) Bootcamp**, taking place from Nov 2024 to Jan 2025. Popular LLMs like OpenAI's GPT-4o and Meta's Llama-3 are very good at natural language and sounding like humans, but their knowledge is limited by the data they were trained on. 

In this demo, we use a data loader from [Llama Hub](https://llamahub.ai/) to access an AWS S3 bucket and search documents stored in the cloud.

## Requirements

* Python 3.10+
* [OpenAI](https://platform.openai.com/) API Key 
* [AWS Credentials](https://dashboard.cohere.com/api-keys)
* [AWS S3 bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/upload-objects.html) containing the [Vector Institute 2021-22 Annual Report (PDF)](https://vectorinstitute.ai/wp-content/uploads/2023/05/vector-institute-2021-22-annual-report_accessible.pdf) file in it.

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

4. **Configure environment variables:**

    Copy the example environment file and update it with your settings:

    ```bash
    cp .env.example .env
    ```

    Update the value for `OPENAI_API_KEY`, `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` env variables.

5. **Install the Jupyter kernel:**

    ```bash
    uv run ipython kernel install --user --name=rag-bootcamp  
    ```

6. **Start Jupyter Lab with environment variables loaded:**

    ```bash
    uv run --env-file .env jupyter lab
    ```