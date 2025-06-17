# aieng-rag-utils

`aieng-rag-utils` is a Python library providing utility functions and tools to streamline Retrieval-Augmented Generation (RAG) workflows. It is designed to help AI engineers build, evaluate, and deploy RAG-based applications efficiently.

## Features

- Document loaders, chunkers and pretty-print
- Web Search
- Vector store integrations
- Query and retrieval helpers
- Evaluation and benchmarking tools

## Installation

```bash
pip3 install aieng-rag-utils
```

## Sample Usage

```python
from aieng.rag.utils import get_device_name
from aieng.rag.utils.search import DocumentReader
from aieng.rag.utils.pubmed import RAGLLM

device = get_device_name()

doc_reader = DocumentReader(directory_path="./source_documents")
docs, chunks = doc_reader.load()

llm = RAGLLM(
    llm_type="openai",
    llm_name="gpt-4o",
    api_base=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
).load_model(**rag_cfg) # RAG Configuration

```
