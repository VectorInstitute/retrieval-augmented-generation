import os
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from langchain.docstore.document import Document
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers import StringIterableReader
from llama_index.core.node_parser import LangchainNodeParser
from llama_index.readers.s3 import S3Reader


class DocumentReader:
    def __init__(
        self,
        directory_path: str = None,
        s3_bucket: str = None,
        aws_access_id: str = None,
        aws_access_secret: str = None,
        aws_session_token: str = None,
        create_nodes: bool = False,
        web_search_query: str = None,
        search_k: int = 5,
    ):
        self.directory_path = directory_path
        self.s3_bucket = s3_bucket
        self.aws_access_id = aws_access_id
        self.aws_access_secret = aws_access_secret
        self.create_nodes = create_nodes
        self.web_search_query = web_search_query
        self.search_k = search_k
        self.chunk_size = 512
        self.chunk_overlap = 32
        self.docs = []
        self.chunks = []

    def _get_text_splitter(self):
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def _load_local_pdfs(self):
        if not self.directory_path or not os.path.exists(self.directory_path):
            raise FileNotFoundError(f"Directory {self.directory_path} does not exist.")
        if not any(f.endswith(".pdf") for f in os.listdir(self.directory_path)):
            raise FileNotFoundError(f"No PDF files found in {self.directory_path}.")

        if self.create_nodes:
            docs = SimpleDirectoryReader(input_dir=self.directory_path).load_data()
        else:
            docs = PyPDFDirectoryLoader(self.directory_path).load()

        return docs

    def _load_s3_pdfs(self):
        if not self.s3_bucket:
            raise ValueError("s3_bucket must be provided for S3 loading.")
        loader = S3Reader(
            bucket=self.s3_bucket,
            aws_access_id=self.aws_access_id,
            aws_access_secret=self.aws_access_secret,
        )
        return loader.load_data()

    def _load_web_search(self):
        docs = []
        # Check if k is a positive integer
        if self.search_k <= 0:
            raise ValueError("k must be a positive integer.")

        # Perform a web search and get the top k results
        for result_url in search(
            self.web_search_query, num_results=self.search_k, sleep_interval=2
        ):
            if result_url and result_url.startswith("http"):
                try:
                    response = requests.get(result_url)
                    soup = BeautifulSoup(response.content, "html.parser")
                    docs.append(soup.get_text())
                except Exception as e:
                    print(f"Failed to fetch {result_url}: {e}")

        if self.create_nodes:
            docs = StringIterableReader().load_data(texts=docs)
        else:
            docs = [
                Document(page_content=web_txt, metadata={"source": "web"})
                for web_txt in docs
            ]

        return docs

    def load(self):
        # Load documents
        if self.directory_path:
            self.docs = self._load_local_pdfs()
        elif self.s3_bucket:
            self.docs = self._load_s3_pdfs()
        elif self.web_search_query:
            self.docs = self._load_web_search()
        else:
            raise ValueError(
                "Either directory_path, s3_bucket, or web_search_query must be provided."
            )

        # Split documents
        splitter = self._get_text_splitter()
        if self.create_nodes:
            parser = LangchainNodeParser(splitter)
            self.chunks = parser.get_nodes_from_documents(self.docs)
        else:
            self.chunks = splitter.split_documents(self.docs)

        return self.docs, self.chunks


def pretty_print(docs: list):
    # If docs are llama_index documents, convert to langchain Document if needed
    if hasattr(docs[0], "text") and not hasattr(docs[0], "page_content"):
        docs = [Document(page_content=d.text) for d in docs]

    if not all(hasattr(d, "page_content") for d in docs):
        print("Not all documents have page_content attribute.")
        return

    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]
        )
    )


def download_file(url: str, directory_path: str):
    os.makedirs(directory_path, exist_ok=True)

    filename = os.path.join(directory_path, url.split("/")[-1])
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(filename, "wb") as file:
            file.write(response.content)
        print(f"Downloaded {url} to {filename}")
    except requests.RequestException as e:
        print(f"Failed to download {url}: {e}")
