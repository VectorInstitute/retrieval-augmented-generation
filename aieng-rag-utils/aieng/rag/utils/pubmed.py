import os
import re
import faiss
import weaviate
import json

from . import get_device_name

import torch.utils.data as data

from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets


from langchain_cohere import ChatCohere
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.openai_like import OpenAILike

from llama_index.core import (
    SimpleDirectoryReader,
    PromptTemplate,
    get_response_synthesizer,
)
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.json import JSONReader
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.vector_stores.weaviate import WeaviateVectorStore

from ragas import EvaluationDataset, evaluate as ragas_evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    NonLLMContextPrecisionWithReference,
    NonLLMContextRecall,
    ResponseRelevancy,
)

RAGAS_METRIC_MAP = {
    "faithfulness": Faithfulness(),
    "relevancy": ResponseRelevancy(),
    "recall": NonLLMContextRecall(),
    "precision": NonLLMContextPrecisionWithReference(),
}

class PubMedQATaskDataset(data.Dataset):
    def __init__(self, name, all_folds=False, split="test"):
        self.name = name
        subset_str = "pubmed_qa_labeled_fold{fold_id}"
        folds = [0] if not all_folds else list(range(10))

        bigbio_data = []
        source_data = []
        for fold_id in folds:
            bb_data = load_dataset(
                self.name,
                f"{subset_str.format(fold_id=fold_id)}_bigbio_qa",
                split=split,
                trust_remote_code=True,
            )
            s_data = load_dataset(
                self.name,
                f"{subset_str.format(fold_id=fold_id)}_source",
                split=split,
                trust_remote_code=True,
            )
            bigbio_data.append(bb_data)
            source_data.append(s_data)
        bigbio_data = concatenate_datasets(bigbio_data)
        source_data = concatenate_datasets(source_data)

        keys_to_keep = ["id", "question", "context", "answer", "LONG_ANSWER"]
        data_elms = []
        for elm_idx in tqdm(range(len(bigbio_data)), desc="Preparing data"):
            data_elms.append({k: bigbio_data[elm_idx][k] for k in keys_to_keep[:4]})
            data_elms[-1].update(
                {keys_to_keep[-1].lower(): source_data[elm_idx][keys_to_keep[-1]]}
            )

        self.data = data_elms

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def mock_knowledge_base(
        self,
        output_dir,
        one_file_per_sample=False,
        samples_per_file=500,
        sep="\n",
        jsonl=False,
    ):
        """
        Write PubMed contexts to a text file, newline seperated
        """
        pubmed_kb_dir = os.path.join(output_dir, "pubmed_doc")
        os.makedirs(pubmed_kb_dir, exist_ok=True)

        file_ext = "jsonl" if jsonl else "txt"

        if not one_file_per_sample:
            context_str = ""
            context_files = []
            for idx in range(len(self.data)):
                if (idx + 1) % samples_per_file == 0:
                    context_files.append(context_str.rstrip(sep))
                else:
                    if jsonl:
                        context_elm_str = json.dumps(
                            {
                                "id": self.data[idx]["id"],
                                "context": self.data[idx]["context"],
                            }
                        )
                    else:
                        context_elm_str = self.data[idx]["context"]
                    context_str += f"{context_elm_str}{sep}"

            for file_idx in range(len(context_files)):
                filepath = os.path.join(pubmed_kb_dir, f"context{file_idx}.{file_ext}")
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(context_files[file_idx])

        else:
            assert not jsonl, "Does not support jsonl if one_file_per_sample is True"
            for idx in range(len(self.data)):
                filepath = os.path.join(
                    pubmed_kb_dir, f'{self.data[idx]["id"]}.{file_ext}'
                )
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(self.data[idx]["context"])


class RAGLLM:
    """
    LlamaIndex supports OpenAI, Cohere, AI21 and HuggingFace LLMs
    https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom.html
    """

    def __init__(self, llm_type, llm_name, api_base=None, api_key=None):
        self.llm_type = llm_type
        self.llm_name = llm_name

        self._api_base = api_base
        self._api_key = api_key

        self.local_model_path = "/model-weights"

    def load_model(self, **kwargs):
        print(f"Configuring {self.llm_type} LLM model ...")
        gen_arg_keys = ["temperature", "top_p", "top_k", "do_sample"]
        gen_kwargs = {k: v for k, v in kwargs.items() if k in gen_arg_keys}
        if self.llm_type == "local":
            # Using local HuggingFace LLM stored at /model-weights
            llm = HuggingFaceLLM(
                tokenizer_name=f"{self.local_model_path}/{self.llm_name}",
                model_name=f"{self.local_model_path}/{self.llm_name}",
                device_map="auto",
                context_window=4096,
                max_new_tokens=kwargs["max_new_tokens"],
                generate_kwargs=gen_kwargs,
                # model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True},
            )
        elif self.llm_type in ["openai", "kscope"]:
            llm = OpenAILike(
                model=self.llm_name,
                api_base=self._api_base,
                api_key=self._api_key,
                is_chat_model=True,
                temperature=kwargs["temperature"],
                max_tokens=kwargs["max_new_tokens"],
                top_p=kwargs["top_p"],
                top_k=kwargs["top_k"],
            )
        return llm



class DocumentReader:
    def __init__(
        self,
        input_dir,
        exclude_llm_metadata_keys=True,
        exclude_embed_metadata_keys=True,
    ):
        self.input_dir = input_dir
        self._file_ext = os.path.splitext(os.listdir(input_dir)[0])[1]

        self.exclude_llm_metadata_keys = exclude_llm_metadata_keys
        self.exclude_embed_metadata_keys = exclude_embed_metadata_keys

    def load_data(self):
        docs = None
        # Use reader based on file extension of documents
        # Only support '.txt' files as of now
        if self._file_ext == ".txt":
            reader = SimpleDirectoryReader(input_dir=self.input_dir)
            docs = reader.load_data()
        elif self._file_ext == ".jsonl":
            reader = JSONReader()
            docs = []
            for file in os.listdir(self.input_dir):
                docs.extend(
                    reader.load_data(os.path.join(self.input_dir, file), is_jsonl=True)
                )
        else:
            raise NotImplementedError(
                f"Does not support {self._file_ext} file extension for document files."
            )

        # Can choose if metadata need to be included as input when passing the doc to LLM or embeddings:
        # https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents.html
        # Exclude metadata keys from embeddings or LLMs based on flag
        if docs is not None:
            all_metadata_keys = list(docs[0].metadata.keys())
            if self.exclude_llm_metadata_keys:
                for doc in docs:
                    doc.excluded_llm_metadata_keys = all_metadata_keys
            if self.exclude_embed_metadata_keys:
                for doc in docs:
                    doc.excluded_embed_metadata_keys = all_metadata_keys

        return docs


class RAGEmbedding:
    """
    LlamaIndex supports embedding models from OpenAI, Cohere, HuggingFace, etc.
    https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html
    We can also build out custom embedding model:
    https://docs.llamaindex.ai/en/stable/module_guides/models/embeddings.html#custom-embedding-model
    """

    def __init__(self, model_type, model_name):
        self.model_type = model_type
        self.model_name = model_name

    def load_model(self):
        print(f"Loading {self.model_type} embedding model ...")
        device = get_device_name()
    
        if self.model_type == "hf":
            # Using bge base HuggingFace embeddings, can choose others based on leaderboard:
            # https://huggingface.co/spaces/mteb/leaderboard
            embed_model = HuggingFaceEmbedding(
                model_name=self.model_name,
                device=device,
                trust_remote_code=True,
            )  # max_length does not have any effect?

        elif self.model_type == "openai":
            # TODO - Add OpenAI embedding model
            # embed_model = OpenAIEmbedding()
            raise NotImplementedError

        return embed_model


class RAGQueryEngine:
    """
    https://docs.llamaindex.ai/en/stable/understanding/querying/querying.html
    TODO - Check other args for RetrieverQueryEngine
    """

    def __init__(self, retriever_type, vector_index):
        self.retriever_type = retriever_type
        self.index = vector_index
        self.retriever = None
        self.node_postprocessor = None
        self.response_synthesizer = None

    def create(self, similarity_top_k, response_mode, **kwargs):
        self.set_retriever(similarity_top_k, **kwargs)
        self.set_response_synthesizer(response_mode=response_mode)
        if kwargs["use_reranker"]:
            self.set_node_postprocessors(rerank_top_k=kwargs["rerank_top_k"])
        query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            node_postprocessors=self.node_postprocessor,
            response_synthesizer=self.response_synthesizer,
        )
        return query_engine

    def set_retriever(self, similarity_top_k, **kwargs):
        # Other retrievers can be used based on the type of index: List, Tree, Knowledge Graph, etc.
        # https://docs.llamaindex.ai/en/stable/api_reference/query/retrievers.html
        # Find LlamaIndex equivalents for the following:
        # Check MultiQueryRetriever from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever
        # Check Contextual compression from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/contextual_compression/
        # Check Ensemble Retriever from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/ensemble
        # Check self-query from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/self_query
        # Check WebSearchRetriever from LangChain: https://python.langchain.com/docs/modules/data_connection/retrievers/web_research
        if self.retriever_type == "vector_index":
            self.retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=similarity_top_k,
                vector_store_query_mode=kwargs["query_mode"],
                alpha=kwargs["hybrid_search_alpha"],
            )
        elif self.retriever_type == "bm25":
            self.retriever = BM25Retriever(
                nodes=kwargs["nodes"],
                tokenizer=kwargs["tokenizer"],
                similarity_top_k=similarity_top_k,
            )
        else:
            raise NotImplementedError(
                f"Incorrect retriever type - {self.retriever_type}"
            )

    def set_node_postprocessors(self, rerank_top_k=2):
        # Node postprocessor: Porcessing nodes after retrieval before passing to the LLM for generation
        # Re-ranking step can be performed here!
        # Nodes can be re-ordered to include more relevant ones at the top: https://python.langchain.com/docs/modules/data_connection/document_transformers/post_retrieval/long_context_reorder
        # https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors.html

        self.node_postprocessor = [LLMRerank(top_n=rerank_top_k)]

    def set_response_synthesizer(self, response_mode):
        # Other response modes: https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/root.html#configuring-the-response-mode
        qa_prompt_tmpl = (
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, answer the query while providing an explanation. "
            "If your answer is in favour of the query, end your response with 'yes' otherwise end your response with 'no'.\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl)

        self.response_synthesizer = get_response_synthesizer(
            text_qa_template=qa_prompt_tmpl,
            response_mode=response_mode,
        )


class RagasEval:
    def __init__(
        self, metrics, eval_llm_type, eval_llm_name, embed_model_name, **kwargs
    ):
        self.eval_llm_type = eval_llm_type  # "openai", "cohere", "local", "kscope"
        self.eval_llm_name = eval_llm_name

        self.temperature = kwargs.get("temperature", 0.0)
        self.max_tokens = kwargs.get("max_tokens", 256)

        self.embed_model_name = embed_model_name

        self._prepare_embedding()
        self._prepare_llm()

        self.metrics = [RAGAS_METRIC_MAP[elm] for elm in metrics]

    def _prepare_data(self, data):
        return EvaluationDataset.from_list(data)

    def _prepare_embedding(self):
        device = get_device_name()
        model_kwargs = {"device": device, "trust_remote_code": True}
        encode_kwargs = {
            "normalize_embeddings": True
        }  # set True to compute cosine similarity

        self.eval_embedding = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name=self.embed_model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
        )

    def _prepare_llm(self):
        if self.eval_llm_type == "local":
            self.eval_llm = LangchainLLMWrapper(
                HuggingFaceEndpoint(
                    repo_id=f"meta-llama/{self.eval_llm_name}",
                    temperautre=self.temperature,
                    max_new_tokens=self.max_tokens,
                    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
                )
            )
        elif self.eval_llm_type == "openai":
            self.eval_llm = LangchainLLMWrapper(
                ChatOpenAI(
                    model=self.eval_llm_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    base_url=os.environ["OPENAI_BASE_URL"],
                    api_key=os.environ["OPENAI_API_KEY"],
                )
            )
        elif self.eval_llm_type == "cohere":
            self.eval_llm = LangchainLLMWrapper(
                ChatCohere(
                    model=self.eval_llm_name,
                )
            )

    def evaluate(self, data):
        eval_data = self._prepare_data(data)
        result = ragas_evaluate(
            dataset=eval_data,
            metrics=self.metrics,
            llm=self.eval_llm,
            embeddings=self.eval_embedding,
        )
        return result

def extract_yes_no(resp):
    match_pat = r"\b(?:yes|no)\b"
    match_txt = re.search(match_pat, resp, re.IGNORECASE)
    if match_txt:
        return match_txt.group(0)
    return "none"


def get_embed_model_dim(embed_model):
    embed_out = embed_model.get_text_embedding("Dummy Text")
    return len(embed_out)


def validate_rag_cfg(cfg):
    if cfg["query_mode"] == "hybrid":
        assert (
            cfg["hybrid_search_alpha"] is not None
        ), "hybrid_search_alpha cannot be None if query_mode is set to 'hybrid'"
    if cfg["vector_db_type"] == "weaviate":
        assert (
            cfg["weaviate_url"] is not None
        ), "weaviate_url cannot be None for weaviate vector db"

class RAGIndex:
    """
    Use storage context to set custom vector store
    Available options: https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores.html
    Use Chroma: https://docs.llamaindex.ai/en/stable/examples/vector_stores/ChromaIndexDemo.html
    LangChain vector stores: https://python.langchain.com/docs/modules/data_connection/vectorstores/
    """

    def __init__(self, db_type, db_name):
        self.db_type = db_type
        self.db_name = db_name
        self._persist_dir = f"./.{db_type}_index_store/"

    def create_index(self, docs, save=True, **kwargs):
        # Only supports Weaviate as of now
        if self.db_type == "weaviate":
           
            weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
            weaviate_client = weaviate.connect_to_wcs(
                cluster_url=kwargs["weaviate_url"],
                auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key),
            )
            vector_store = WeaviateVectorStore(
                weaviate_client=weaviate_client,
                index_name=self.db_name,
            )
        elif self.db_type == "local":
            # Use FAISS vector database for local index
            faiss_dim = get_embed_model_dim(kwargs["embed_model"])
            faiss_index = faiss.IndexFlatL2(faiss_dim)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
        else:
            raise NotImplementedError(f"Incorrect vector db type - {self.db_type}")

        if os.path.isdir(self._persist_dir):
            # Load if index already saved
            print(f"Loading index from {self._persist_dir} ...")
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=self._persist_dir,
            )
            index = load_index_from_storage(storage_context)
        else:
            # Re-index
            print("Creating new index ...")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(
                docs, storage_context=storage_context
            )
            if save:
                os.makedirs(self._persist_dir, exist_ok=True)
                index.storage_context.persist(persist_dir=self._persist_dir)

        return index

