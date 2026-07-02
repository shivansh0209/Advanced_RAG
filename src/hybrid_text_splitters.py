import uuid
import pickle
from langsmith import traceable
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_classic.storage import LocalFileStore
from src.hybrid_retrieval_class import CustomHybridParentRetriever
from src.utils import LocalEmbeddings, split_by_legal_section
from src.data_loaders  import load_documents


def _is_vectorstore_populated(persist_directory: str, collection_name: str, embedding_model_name: str) -> tuple[bool, Chroma]:
    """Returns (is_populated, vectorstore). Vectorstore is always initialized."""
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=LocalEmbeddings(model_name=embedding_model_name)
    )
    count = vectorstore._collection.count()
    return count > 0, vectorstore


def _load_bm25(bm25_path: str) -> BM25Retriever | None:
    """Load a pickled BM25Retriever if it exists, else return None."""
    if Path(bm25_path).exists():
        with open(bm25_path, "rb") as f:
            return pickle.load(f)
    return None


def _save_bm25(retriever: BM25Retriever, bm25_path: str):
    with open(bm25_path, "wb") as f:
        pickle.dump(retriever, f)


@traceable(name="parent_document_retriever_setup")
def parent_document_retriever_setup(
    docs=None,
    collection_name="research_papers",
    persist_directory="chroma_db",
    embedding_model_name="all-MiniLM-L6-v2",
    child_chunk_size=400,
    child_chunk_overlap=60,
    bm25_path="bm25_retriever.pkl",
    parent_docs_dir="parent_docs",
):
    file_store = LocalFileStore(parent_docs_dir)
    docstore = create_kv_docstore(file_store)

    # ── Check what's already persisted ──────────────────────────────────────
    already_indexed, vectorstore = _is_vectorstore_populated(
        persist_directory, collection_name, embedding_model_name
    )
    bm25_retriever = _load_bm25(bm25_path)
    docstore_populated = any(True for _ in Path(parent_docs_dir).iterdir()) if Path(parent_docs_dir).exists() else False

    if already_indexed and bm25_retriever is not None and docstore_populated:
        print("✅ All indexes found on disk — skipping ingestion, loading directly.")
        return CustomHybridParentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            bm25_retriever=bm25_retriever,
        )

    # Only load docs if we actually need to build indexes
    if docs is None:
        print("⚙️  Loading documents...")
        docs = load_documents()

    # ── Fresh build ──────────────────────────────────────────────────────────
    print("⚙️  Indexes not found (or incomplete) — building from scratch...")

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap
    )

    all_child_chunks = []
    parent_storage_map = {}

    # Step 1: Section-aware parent chunks
    parent_chunks = split_by_legal_section(docs)

    for parent in parent_chunks:
        parent_id = str(uuid.uuid4())
        parent.metadata["id"] = parent_id
        parent_storage_map[parent_id] = parent

        child_chunks = child_splitter.split_documents([parent])
        for child in child_chunks:
            child.metadata["id"] = parent_id
            all_child_chunks.append(child)

    # Step 2: Populate Vector Store (Chroma)
    for i in range(0, len(all_child_chunks), 5000):
        vectorstore.add_documents(all_child_chunks[i:i + 5000])
    print(f"✅ Chroma vectorstore populated with {len(all_child_chunks)} child chunks.")

    # Step 3: Build & persist BM25
    bm25_retriever = BM25Retriever.from_documents(all_child_chunks)
    _save_bm25(bm25_retriever, bm25_path)
    print(f"✅ BM25 index saved to '{bm25_path}'.")

    # Step 4: Populate docstore
    kv_pairs_to_save = []
    seen_ids = set()
    for child in all_child_chunks:
        pid = child.metadata["id"]
        if pid in parent_storage_map and pid not in seen_ids:
            kv_pairs_to_save.append((pid, parent_storage_map[pid]))
            seen_ids.add(pid)

    if kv_pairs_to_save:
        docstore.mset(kv_pairs_to_save)
    print(f"✅ Docstore populated with {len(kv_pairs_to_save)} parent docs.")

    # Step 5: Assemble and return the unified retriever
    return CustomHybridParentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        bm25_retriever=bm25_retriever,
    )




























# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# from langchain_chroma import Chroma
# from langchain_classic.retrievers import ParentDocumentRetriever
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_classic.storage._lc_store import create_kv_docstore
# from langchain_classic.storage import LocalFileStore
# from utils import LocalEmbeddings

# def ParentDocumentRetrieverSetup(docs, collection_name="research_papers", persist_directory="chroma_db", embedding_model_name="all-MiniLM-L6-v2", parent_chunk_size=1500, parent_chunk_overlap=200, child_chunk_size=200, child_chunk_overlap=20):
#     vectorstore = Chroma(
#         collection_name=collection_name,
#         persist_directory=persist_directory,
#         embedding_function=LocalEmbeddings(model_name=embedding_model_name)
#     )

#     fs = LocalFileStore("parent_docs")
#     docstore = create_kv_docstore(fs)

#     parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap)
#     child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap)

#     retriever = ParentDocumentRetriever(
#         child_splitter=child_splitter,
#         parent_splitter=parent_splitter,
#         vectorstore=vectorstore,
#         docstore=docstore
#     )

#     retriever.add_documents(docs)
#     return retriever