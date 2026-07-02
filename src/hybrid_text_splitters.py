import uuid
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_classic.storage import LocalFileStore
from hybrid_retrieval_class import CustomHybridParentRetriever
from utils import LocalEmbeddings


def parent_document_retriever_setup(docs, collection_name="research_papers", persist_directory="chroma_db", embedding_model_name="all-MiniLM-L6-v2", parent_chunk_size=1500, parent_chunk_overlap=200, child_chunk_size=400, child_chunk_overlap=60):
    
    # Initialize our text splitters
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap)

    all_child_chunks = []
    parent_storage_map = {}

    # Step 1: Chunk documents and link them together via unique IDs
    for doc in docs:
        parent_chunks = parent_splitter.split_documents([doc])
        
        for parent in parent_chunks:
            # Generate a distinct ID for this parent chunk
            parent_id = str(uuid.uuid4())
            parent.metadata["id"] = parent_id
            parent_storage_map[parent_id] = parent  # Map it for ultra-fast lookup later

            # Break parent down into smaller child chunks
            child_chunks = child_splitter.split_documents([parent])
            for child in child_chunks:
                child.metadata["id"] = parent_id  # Point child back to parent
                all_child_chunks.append(child)

    # Step 2: Initialize & populate Vector Store (Chroma)
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=LocalEmbeddings(model_name=embedding_model_name)
    )
    vectorstore.add_documents(all_child_chunks)

    # Step 3: Initialize Keyword Search (BM25)
    bm25_retriever = BM25Retriever.from_documents(all_child_chunks)
    

    file_store = LocalFileStore("parent_docs")
    docstore = create_kv_docstore(file_store)
    
    # Build a list of (parent_id, parent_document) pairs to save all at once
    kv_pairs_to_save = []
    for child in all_child_chunks:
        pid = child.metadata["id"]
        if pid in parent_storage_map:
            # LangChain's mset expects a list of (key, value) tuples
            kv_pairs_to_save.append((pid, parent_storage_map[pid]))
            
    # Save everything in one single, fast operation
    if kv_pairs_to_save:
        docstore.mset(kv_pairs_to_save)

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
