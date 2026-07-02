import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_classic.storage import LocalFileStore
from utils import LocalEmbeddings

def ParentDocumentRetrieverSetup(docs, collection_name="research_papers", persist_directory="chroma_db", embedding_model_name="all-MiniLM-L6-v2", parent_chunk_size=1500, parent_chunk_overlap=200, child_chunk_size=200, child_chunk_overlap=20):
    vectorstore = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=LocalEmbeddings(model_name=embedding_model_name)
    )

    fs = LocalFileStore("parent_docs")
    docstore = create_kv_docstore(fs)

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap)

    retriever = ParentDocumentRetriever(
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        vectorstore=vectorstore,
        docstore=docstore
    )

    retriever.add_documents(docs)
    return retriever
