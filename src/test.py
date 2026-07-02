from text_splitters import retriever_setup
from langchain_core.documents import Document
import chromadb

doc = Document(page_content="Python is an interpreted programming language.", metadata={"source": "tech_doc"})
# retriever = retriever_setup([doc])

client = chromadb.PersistentClient("chroma_db")
collection = client.get_collection("research_papers")
data = collection.get(include=["embeddings"])
print(data)