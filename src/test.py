from text_splitters import parent_document_retriever_setup
from langchain_core.documents import Document
from data_loaders import load_documents
from utils import get_hyde_answer
# retriever = retriever_setup([doc])


docs = load_documents()
# print(docs[0].page_content)
parent_retriever = parent_document_retriever_setup(docs)
query = "What is the abstract of the paper regarding Llama2"
hyde_answer = get_hyde_answer(query)
relevant_docs = parent_retriever._get_relevant_documents(query + " ||| " + hyde_answer)
print("Relevant Parent Documents:")
for doc in relevant_docs:
    print(f"Document ID: {doc.metadata.get('id')}, Content: {doc.page_content}...")  # Print first 100 characters of content