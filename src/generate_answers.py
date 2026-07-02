from langchain_ollama import ChatOllama 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="mistral")

prompt = ChatPromptTemplate([
    ("system", "Answer the question based only on the following context and say no if the answer is not contained within the context.\nJust give the answer without any additional commentary or explanation."),
    ("user", "{context}\n\nQuestion: {question}")
])

parser = StrOutputParser()

def generate_answer(query, retrieved_docs):
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    chain = prompt | llm | parser
    response = chain.invoke({"context": context, "question": query})
    return response