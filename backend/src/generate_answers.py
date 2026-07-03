from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langsmith import traceable



@traceable(name="generate_answer")
def generate_answer(query, retrieved_docs):
    llm = ChatGroq(model="llama-3.3-70b-versatile")  

    prompt = ChatPromptTemplate([
    ("system", 
     "You are a legal expert on Indian statutes.\n"
     "Answer using ONLY the provided context. If the context covers multiple acts, address each one separately.\n"
     "If the answer is not in the context, say 'Not found in context.'\n"
     "Be precise. Cite section numbers when available."),
    ("user", "{context}\n\nQuestion: {question}")
])

    parser = StrOutputParser()

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    chain = prompt | llm | parser
    response = chain.invoke({"context": context, "question": query})
    return response