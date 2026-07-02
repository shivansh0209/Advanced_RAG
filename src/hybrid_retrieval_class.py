import cohere
from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from utils import rerank_with_cohere

class CustomHybridParentRetriever(BaseRetriever):
    vectorstore: Chroma
    docstore: any
    bm25_retriever: BM25Retriever
    top_k: int =6
    rrf_k: int =60


    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query):
        
        # 1. Fetch raw candidates from both search engines (get double top_k for filtering)
        if "|||" in query:
            original_query, hyde_answer = query.split("|||", 1)
        else:
            original_query = query
            hyde_answer = query

        fetch_count = self.top_k * 5
        vector_results = self.vectorstore.similarity_search(hyde_answer, k=fetch_count)
        self.bm25_retriever.k = self.top_k * 2
        bm25_results = self.bm25_retriever.invoke(original_query)[:fetch_count]

        # 2. Score parents using Reciprocal Rank Fusion (RRF)
        rrf_scores = {}

        def apply_rrf_scoring(search_results):
            for rank, doc in enumerate(search_results, start=1):
                parent_id = doc.metadata.get("id")
                if not parent_id:
                    continue
                
                # If it's a new ID, initialize it; otherwise add to the score
                rrf_scores[parent_id] = rrf_scores.get(parent_id, 0.0) + (1.0 / (self.rrf_k + rank))

        apply_rrf_scoring(vector_results)
        apply_rrf_scoring(bm25_results)

        sorted_parent_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        target_ids = sorted_parent_ids[:self.top_k]
        fetched_docs = self.docstore.mget(target_ids)
        final_parent_docs = [doc for doc in fetched_docs if doc is not None]

        if not final_parent_docs:
            return []

        print(f"Reranking {len(final_parent_docs)} parent documents using Cohere Rerank API...")

        return rerank_with_cohere(original_query, final_parent_docs, top_k=self.top_k)

