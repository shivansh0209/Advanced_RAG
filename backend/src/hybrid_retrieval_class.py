from typing import Any
from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever
from langchain_community.retrievers import BM25Retriever
from src.utils import rerank_with_cohere
from langsmith import traceable

class CustomHybridParentRetriever(BaseRetriever):
    vectorstore: Chroma
    docstore: Any
    bm25_retriever: BM25Retriever
    top_k: int = 10
    rrf_k: int = 60
    reranking_k: int = 15
    bm25_fetch_k: int = 18
    vector_fetch_k: int = 36

    @traceable(name="BM25Retriever")
    def _get_bm25_results(self, query):
        self.bm25_retriever.k = self.bm25_fetch_k
        return self.bm25_retriever.invoke(query)

    @traceable(name="vectorstore_retrieval")
    def get_vectorstore_results(self, query):
        return self.vectorstore.similarity_search(query, k=self.vector_fetch_k)

    @traceable(name="CustomHybridParentRetriever.invoke")
    def _get_relevant_documents(self, query):
        
        # 1. Fetch raw candidates from both search engines (get double top_k for filtering)
        if "|||" in query:
            original_query, hyde_answer = query.split("|||", 1)
        else:
            original_query = query
            hyde_answer = query

        vector_results = self.get_vectorstore_results(hyde_answer)
        bm25_results = self._get_bm25_results(original_query)

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
        seen_acts = set()
        diverse_ids = []

        # First pass: take the highest-RRF chunk from each unique act
        for pid in sorted_parent_ids:
            doc_list = self.docstore.mget([pid])
            doc = doc_list[0] if doc_list else None
            if doc is None:
                continue
            act = doc.metadata.get("act_name")
            if act not in seen_acts:
                diverse_ids.append(pid)
                seen_acts.add(act)

        # Second pass: fill remaining reranking_k slots by pure RRF score
        for pid in sorted_parent_ids:
            if len(diverse_ids) >= self.reranking_k:
                break
            if pid not in diverse_ids:
                diverse_ids.append(pid)

        target_ids = diverse_ids[:self.reranking_k]
        fetched_docs = self.docstore.mget(target_ids)
        final_parent_docs = [doc for doc in fetched_docs if doc is not None]

        if not final_parent_docs:
            return []

        reranked = rerank_with_cohere(original_query, final_parent_docs, top_k=self.reranking_k)
        seen_acts = {}
        diverse_final = []
        MAX_PER_ACT = 3

        for doc in reranked:
            act = doc.metadata.get("act_name")
            count = seen_acts.get(act, 0)
            if count < MAX_PER_ACT:
                diverse_final.append(doc)
                seen_acts[act] = count + 1
            if len(diverse_final) >= self.top_k:
                break

        return diverse_final
