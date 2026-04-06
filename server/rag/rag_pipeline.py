import os
import pickle
from typing import List, Tuple
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

class AnswerWithCitations(BaseModel):
    answer: str = Field(description="The answer to the user's question, based entirely on the provided contexts.")
    citations: List[str] = Field(description="Exact quotes or identifiers from the contexts that support the answer.")

class RAGPipeline:
    def __init__(self, index_dir="faiss_index", bm25_path="bm25_retriever.pkl"):
        self.index_dir = index_dir
        self.bm25_path = bm25_path
        
        # Load API keys and instantiate clients
        self.embeddings = OpenAIEmbeddings()
        # Initialize instructor-wrapped OpenAI client
        self.llm_client = instructor.from_openai(OpenAI())
        
        self.vectorstore = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.reranker_retriever = None

    def build_index(self, documents: List[Document]):
        """Builds the FAISS and BM25 indices from scratch and saves them."""
        print("Building FAISS index...")
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.vectorstore.save_local(self.index_dir)
        
        print("Building BM25 index...")
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        with open(self.bm25_path, 'wb') as f:
            pickle.dump(self.bm25_retriever, f)
            
        self._setup_retrievers()

    def load_index(self) -> bool:
        """Loads indices from disk. Returns True if successful, False otherwise."""
        if os.path.exists(self.index_dir) and os.path.exists(self.bm25_path):
            print("Loading FAISS index from disk...")
            self.vectorstore = FAISS.load_local(self.index_dir, self.embeddings, allow_dangerous_deserialization=True)
            
            print("Loading BM25 index from disk...")
            with open(self.bm25_path, 'rb') as f:
                self.bm25_retriever = pickle.load(f)
                
            self._setup_retrievers()
            return True
        return False

    def _setup_retrievers(self):
        """Sets up the ensemble and reranking retrievers."""
        faiss_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        self.bm25_retriever.k = 5
        
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, self.bm25_retriever],
            weights=[0.5, 0.5]
        )
        
        compressor = CohereRerank(top_n=3)
        self.reranker_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.ensemble_retriever
        )

    def query(self, question: str) -> Tuple[AnswerWithCitations, List[Document]]:
        """Runs the query through the pipeline and returns a structured answer and contexts."""
        if not self.reranker_retriever:
            raise ValueError("Retrievers not initialized. Call build_index or load_index first.")
            
        print(f"Retrieving and reranking documents for: '{question}'")
        docs = self.reranker_retriever.invoke(question)
        
        context_str = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        
        prompt = f"""You are a helpful assistant. Answer the question using ONLY the provided contexts.
If the contexts do not contain the answer, say "I don't know based on the provided context."

Contexts:
{context_str}

Question: {question}
"""
        print("Calling LLM with Instructor for structured output...")
        response = self.llm_client.chat.completions.create(
            model="gpt-4o",
            response_model=AnswerWithCitations,
            messages=[
                {"role": "system", "content": "You are a precise QA assistant that strictly uses the provided context and cites your sources."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response, docs
