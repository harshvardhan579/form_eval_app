import os
from rag_pipeline import RAGPipeline
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

def main():
    # Sample documents
    docs = [
        Document(page_content="LangChain is a framework for developing applications powered by language models.", metadata={"source": "doc1"}),
        Document(page_content="RAG stands for Retrieval-Augmented Generation. It helps to ground the model on specific facts.", metadata={"source": "doc2"}),
        Document(page_content="FAISS is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size.", metadata={"source": "doc3"}),
        Document(page_content="Cohere provides API access for natural language processing models, including reranking models.", metadata={"source": "doc4"}),
        Document(page_content="Instructor is a Python library that makes it easy to work with structured extraction using OpenAI's Function Calling API.", metadata={"source": "doc5"})
    ]

    pipeline = RAGPipeline()
    
    # Try loading from disk, else build from scratch
    if not pipeline.load_index():
        pipeline.build_index(docs)

    question = "What does Instructor do and what does RAG stand for?"
    
    response, retrieved_docs = pipeline.query(question)
    
    print("\n--- Final Structured Response ---")
    print(f"Answer: {response.answer}")
    print("\nCitations:")
    for i, citation in enumerate(response.citations):
        print(f"[{i+1}] {citation}")
        
    print("\n--- Context Documents Used ---")
    for i, doc in enumerate(retrieved_docs):
        print(f"Doc {i+1}: {doc.page_content} (Source: {doc.metadata.get('source', 'Unknown')})")

if __name__ == "__main__":
    main()
