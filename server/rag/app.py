import gradio as gr
from rag_pipeline import RAGPipeline
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Initialize pipeline
pipeline = RAGPipeline()

# Try loading from disk, otherwise build with defaults
if not pipeline.load_index():
    print("No existing index found. Building with sample documents...")
    docs = [
        Document(page_content="LangChain is a framework for developing applications powered by language models.", metadata={"source": "doc1"}),
        Document(page_content="RAG stands for Retrieval-Augmented Generation. It helps to ground the model on specific facts.", metadata={"source": "doc2"}),
        Document(page_content="FAISS is a library for efficient similarity search and clustering of dense vectors.", metadata={"source": "doc3"}),
        Document(page_content="Instructor is a library for structured extraction with LLMs.", metadata={"source": "doc4"})
    ]
    pipeline.build_index(docs)

def query_rag(question):
    try:
        response, docs = pipeline.query(question)
        
        answer = response.answer
        citations = "\n".join([f"- {c}" for c in response.citations])
        contexts = "\n\n".join([f"**Document {i+1}:** {d.page_content} (Source: {d.metadata.get('source', 'N/A')})" for i, d in enumerate(docs)])
        
        return answer, citations, contexts
    except Exception as e:
        return f"Error: {str(e)}", "", ""

demo = gr.Interface(
    fn=query_rag,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about LangChain, RAG, etc..."),
    outputs=[
        gr.Textbox(label="Generated Answer"),
        gr.Textbox(label="Strict Citations extracted by Instructor"),
        gr.Markdown(label="Retrieved Contexts (from FAISS + BM25 -> Cohere Rerank)")
    ],
    title="Production LangChain RAG System",
    description="A hybrid search (FAISS + BM25) RAG system using Cohere for reranking and Instructor for strict LLM citations.",
)

if __name__ == "__main__":
    demo.launch()
