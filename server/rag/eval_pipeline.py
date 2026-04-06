from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from rag_pipeline import RAGPipeline
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

def run_eval():
    pipeline = RAGPipeline()
    
    # We create some sample docs if the index doesn't exist
    if not pipeline.load_index():
        docs = [
            Document(page_content="The primary benefit of RAG is grounding LLM responses in factual, retrieved data, which reduces hallucinations.", metadata={"source": "A1"}),
            Document(page_content="Gradio is a tool to create UIs for machine learning models quickly.", metadata={"source": "A2"}),
            Document(page_content="Evaluation of RAG systems typically uses metrics like Faithfulness and Answer Relevancy.", metadata={"source": "A3"})
        ]
        pipeline.build_index(docs)

    # Define test set
    test_questions = [
        "What is the primary benefit of RAG?",
        "What are typical metrics for evaluating a RAG system?"
    ]
    
    # The expected answers aren't strictly required for Faithfulness and Answer Relevancy, 
    # but we can provide 'reference' if we wanted to use Answer Correctness.
    # We will just collect questions, answers, and contexts for Ragas.
    
    data = {
        "question": [],
        "answer": [],
        "contexts": []
    }
    
    print("Generating answers for evaluation dataset...")
    for q in test_questions:
        response, docs = pipeline.query(q)
        data["question"].append(q)
        data["answer"].append(response.answer)
        data["contexts"].append([d.page_content for d in docs])
        
    dataset = Dataset.from_dict(data)
    
    print("Running Ragas evaluation...")
    # Ragas evaluate needs OpenAI set in environment
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
        ],
    )
    
    print("\n--- Evaluation Results ---")
    print(result)

if __name__ == "__main__":
    run_eval()
