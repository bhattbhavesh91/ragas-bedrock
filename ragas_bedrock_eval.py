"""
RAGAS Evaluation Script for RAG Applications using AWS Bedrock
Evaluates: Faithfulness, Answer Relevancy, Context Precision, Context Recall, and more
"""

import boto3
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_entity_recall,
    answer_similarity,
    answer_correctness
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_aws import ChatBedrock, BedrockEmbeddings
from datasets import Dataset
import pandas as pd
from typing import List, Dict
import os


class RAGASBedrockEvaluator:
    """
    Evaluator class for RAG applications using RAGAS framework with AWS Bedrock
    """
    
    def __init__(
        self,
        aws_region: str = "us-east-1",
        llm_model_id: str = "meta.llama3-70b-instruct-v1:0",
        embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    ):
        """
        Initialize the evaluator with AWS Bedrock models
        
        Args:
            aws_region: AWS region for Bedrock
            llm_model_id: Bedrock model ID for LLM (e.g., Llama, Mistral)
            embedding_model_id: Bedrock model ID for embeddings
        """
        self.aws_region = aws_region
        self.llm_model_id = llm_model_id
        self.embedding_model_id = embedding_model_id
        
        # Initialize Bedrock client
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=aws_region
        )
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize Embeddings
        self.embeddings = self._initialize_embeddings()
        
    def _initialize_llm(self):
        """Initialize the Bedrock LLM with RAGAS wrapper"""
        bedrock_llm = ChatBedrock(
            client=self.bedrock_runtime,
            model_id=self.llm_model_id,
            model_kwargs={
                "temperature": 0.1,
                "max_gen_len": 2048,
            }
        )
        return LangchainLLMWrapper(bedrock_llm)
    
    def _initialize_embeddings(self):
        """Initialize Bedrock embeddings with RAGAS wrapper"""
        bedrock_embeddings = BedrockEmbeddings(
            client=self.bedrock_runtime,
            model_id=self.embedding_model_id
        )
        return LangchainEmbeddingsWrapper(bedrock_embeddings)
    
    def prepare_evaluation_dataset(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str]
    ) -> Dataset:
        """
        Prepare dataset for RAGAS evaluation
        
        Args:
            questions: List of user questions
            answers: List of RAG-generated answers
            contexts: List of retrieved context chunks (list of lists)
            ground_truths: List of reference/expected answers
            
        Returns:
            Dataset object for RAGAS evaluation
        """
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        
        return Dataset.from_dict(data)
    
    def evaluate_rag(
        self,
        dataset: Dataset,
        metrics: List = None
    ) -> Dict:
        """
        Evaluate RAG application using specified metrics
        
        Args:
            dataset: Prepared evaluation dataset
            metrics: List of RAGAS metrics to evaluate (None = all metrics)
            
        Returns:
            Dictionary containing evaluation results
        """
        # Default metrics if none specified
        if metrics is None:
            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                context_entity_recall,
                answer_similarity,
                answer_correctness
            ]
        
        # Run evaluation
        print("Starting RAGAS evaluation...")
        print(f"Using LLM: {self.llm_model_id}")
        print(f"Using Embeddings: {self.embedding_model_id}")
        print(f"Evaluating {len(dataset)} samples...")
        
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self.llm,
            embeddings=self.embeddings
        )
        
        return result
    
    def display_results(self, result: Dict):
        """
        Display evaluation results in a readable format
        
        Args:
            result: Evaluation result dictionary from RAGAS
        """
        print("\n" + "="*60)
        print("RAGAS EVALUATION RESULTS")
        print("="*60)
        
        # Display overall scores
        print("\nOverall Metrics:")
        print("-" * 60)
        for metric, score in result.items():
            if not isinstance(score, pd.DataFrame):
                print(f"{metric:.<40} {score:.4f}")
        
        # Convert to DataFrame for detailed view
        df = result.to_pandas()
        
        print("\n" + "="*60)
        print("Detailed Sample-wise Results:")
        print("="*60)
        print(df.to_string())
        
        return df
    
    def save_results(self, result: Dict, output_path: str = "ragas_results.csv"):
        """
        Save evaluation results to CSV
        
        Args:
            result: Evaluation result dictionary
            output_path: Path to save CSV file
        """
        df = result.to_pandas()
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize evaluator with AWS Bedrock
    # You can use different open-source models:
    # - meta.llama3-70b-instruct-v1:0
    # - meta.llama3-8b-instruct-v1:0
    # - mistral.mistral-7b-instruct-v0:2
    # - mistral.mixtral-8x7b-instruct-v0:1
    
    evaluator = RAGASBedrockEvaluator(
        aws_region="us-east-1",
        llm_model_id="meta.llama3-70b-instruct-v1:0",
        embedding_model_id="amazon.titan-embed-text-v2:0"
    )
    
    # Example data - replace with your actual RAG outputs
    questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is photosynthesis?"
    ]
    
    answers = [
        "The capital of France is Paris, which is located in the north-central part of the country.",
        "Romeo and Juliet was written by William Shakespeare in the late 16th century.",
        "Photosynthesis is the process by which plants convert light energy into chemical energy."
    ]
    
    contexts = [
        [
            "Paris is the capital and most populous city of France.",
            "Paris is located in north-central France on the Seine River."
        ],
        [
            "William Shakespeare was an English playwright and poet.",
            "Romeo and Juliet is a tragedy written by Shakespeare early in his career, around 1594-1596."
        ],
        [
            "Photosynthesis is a process used by plants to convert light into energy.",
            "During photosynthesis, plants take in carbon dioxide and water, using sunlight to produce glucose and oxygen."
        ]
    ]
    
    ground_truths = [
        "Paris is the capital of France.",
        "William Shakespeare wrote Romeo and Juliet.",
        "Photosynthesis is the process plants use to convert light energy into chemical energy stored in glucose."
    ]
    
    # Prepare dataset
    dataset = evaluator.prepare_evaluation_dataset(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths
    )
    
    # Run evaluation
    results = evaluator.evaluate_rag(dataset)
    
    # Display results
    df_results = evaluator.display_results(results)
    
    # Save results
    evaluator.save_results(results, "ragas_evaluation_results.csv")
    
    print("\n" + "="*60)
    print("METRIC DESCRIPTIONS:")
    print("="*60)
    print("""
    1. Faithfulness (0-1): Measures if the answer is grounded in the context
       Higher = More faithful to provided context
    
    2. Answer Relevancy (0-1): Measures how relevant the answer is to the question
       Higher = More relevant answer
    
    3. Context Precision (0-1): Measures if relevant context is ranked higher
       Higher = Better context retrieval ranking
    
    4. Context Recall (0-1): Measures if all relevant information is retrieved
       Higher = More complete context retrieval
    
    5. Context Entity Recall (0-1): Measures recall of entities from ground truth
       Higher = More entities from answer found in context
    
    6. Answer Similarity (0-1): Semantic similarity between answer and ground truth
       Higher = More similar to expected answer
    
    7. Answer Correctness (0-1): Weighted combination of similarity and factuality
       Higher = More correct overall
    """)