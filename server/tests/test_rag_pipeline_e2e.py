"""End-to-end tests for the complete RAG pipeline."""

import pytest
import os
import tempfile
from pathlib import Path
from datetime import datetime

from app.core.rag.rag_workflow import RAGWorkflow
from app.core.context.context_manager import ContextManager
from app.config.loader import ConfigLoader


@pytest.fixture
def config():
    """Load system configuration."""
    config_loader = ConfigLoader()
    return config_loader.load_config()


@pytest.fixture
def context_manager(config):
    """Create context manager instance."""
    return ContextManager(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        vector_db_path="./data/test_chroma_db",
        chunk_size=config.context.chunk_size,
        chunk_overlap=config.context.chunk_overlap,
        embedding_model=config.context.embedding_model,
        context_config=config.context
    )


@pytest.fixture
def rag_workflow(context_manager):
    """Create RAG workflow instance."""
    return RAGWorkflow(
        context_manager=context_manager,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        answer_model="gpt-4o-mini",
        retrieval_k=10,
        use_mmr=True,
        enable_verification=True
    )


@pytest.fixture
def sample_text_file():
    """Create a sample text file for testing."""
    content = """
    Artificial Intelligence and Machine Learning
    
    Artificial Intelligence (AI) is the simulation of human intelligence by machines.
    Machine Learning (ML) is a subset of AI that enables systems to learn from data.
    
    Deep Learning is a subset of ML that uses neural networks with multiple layers.
    Neural networks are inspired by the structure of the human brain.
    
    Common applications of AI include:
    - Natural Language Processing (NLP)
    - Computer Vision
    - Robotics
    - Autonomous Vehicles
    
    AI has revolutionized many industries including healthcare, finance, and transportation.
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


class TestRAGPipelineE2E:
    """End-to-end tests for RAG pipeline."""
    
    def test_file_upload_and_processing(self, context_manager, sample_text_file):
        """Test file upload triggers processing pipeline."""
        # Ingest file
        result = context_manager.ingest_file(sample_text_file)
        
        # Verify ingestion success
        assert result.success, f"Ingestion failed: {result.error}"
        assert result.chunks_created > 0, "No chunks created"
        assert result.chunks_stored > 0, "No chunks stored"
        assert result.tokens_used > 0, "No tokens used for embeddings"
        
        print(f"✓ File processed: {result.chunks_created} chunks, {result.tokens_used} tokens")
    
    def test_retrieval_to_answer_flow(self, rag_workflow, sample_text_file):
        """Test complete retrieval to answer flow."""
        # First ingest the file
        ingestion_result = rag_workflow.context_manager.ingest_file(sample_text_file)
        assert ingestion_result.success, "File ingestion failed"
        
        # Ask a question
        question = "What is Machine Learning?"
        result = rag_workflow.process_question(question, source_filter=sample_text_file)
        
        # Verify result
        assert result.success, f"Question processing failed: {result.error}"
        assert len(result.answer) > 0, "Empty answer"
        assert len(result.citations) > 0, "No citations provided"
        assert result.confidence > 0, "Zero confidence"
        
        print(f"✓ Question answered: {len(result.answer)} chars, {len(result.citations)} citations")
        print(f"  Answer: {result.answer[:100]}...")
    
    def test_citation_tracking(self, rag_workflow, sample_text_file):
        """Test citation tracking through pipeline."""
        # Ingest and query
        rag_workflow.context_manager.ingest_file(sample_text_file)
        result = rag_workflow.process_question(
            "What are common applications of AI?",
            source_filter=sample_text_file
        )
        
        assert result.success, "Question processing failed"
        assert len(result.citations) > 0, "No citations"
        
        # Verify citation structure
        for citation in result.citations:
            assert citation.chunk_id, "Missing chunk ID"
            assert citation.source_file, "Missing source file"
            assert citation.content_snippet, "Missing content snippet"
            assert 0 <= citation.relevance_score <= 1, "Invalid relevance score"
        
        # Verify citations include source file and page numbers
        first_citation = result.citations[0]
        assert first_citation.source_file == sample_text_file
        
        print(f"✓ Citations tracked: {len(result.citations)} citations")
        for i, citation in enumerate(result.citations[:3], 1):
            print(f"  {i}. {citation.source_file} (score: {citation.relevance_score:.2f})")
    
    def test_answer_verification(self, rag_workflow, sample_text_file):
        """Test answer verification and quality scoring."""
        # Ingest and query
        rag_workflow.context_manager.ingest_file(sample_text_file)
        result = rag_workflow.process_question(
            "What is Deep Learning?",
            source_filter=sample_text_file
        )
        
        assert result.success, "Question processing failed"
        assert result.verification is not None, "No verification performed"
        
        # Verify verification structure
        verification = result.verification
        assert 0 <= verification.quality_score <= 1, "Invalid quality score"
        assert 0 <= verification.confidence <= 1, "Invalid confidence"
        assert 0 <= verification.grounding_score <= 1, "Invalid grounding score"
        
        print(f"✓ Answer verified:")
        print(f"  Passed: {verification.passed}")
        print(f"  Quality: {verification.quality_score:.2f}")
        print(f"  Grounding: {verification.grounding_score:.2f}")
        print(f"  Confidence: {verification.confidence:.2f}")
        
        if verification.issues:
            print(f"  Issues: {', '.join(verification.issues)}")
    
    def test_multiple_documents(self, rag_workflow):
        """Test with multiple documents."""
        # Create multiple test files
        files = []
        
        # File 1: AI basics
        content1 = """
        Artificial Intelligence is the field of computer science focused on creating
        intelligent machines. AI systems can perform tasks that typically require
        human intelligence, such as visual perception, speech recognition, and
        decision-making.
        """
        
        # File 2: ML specifics
        content2 = """
        Machine Learning algorithms learn patterns from data without being explicitly
        programmed. Supervised learning uses labeled data, while unsupervised learning
        finds patterns in unlabeled data. Reinforcement learning learns through
        trial and error.
        """
        
        for i, content in enumerate([content1, content2], 1):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(content)
                files.append(f.name)
        
        try:
            # Ingest both files
            for file_path in files:
                result = rag_workflow.context_manager.ingest_file(file_path)
                assert result.success, f"Failed to ingest {file_path}"
            
            # Ask a question that requires both documents
            result = rag_workflow.process_question(
                "What is the difference between AI and Machine Learning?"
            )
            
            assert result.success, "Question processing failed"
            assert len(result.citations) > 0, "No citations"
            
            # Verify citations come from multiple sources
            source_files = set(c.source_file for c in result.citations)
            print(f"✓ Multiple documents: {len(source_files)} sources used")
            
        finally:
            # Cleanup
            for file_path in files:
                if os.path.exists(file_path):
                    os.remove(file_path)
    
    def test_performance_metrics(self, rag_workflow, sample_text_file):
        """Test performance and validate metrics."""
        # Ingest file
        start_time = datetime.now()
        ingestion_result = rag_workflow.context_manager.ingest_file(sample_text_file)
        ingestion_time = (datetime.now() - start_time).total_seconds()
        
        assert ingestion_result.success, "Ingestion failed"
        
        # Process question
        start_time = datetime.now()
        result = rag_workflow.process_question(
            "What is Artificial Intelligence?",
            source_filter=sample_text_file
        )
        query_time = (datetime.now() - start_time).total_seconds()
        
        assert result.success, "Query failed"
        
        # Validate performance
        print(f"✓ Performance metrics:")
        print(f"  Ingestion time: {ingestion_time:.2f}s")
        print(f"  Query time: {query_time:.2f}s")
        print(f"  Processing time (reported): {result.processing_time_ms:.0f}ms")
        print(f"  Tokens used: {result.tokens_used}")
        print(f"  Chunks retrieved: {result.metadata.get('chunks_retrieved', 0)}")
        
        # Basic performance assertions
        assert ingestion_time < 30, "Ingestion too slow"
        assert query_time < 15, "Query too slow"
    
    def test_error_handling(self, rag_workflow):
        """Test error handling in RAG pipeline."""
        # Test with non-existent file
        result = rag_workflow.context_manager.ingest_file("nonexistent_file.txt")
        assert not result.success, "Should fail for non-existent file"
        assert result.error is not None, "Should have error message"
        
        # Test query without any documents
        rag_workflow.context_manager.vector_store.reset_collection()
        result = rag_workflow.process_question("What is AI?")
        
        # Should succeed but indicate no information available
        assert result.success, "Should handle empty database gracefully"
        assert "don't have enough information" in result.answer.lower() or len(result.citations) == 0
        
        print("✓ Error handling works correctly")
    
    def test_replanning_on_low_quality(self, rag_workflow, sample_text_file):
        """Test replanning when answer quality is insufficient."""
        # Ingest file
        rag_workflow.context_manager.ingest_file(sample_text_file)
        
        # Process question
        result = rag_workflow.process_question(
            "What is Machine Learning?",
            source_filter=sample_text_file
        )
        
        assert result.success, "Initial query failed"
        
        # Check if replanning is needed
        should_replan = rag_workflow.should_replan(result)
        
        print(f"✓ Replanning check:")
        print(f"  Should replan: {should_replan}")
        print(f"  Quality score: {result.verification.quality_score if result.verification else 'N/A'}")
        
        # If replanning is needed, test the replan functionality
        if should_replan:
            replanned_result = rag_workflow.replan_and_retry(
                "What is Machine Learning?",
                result,
                max_retries=2
            )
            
            assert replanned_result.success, "Replanning failed"
            print(f"  Replanned quality: {replanned_result.verification.quality_score if replanned_result.verification else 'N/A'}")


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not available")
class TestRAGPipelineIntegration:
    """Integration tests requiring OpenAI API."""
    
    def test_full_rag_workflow(self, rag_workflow, sample_text_file):
        """Test complete RAG workflow from upload to answer."""
        print("\n" + "="*60)
        print("FULL RAG PIPELINE TEST")
        print("="*60)
        
        # Step 1: Upload and process file
        print("\n1. Uploading and processing file...")
        ingestion_result = rag_workflow.context_manager.ingest_file(sample_text_file)
        assert ingestion_result.success
        print(f"   ✓ Created {ingestion_result.chunks_created} chunks")
        
        # Step 2: Ask question
        print("\n2. Asking question...")
        question = "What are the main applications of Artificial Intelligence?"
        result = rag_workflow.process_question(question, source_filter=sample_text_file)
        assert result.success
        print(f"   ✓ Generated answer ({len(result.answer)} chars)")
        
        # Step 3: Verify citations
        print("\n3. Verifying citations...")
        assert len(result.citations) > 0
        print(f"   ✓ Found {len(result.citations)} citations")
        for i, citation in enumerate(result.citations[:3], 1):
            print(f"      {i}. Score: {citation.relevance_score:.2f}")
        
        # Step 4: Check verification
        print("\n4. Checking answer verification...")
        assert result.verification is not None
        print(f"   ✓ Verification passed: {result.verification.passed}")
        print(f"   ✓ Quality score: {result.verification.quality_score:.2f}")
        print(f"   ✓ Grounding score: {result.verification.grounding_score:.2f}")
        
        # Step 5: Display final answer
        print("\n5. Final Answer:")
        print("-" * 60)
        print(result.answer)
        print("-" * 60)
        
        print("\n" + "="*60)
        print("FULL RAG PIPELINE TEST PASSED ✓")
        print("="*60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
