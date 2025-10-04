# Predefined Workflows Implementation

This module implements the predefined workflows system for Local Agent Studio, providing a framework for executing structured, multi-step workflows with confidence-based matching and comprehensive testing.

## Overview

The workflow system consists of three main components:

1. **Workflow Execution Framework** - Core infrastructure for defining and executing workflows
2. **Predefined Workflows** - Four built-in workflows for common tasks
3. **Integration Tests** - Comprehensive test suite for workflow functionality

## Architecture

### Workflow Execution Framework

#### WorkflowMatcher (`workflow_matcher.py`)
- Matches user requests to predefined workflows using confidence scoring
- Uses multiple matching strategies:
  - Exact trigger matching (40% weight)
  - Pattern-based matching (30% weight)
  - Semantic similarity (20% weight)
  - Context relevance (10% weight)
- Supports context-aware matching (e.g., file uploads, image context)
- Provides workflow suggestions with configurable thresholds

#### WorkflowExecutor (`workflow_executor.py`)
- Executes workflows step-by-step with proper error handling
- Manages workflow execution context and state
- Checks success criteria at step and workflow levels
- Integrates with context manager, RAG workflow, and memory manager
- Supports multiple node types:
  - `context_manager` - Document retrieval and processing
  - `llm_generator` - Answer generation
  - `evaluator` - Answer verification
  - `analyzer` - Content analysis
  - `summarizer` - Document summarization
  - `fact_extractor` - Fact extraction
  - `comparator` - Document comparison
  - `synthesizer` - Multi-document synthesis
  - `image_processor` - Image preprocessing
  - `ocr_engine` - OCR text extraction
  - `text_processor` - Text cleaning and processing
  - `qa_engine` - Question answering

#### WorkflowModels (`workflow_models.py`)
- `WorkflowStepStatus` - Enum for step status (pending, in_progress, completed, failed, skipped)
- `WorkflowStepResult` - Result of executing a single step
- `WorkflowExecutionContext` - Context for workflow execution with tracking

## Predefined Workflows

### 1. RAG QA Workflow (`rag_qa_workflow.py`)
**Purpose:** Answer questions based on uploaded documents using retrieval-augmented generation

**Pipeline:**
1. Context retrieval - Retrieve relevant document chunks
2. Answer generation - Generate answer with LLM
3. Answer verification - Verify quality and grounding

**Triggers:**
- "question about document"
- "what does the document say"
- "find information in"
- "search for"
- "explain from document"

**Configuration:**
- Retrieval k: 10 chunks
- Uses MMR for diversity
- Includes answer verification
- Provides citations with sources

### 2. Summarize & Extract Workflow (`summarize_extract_workflow.py`)
**Purpose:** Summarize documents and extract key facts, entities, and insights

**Pipeline:**
1. Document chunking - Process entire document
2. Content analysis - Analyze for key information
3. Summary generation - Generate comprehensive summary
4. Fact extraction - Extract and structure key facts

**Triggers:**
- "summarize document"
- "extract key points"
- "what are the main ideas"
- "give me a summary"
- "extract facts from"

**Features:**
- Configurable summary length (short, medium, long)
- Entity extraction
- Theme identification
- Fact categorization with confidence scores

### 3. Compare & Synthesize Workflow (`compare_synthesize_workflow.py`)
**Purpose:** Compare multiple documents and synthesize insights across sources

**Pipeline:**
1. Multi-doc processing - Process and index all documents
2. Comparative analysis - Identify similarities and differences
3. Synthesis generation - Synthesize insights across documents

**Triggers:**
- "compare documents"
- "find differences between"
- "synthesize information from"
- "what do these documents say about"
- "cross-reference"

**Features:**
- Requires at least 2 documents
- Identifies common themes
- Highlights unique topics per document
- Detects potential conflicts
- Provides cross-document synthesis

### 4. Image OCR → QA Workflow (`image_ocr_qa_workflow.py`)
**Purpose:** Extract text from images using OCR and answer questions about the content

**Pipeline:**
1. Image preprocessing - Prepare image for OCR
2. OCR extraction - Extract text using OCR
3. Text processing - Clean and structure extracted text
4. QA processing - Answer questions about extracted text

**Triggers:**
- "read text from image"
- "what does this image say"
- "extract text from"
- "ocr this image"
- "question about image text"

**Features:**
- Image quality enhancement
- OCR confidence scoring
- Text structure preservation
- Question answering on extracted text

## Integration Tests

The test suite (`test_workflow_integration.py`) includes:

### WorkflowMatcher Tests
- ✓ Match RAG QA workflow
- ✓ Match summarize workflow
- ✓ Match compare workflow
- ✓ No match with low confidence
- ✓ Disabled workflows not matched
- ✓ Context-aware matching
- ✓ Get workflow suggestions

### Predefined Workflow Tests
- ✓ RAG QA workflow execution
- ✓ RAG QA workflow failure handling
- ✓ Summarize & extract execution
- ✓ Summarize & extract with no chunks
- ✓ Compare & synthesize execution
- ✓ Compare & synthesize with insufficient docs
- ✓ Image OCR QA execution
- ✓ Image OCR QA without image

### WorkflowExecutor Tests
- ✓ Workflow executor initialization
- ✓ Workflow execution success
- ✓ Workflow execution step failure
- ✓ Workflow success criteria checking

**Test Results:** 19/19 tests passing ✓

## Usage Examples

### Using WorkflowMatcher

```python
from app.core.workflows import WorkflowMatcher
from app.config.models import WorkflowConfig

# Create workflow configs
workflows = {
    "rag_qa": WorkflowConfig(
        name="RAG QA",
        description="Answer questions",
        triggers=["question", "what"],
        confidence_threshold=0.7,
        enabled=True
    )
}

# Initialize matcher
matcher = WorkflowMatcher(workflows)

# Match user request
workflow_name, confidence = matcher.match_workflow(
    "What does the document say about climate change?"
)

print(f"Matched: {workflow_name} (confidence: {confidence:.2f})")
```

### Using WorkflowExecutor

```python
from app.core.workflows import WorkflowExecutor
from app.config.models import WorkflowConfig, WorkflowStep
from app.core.state import ExecutionState

# Define workflow
workflow_config = WorkflowConfig(
    name="My Workflow",
    description="Custom workflow",
    steps=[
        WorkflowStep(
            name="retrieve",
            description="Retrieve context",
            node_type="context_manager",
            parameters={"retrieval_k": 10},
            success_criteria=["Retrieved at least 3 chunks"]
        )
    ],
    success_criteria=["Workflow completed"],
    confidence_threshold=0.7,
    enabled=True
)

# Execute workflow
executor = WorkflowExecutor(workflow_config, context_manager=context_mgr)
state = ExecutionState(user_request="Test request")

result = executor.execute(
    user_request="Test request",
    execution_state=state
)

print(f"Complete: {result.is_complete}")
print(f"Steps: {len(result.step_results)}")
```

### Using Predefined Workflows

```python
from app.core.workflows.predefined import RAGQAWorkflow
from app.core.state import ExecutionState
from app.core.workflows.workflow_models import WorkflowExecutionContext

# Initialize workflow
workflow = RAGQAWorkflow(
    context_manager=context_mgr,
    openai_api_key="your-key",
    retrieval_k=10
)

# Execute
state = ExecutionState(user_request="What is the answer?")
context = WorkflowExecutionContext(
    workflow_name="RAG QA",
    workflow_description="Test",
    user_request="What is the answer?"
)

result = workflow.execute(
    user_request="What is the answer?",
    execution_state=state,
    workflow_context=context
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
```

## Configuration

Workflows are configured in `config/workflows.yaml`:

```yaml
workflows:
  rag_qa:
    name: "RAG Question Answering"
    description: "Answer questions based on documents"
    confidence_threshold: 0.8
    enabled: true
    triggers:
      - "question about document"
      - "what does the document say"
    steps:
      - name: "context_retrieval"
        node_type: "context_manager"
        parameters:
          retrieval_k: 10
          use_mmr: true
        success_criteria:
          - "Retrieved at least 3 relevant chunks"
```

## Requirements Satisfied

This implementation satisfies the following requirements from the design document:

- **Requirement 1.1:** Main orchestrator routes between predefined workflows and planner-driven execution
- **Requirement 8.1:** All workflow parameters defined in configuration files
- **Requirement 8.2:** Predefined workflows specified in workflows.yaml
- **Requirement 3.1, 3.2, 3.3:** File processing workflows for Office, PDF, and image documents

## Future Enhancements

1. **LLM Integration:** Replace placeholder implementations with actual LLM calls for:
   - Content analysis
   - Summarization
   - Fact extraction
   - Document comparison

2. **Advanced OCR:** Integrate Tesseract or similar OCR engines for image processing

3. **Workflow Composition:** Support for composing workflows from smaller reusable components

4. **Dynamic Workflow Creation:** Allow users to define custom workflows at runtime

5. **Workflow Optimization:** Learn from execution patterns to improve matching and performance

## Testing

Run the test suite:

```bash
cd server
python -m pytest tests/test_workflow_integration.py -v
```

All 19 tests should pass, covering:
- Workflow matching with various confidence levels
- Predefined workflow execution
- Error handling and edge cases
- Success criteria validation
