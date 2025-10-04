#!/usr/bin/env python3
"""Simple test runner for Local Agent Studio tests."""

import sys
import os
from pathlib import Path

# Add the server directory to Python path
server_dir = Path(__file__).parent
sys.path.insert(0, str(server_dir))

# Now run a simple test
def test_imports():
    """Test that all core modules can be imported."""
    try:
        from app.core.state import ExecutionState, Task, TaskStatus
        from app.core.orchestrator import MainOrchestrator
        from app.core.base_nodes import BaseLangGraphNode
        from app.config.models import SystemConfig
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_execution_state():
    """Test ExecutionState basic functionality."""
    try:
        from app.core.state import ExecutionState, Task
        
        # Test initialization
        state = ExecutionState(user_request="Test request")
        assert state.user_request == "Test request"
        assert state.current_step == 0
        print("✓ ExecutionState initialization works")
        
        # Test task management
        task = Task(name="test_task")
        state.add_task(task)
        assert len(state.tasks) == 1
        assert state.get_task_by_id(task.id) == task
        print("✓ Task management works")
        
        # Test step increment
        state.increment_step()
        assert state.current_step == 1
        print("✓ Step increment works")
        
        return True
    except Exception as e:
        print(f"✗ ExecutionState test failed: {e}")
        return False

def test_orchestrator():
    """Test MainOrchestrator basic functionality."""
    try:
        from app.core.orchestrator import MainOrchestrator, WorkflowMatcher
        from app.core.state import ExecutionState
        from app.config.models import SystemConfig, WorkflowConfig
        
        # Test WorkflowMatcher
        workflows = {
            "test_workflow": WorkflowConfig(
                name="test_workflow",
                description="Test workflow",
                triggers=["test", "question"],
                enabled=True
            )
        }
        matcher = WorkflowMatcher(workflows)
        
        workflow, confidence = matcher.match_workflow("This is a test question")
        assert workflow == "test_workflow"
        assert confidence > 0
        print("✓ WorkflowMatcher works")
        
        # Test MainOrchestrator initialization
        config = SystemConfig()
        orchestrator = MainOrchestrator(config)
        assert orchestrator.name == "MainOrchestrator"
        print("✓ MainOrchestrator initialization works")
        
        return True
    except Exception as e:
        print(f"✗ Orchestrator test failed: {e}")
        return False

def test_base_nodes():
    """Test base node functionality."""
    try:
        from app.core.base_nodes import BaseLangGraphNode
        from app.core.state import ExecutionState
        
        class TestNode(BaseLangGraphNode):
            def execute(self, state):
                state.context["test"] = "success"
                return state
        
        node = TestNode("TestNode")
        state = ExecutionState()
        
        result_state = node(state)
        assert result_state.context["test"] == "success"
        assert result_state.current_step == 1
        print("✓ BaseLangGraphNode works")
        
        return True
    except Exception as e:
        print(f"✗ Base nodes test failed: {e}")
        return False

if __name__ == "__main__":
    print("Running Local Agent Studio Core Tests...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_execution_state,
        test_orchestrator,
        test_base_nodes
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)