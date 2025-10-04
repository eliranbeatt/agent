"""
System health check script.

Validates that all components are properly configured and operational.
"""

import os
import sys
import logging
import requests
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.loader import ConfigLoader


logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class HealthChecker:
    """System health checker."""
    
    def __init__(self):
        self.checks: List[Tuple[str, bool, str]] = []
        self.config = None
    
    def add_check(self, name: str, passed: bool, message: str = ""):
        """Add a check result."""
        self.checks.append((name, passed, message))
        status = "✓" if passed else "✗"
        level = logging.INFO if passed else logging.ERROR
        logger.log(level, f"{status} {name}: {message}")
    
    def check_python_version(self) -> bool:
        """Check Python version."""
        version = sys.version_info
        required = (3, 11)
        passed = version >= required
        message = f"Python {version.major}.{version.minor}.{version.micro}"
        if not passed:
            message += f" (requires >= {required[0]}.{required[1]})"
        self.add_check("Python Version", passed, message)
        return passed
    

    
    def check_directories(self) -> bool:
        """Check required directories exist."""
        required_dirs = [
            "data",
            "data/vector_db",
            "data/memory",
            "config"
        ]
        
        missing = []
        for directory in required_dirs:
            if not Path(directory).exists():
                missing.append(directory)
        
        passed = len(missing) == 0
        message = "All directories exist" if passed else f"Missing: {', '.join(missing)}"
        self.add_check("Directories", passed, message)
        return passed
    
    def check_configuration(self) -> bool:
        """Check configuration files."""
        try:
            loader = ConfigLoader()
            self.config = loader.load_config()
            
            # Check critical config values
            checks = [
                (self.config.orchestrator.max_iterations > 0, "max_iterations"),
                (self.config.orchestrator.token_budget > 0, "token_budget"),
                (0 <= self.config.orchestrator.workflow_confidence_threshold <= 1, "confidence_threshold"),
            ]
            
            failed = [name for passed, name in checks if not passed]
            
            passed = len(failed) == 0
            message = "Configuration valid" if passed else f"Invalid: {', '.join(failed)}"
            self.add_check("Configuration", passed, message)
            return passed
            
        except Exception as e:
            self.add_check("Configuration", False, str(e))
            return False
    
    def check_dependencies(self) -> bool:
        """Check Python dependencies."""
        required_packages = [
            "fastapi",
            "uvicorn",
            "openai",
            "chromadb",
            "pydantic",
            "yaml"
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        passed = len(missing) == 0
        message = "All dependencies installed" if passed else f"Missing: {', '.join(missing)}"
        self.add_check("Python Dependencies", passed, message)
        return passed
    
    def check_api_server(self, url: str = "http://localhost:8000") -> bool:
        """Check if API server is running."""
        try:
            response = requests.get(f"{url}/healthz", timeout=5)
            passed = response.status_code == 200
            message = f"Server responding at {url}" if passed else f"Server returned {response.status_code}"
            self.add_check("API Server", passed, message)
            return passed
        except requests.exceptions.ConnectionError:
            self.add_check("API Server", False, "Server not running")
            return False
        except Exception as e:
            self.add_check("API Server", False, str(e))
            return False
    
    def check_openai_connection(self) -> bool:
        """Check OpenAI API connection."""
        try:
            import openai
            
            if not self.config:
                self.add_check("OpenAI Connection", False, "Configuration not loaded")
                return False

            api_key = self.config.openai_api_key
            
            if not api_key or api_key == "your-api-key-here":
                self.add_check("OpenAI Connection", False, "API key not configured")
                return False
            
            # Try to create a client (doesn't make API call)
            client = openai.OpenAI(api_key=api_key)
            self.add_check("OpenAI Connection", True, "API key configured")
            return True
            
        except Exception as e:
            self.add_check("OpenAI Connection", False, str(e))
            return False
    
    def check_vector_database(self) -> bool:
        """Check vector database."""
        try:
            from app.core.context.vector_store import VectorStore
            
            if not self.config:
                self.add_check("Vector Database", False, "Configuration not loaded")
                return False
            
            vector_store = VectorStore(persist_directory=self.config.context.vector_db_path)
            # Try a simple search
            results = vector_store.search("test", k=1)
            
            self.add_check("Vector Database", True, "ChromaDB operational")
            return True
            
        except Exception as e:
            self.add_check("Vector Database", False, str(e))
            return False
    
    def run_all_checks(self, check_server: bool = False) -> bool:
        """Run all health checks."""
        logger.info("=" * 60)
        logger.info("Local Agent Studio - System Health Check")
        logger.info("=" * 60)
        logger.info("")
        
        # Run checks
        self.check_python_version()
        self.check_directories()
        self.check_configuration()
        self.check_dependencies()
        self.check_openai_connection()
        self.check_vector_database()
        
        if check_server:
            self.check_api_server()
        
        # Summary
        logger.info("")
        logger.info("=" * 60)
        
        passed = sum(1 for _, p, _ in self.checks if p)
        total = len(self.checks)
        
        if passed == total:
            logger.info(f"✓ All checks passed ({passed}/{total})")
            logger.info("=" * 60)
            logger.info("System is healthy and ready to use!")
            return True
        else:
            logger.error(f"✗ {total - passed} check(s) failed ({passed}/{total} passed)")
            logger.info("=" * 60)
            logger.error("Please fix the issues above before starting the system")
            return False
    
    def get_summary(self) -> Dict[str, any]:
        """Get check summary."""
        return {
            "total": len(self.checks),
            "passed": sum(1 for _, p, _ in self.checks if p),
            "failed": sum(1 for _, p, _ in self.checks if not p),
            "checks": [
                {"name": name, "passed": passed, "message": message}
                for name, passed, message in self.checks
            ]
        }


def main():
    """Main health check function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="System health check")
    parser.add_argument(
        "--check-server",
        action="store_true",
        help="Also check if API server is running"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    checker = HealthChecker()
    success = checker.run_all_checks(check_server=args.check_server)
    
    if args.json:
        import json
        print(json.dumps(checker.get_summary(), indent=2))
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
