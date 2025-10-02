"""
LangGraph Studio Configuration for EDA Workflow

This module provides configuration for visualizing the EDA workflow in LangGraph Studio.
"""

import json
from pathlib import Path
from EDA.workflow.eda_workflow import create_eda_workflow, create_simplified_workflow
from EDA.workflow.eda_workflow_state import EDAWorkflowState

def export_workflow_for_studio(workflow_type: str = "full", output_dir: str = "langgraph_studio"):
    """
    Export workflow configuration for LangGraph Studio
    
    Args:
        workflow_type: "full" or "simplified"
        output_dir: Directory to save studio files
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create workflow
    if workflow_type == "simplified":
        workflow = create_simplified_workflow()
        workflow_name = "eda_simplified_workflow"
    else:
        workflow = create_eda_workflow()
        workflow_name = "eda_full_workflow"
    
    # Create LangGraph Studio configuration
    studio_config = {
        "graphs": {
            workflow_name: {
                "path": f"EDA.workflow.eda_workflow:create_{workflow_type}_workflow",
                "description": f"EDA {workflow_type.title()} Workflow for Exploratory Data Analysis"
            }
        },
        "env": ".env"
    }
    
    # Save configuration
    config_file = output_path / "langgraph.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(studio_config, f, indent=2)
    
    # Define workflow structure manually
    if workflow_type == "simplified":
        nodes = ["univariate_analysis", "hypothesis_generation", "error_handler"]
        edges = [
            {"source": "univariate_analysis", "target": "hypothesis_generation", "condition": "continue"},
            {"source": "univariate_analysis", "target": "error_handler", "condition": "error_handler"},
            {"source": "hypothesis_generation", "target": "END", "condition": "continue"},
            {"source": "hypothesis_generation", "target": "error_handler", "condition": "error_handler"},
            {"source": "error_handler", "target": "END", "condition": "end"}
        ]
    else:
        nodes = ["univariate_analysis", "hypothesis_generation", "eda_worker_loop", "summarizer", "error_handler"]
        edges = [
            {"source": "univariate_analysis", "target": "hypothesis_generation", "condition": "continue"},
            {"source": "univariate_analysis", "target": "error_handler", "condition": "error_handler"},
            {"source": "hypothesis_generation", "target": "eda_worker_loop", "condition": "continue"},
            {"source": "hypothesis_generation", "target": "error_handler", "condition": "error_handler"},
            {"source": "eda_worker_loop", "target": "summarizer", "condition": "continue"},
            {"source": "eda_worker_loop", "target": "error_handler", "condition": "error_handler"},
            {"source": "summarizer", "target": "END", "condition": "continue"},
            {"source": "summarizer", "target": "error_handler", "condition": "error_handler"},
            {"source": "error_handler", "target": "END", "condition": "end"}
        ]
    
    # Create workflow export
    workflow_export = {
        "name": workflow_name,
        "description": f"EDA {workflow_type.title()} Workflow",
        "nodes": nodes,
        "edges": edges,
        "entry_point": "univariate_analysis"
    }
    
    # Save workflow export
    workflow_file = output_path / f"{workflow_name}.json"
    with open(workflow_file, "w", encoding="utf-8") as f:
        json.dump(workflow_export, f, indent=2)
    
    # Create sample state for testing
    sample_state = {
        "target_variable": "sales.units_sold",
        "eda_request": "Analyze factors influencing sales volume",
        "domain_context": "retail",
        "hypothesis_limit": 5,
        "univariate_results": None,
        "generated_hypotheses": None,
        "hypothesis_testing_results": None,
        "final_summary": None,
        "execution_status": "initialized",
        "error_messages": [],
        "performance_metrics": {},
        "current_agent": "",
        "start_time": "2024-01-01T00:00:00Z",
        "end_time": None,
        "data": None,
        "config": {},
        "kb": None
    }
    
    # Save sample state
    state_file = output_path / "sample_state.json"
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(sample_state, f, indent=2)
    
    print(f"✅ LangGraph Studio configuration exported to {output_dir}/")
    print(f"📁 Files created:")
    print(f"   - {config_file}")
    print(f"   - {workflow_file}")
    print(f"   - {state_file}")
    print(f"\n🚀 To use in LangGraph Studio:")
    print(f"   1. Navigate to the {output_dir} directory")
    print(f"   2. Run: langgraph studio")
    print(f"   3. Open http://localhost:8123 in your browser")
    
    return {
        "config_file": str(config_file),
        "workflow_file": str(workflow_file),
        "state_file": str(state_file)
    }

def create_studio_runner():
    """Create a runner script for LangGraph Studio"""
    
    runner_script = '''#!/usr/bin/env python3
"""
LangGraph Studio Runner for EDA Workflow

Run this script to start LangGraph Studio with the EDA workflow.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Start LangGraph Studio"""
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    
    print("🚀 Starting LangGraph Studio for EDA Workflow...")
    print(f"📁 Working directory: {script_dir}")
    
    try:
        # Run langgraph studio
        subprocess.run([
            sys.executable, "-m", "langgraph", "studio",
            "--port", "8123",
            "--host", "localhost"
        ], cwd=script_dir, check=True)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start LangGraph Studio: {e}")
        print("💡 Make sure you have langgraph installed: pip install langgraph[studio]")
        return 1
    except KeyboardInterrupt:
        print("\n👋 LangGraph Studio stopped by user")
        return 0

if __name__ == "__main__":
    sys.exit(main())
'''
    
    output_path = Path("langgraph_studio")
    output_path.mkdir(exist_ok=True)
    
    runner_file = output_path / "run_studio.py"
    with open(runner_file, "w", encoding="utf-8") as f:
        f.write(runner_script)
    
    # Make it executable on Unix systems
    runner_file.chmod(0o755)
    
    print(f"✅ Studio runner created: {runner_file}")
    return str(runner_file)

def print_studio_instructions():
    """Print instructions for using LangGraph Studio"""
    
    instructions = """
🎯 LangGraph Studio Setup Instructions

1. Install LangGraph Studio:
   pip install langgraph[studio]

2. Export workflow configuration:
   python -c "from EDA.workflow.langgraph_studio_config import export_workflow_for_studio; export_workflow_for_studio()"

3. Navigate to the langgraph_studio directory:
   cd langgraph_studio

4. Start LangGraph Studio:
   langgraph studio

5. Open your browser to:
   http://localhost:8123

📊 Workflow Visualization Features:
   - Interactive graph visualization
   - Node-by-node execution tracking
   - State inspection at each step
   - Debug mode with breakpoints
   - Real-time workflow monitoring

🔧 Available Workflows:
   - eda_full_workflow: Complete EDA workflow with all agents
   - eda_simplified_workflow: Simplified workflow for testing

📝 Sample State:
   Use the sample_state.json file to test the workflow with sample data.

🛠️ Troubleshooting:
   - Ensure all dependencies are installed
   - Check that the EDA workflow modules are importable
   - Verify the target_data_path exists for testing
"""
    
    print(instructions)

if __name__ == "__main__":
    print("🎯 Setting up LangGraph Studio for EDA Workflow...")
    
    # Export both workflows
    export_workflow_for_studio("full")
    export_workflow_for_studio("simplified")
    
    # Create runner script
    create_studio_runner()
    
    # Print instructions
    print_studio_instructions()
