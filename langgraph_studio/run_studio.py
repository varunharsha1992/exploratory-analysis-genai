#!/usr/bin/env python3
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
        print("
👋 LangGraph Studio stopped by user")
        return 0

if __name__ == "__main__":
    sys.exit(main())
