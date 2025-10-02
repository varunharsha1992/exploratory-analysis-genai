#!/usr/bin/env python3
"""
Setup LangGraph Studio for EDA Workflow Visualization

This script exports the EDA workflow configuration for LangGraph Studio.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from EDA.workflow.langgraph_studio_config import export_workflow_for_studio, create_studio_runner, print_studio_instructions

def main():
    """Main setup function"""
    
    print("🎯 Setting up LangGraph Studio for EDA Workflow...")
    print("=" * 60)
    
    try:
        # Export both workflow types
        print("📤 Exporting full EDA workflow...")
        export_workflow_for_studio("full")
        
        print("\n📤 Exporting simplified EDA workflow...")
        export_workflow_for_studio("simplified")
        
        print("\n🏃 Creating studio runner script...")
        create_studio_runner()
        
        print("\n✅ Setup completed successfully!")
        print("\n" + "=" * 60)
        
        # Print instructions
        print_studio_instructions()
        
        return 0
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
