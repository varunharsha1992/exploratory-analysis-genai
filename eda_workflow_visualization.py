#!/usr/bin/env python3
"""
EDA Workflow Visualization using LangGraph's built-in methods

This script demonstrates the exact approach from your example,
applied to the EDA workflow.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langgraph.graph import StateGraph, START, END
from EDA.workflow.eda_workflow_state import EDAWorkflowState

def visualize_eda_workflow():
    """Visualize the EDA workflow using LangGraph's built-in visualization"""
    
    print("🎯 EDA Workflow Visualization")
    print("=" * 50)
    
    # Import your existing workflow
    from EDA.workflow.eda_workflow import create_eda_workflow
    
    # Create the workflow (this is your existing workflow)
    workflow = create_eda_workflow()
    
    # Get the graph
    graph = workflow.get_graph()
    
    print("📊 Generating Mermaid diagram...")
    
    # Generate Mermaid diagram (this is the key part from your example)
    mermaid_diagram = graph.draw_mermaid()
    
    print("✅ Mermaid diagram generated!")
    print("\n📋 Diagram Content:")
    print("-" * 40)
    print(mermaid_diagram)
    print("-" * 40)
    
    # Save to file
    output_path = Path("workflow_visualizations")
    output_path.mkdir(exist_ok=True)
    
    mermaid_file = output_path / "eda_workflow_langgraph.mmd"
    with open(mermaid_file, "w", encoding="utf-8") as f:
        f.write(mermaid_diagram)
    
    print(f"\n💾 Saved to: {mermaid_file}")
    
    # Try to generate PNG (like in your example)
    try:
        print("\n🖼️  Generating PNG image...")
        png_data = graph.draw_mermaid_png()
        
        png_file = output_path / "eda_workflow_langgraph.png"
        with open(png_file, "wb") as f:
            f.write(png_data)
        
        print(f"✅ PNG saved to: {png_file}")
        
        # If running in Jupyter, display the image
        try:
            from IPython.display import Image, display
            print("🖼️  Displaying image in Jupyter...")
            display(Image(graph.draw_mermaid_png()))
        except ImportError:
            print("💡 To display images, run this in a Jupyter notebook")
            
    except Exception as e:
        print(f"⚠️  PNG generation failed: {e}")
        print("💡 PNG requires additional dependencies")
    
    return mermaid_diagram

def create_simple_example():
    """Create the exact simple example from your code"""
    
    print("\n🎯 Simple Example (from your code)")
    print("=" * 40)
    
    # Define a simple state schema
    class State:
        pass
    
    # Initialize the graph builder
    builder = StateGraph(State)
    
    # Add nodes and edges
    builder.add_node("start_node", lambda state: {"message": "Start"})
    builder.add_edge(START, "start_node")
    builder.add_edge("start_node", END)
    
    # Compile the graph
    graph = builder.compile()
    
    # Generate Mermaid diagram
    mermaid_diagram = graph.get_graph().draw_mermaid()
    
    print("📊 Simple Example Mermaid:")
    print("-" * 30)
    print(mermaid_diagram)
    print("-" * 30)
    
    # Save to file
    output_path = Path("workflow_visualizations")
    output_path.mkdir(exist_ok=True)
    
    simple_file = output_path / "simple_example.mmd"
    with open(simple_file, "w", encoding="utf-8") as f:
        f.write(mermaid_diagram)
    
    print(f"✅ Simple example saved to: {simple_file}")
    
    return graph

def main():
    """Main function"""
    
    try:
        # Create simple example
        simple_graph = create_simple_example()
        
        # Visualize EDA workflow
        eda_diagram = visualize_eda_workflow()
        
        print("\n✅ All visualizations completed!")
        print("\n📁 Files created:")
        print("   - workflow_visualizations/simple_example.mmd")
        print("   - workflow_visualizations/eda_workflow_langgraph.mmd")
        
        print("\n🎯 Usage Instructions:")
        print("   1. Copy content from any .mmd file")
        print("   2. Paste into https://mermaid.live/")
        print("   3. Or use VS Code with Mermaid extension")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
