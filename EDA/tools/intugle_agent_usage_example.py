"""
Example showing how agents can use cached Intugle components
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from EDA.tools.intugle_setup import get_cached_intugle_components, create_intugle_agent_integration

def example_agent_usage():
    """Example of how agents can use cached Intugle components"""
    
    print("=== AGENT USAGE EXAMPLE ===")
    
    # Method 1: Get cached components directly
    print("\n1. Getting cached components directly...")
    cached_components = get_cached_intugle_components()
    
    if cached_components:
        print("✓ Found cached components!")
        print(f"  - KnowledgeBuilder: {'Available' if cached_components['knowledge_builder'] else 'Not available'}")
        print(f"  - DataProductBuilder: {'Available' if cached_components['data_product_builder'] else 'Not available'}")
        print(f"  - Cache directory: {cached_components['cache_dir']}")
        
        # Use the KnowledgeBuilder directly
        kb = cached_components['knowledge_builder']
        if kb:
            print(f"  - Datasets available: {list(kb.datasets.keys())}")
            print(f"  - Total variables: {sum(len(dataset.source_table_model.columns) for dataset in kb.datasets.values())}")
    else:
        print("✗ No cached components found. Please run setup first.")
        return
    
    # Method 2: Create full integration for agents
    print("\n2. Creating full integration for agents...")
    integration = create_intugle_agent_integration()
    
    if integration:
        print("✓ Integration created successfully!")
        
        # Get available variables
        variables = integration.get_available_variables()
        print(f"  - Available variables: {len(variables)}")
        
        # Example: Search for variables related to "customer"
        if variables:
            customer_vars = [var for var in variables if 'customer' in var.lower()]
            print(f"  - Customer-related variables: {customer_vars[:5]}")  # Show first 5
        
        # Example: Get related variables
        if variables:
            target_var = variables[0]  # Use first variable as example
            related = integration.get_related_variables(target_var, max_results=3)
            print(f"  - Variables related to '{target_var}': {len(related)}")
    else:
        print("✗ Failed to create integration")

def example_agent_integration():
    """Example of how to integrate with an agent"""
    
    print("\n=== AGENT INTEGRATION EXAMPLE ===")
    
    # This is how you would use it in an actual agent
    class ExampleAgent:
        def __init__(self):
            self.intugle = create_intugle_agent_integration()
        
        def analyze_variable(self, variable_name: str):
            """Analyze a specific variable using Intugle"""
            if not self.intugle:
                return {"error": "Intugle not available"}
            
            # Get variable information
            variables = self.intugle.get_available_variables()
            if variable_name not in variables:
                return {"error": f"Variable '{variable_name}' not found"}
            
            # Get related variables
            related = self.intugle.get_related_variables(variable_name, max_results=5)
            
            return {
                "variable": variable_name,
                "related_variables": [var["variable_id"] for var in related],
                "total_available_variables": len(variables)
            }
    
    # Test the agent
    agent = ExampleAgent()
    if agent.intugle:
        # Test with first available variable
        variables = agent.intugle.get_available_variables()
        if variables:
            result = agent.analyze_variable(variables[0])
            print(f"Agent analysis result: {result}")
        else:
            print("No variables available for testing")
    else:
        print("Agent could not initialize Intugle integration")

if __name__ == "__main__":
    example_agent_usage()
    example_agent_integration()
