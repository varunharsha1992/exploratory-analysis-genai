"""
Comprehensive debugging test for EDA Worker Loop Agent

This test creates dummy hypotheses and tests the EDA worker loop functionality
with detailed debugging for column mapping issues between user input and KB.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from EDA.agents.eda_worker_loop.eda_worker_loop import EDAWorkerLoopAgent, eda_worker_loop_agent
from EDA.agents.eda_analysis.eda_analysis import EDAAnalysisAgent
from EDA.tools.intugle_agent_tools import intugle_tools
from EDA.workflow.eda_workflow_state import EDAWorkflowState
from datetime import datetime
import json
import pandas as pd

def debug_kb_availability():
    """Debug KB availability and variable profiles"""
    print("=== DEBUGGING KB AVAILABILITY ===")
    # Check if Intugle tools are available
    print(f"Intugle tools available: {intugle_tools.is_available()}")
    
    if intugle_tools.is_available():
        # Get variable profiles
        print("\n--- Fetching KB Variable Profiles ---")
        profiles = intugle_tools.get_variable_profiles()
        
        if "error" in profiles:
            print(f"❌ KB Error: {profiles['error']}")
            return False
        else:
            print(f"✅ KB Profiles loaded: {len(profiles.get('profiles', {}))} tables")
            
            # Show available tables and their fields
            for table_name, table_data in profiles.get('profiles', {}).items():
                print(f"\nTable: {table_name}")
                if 'fields' in table_data:
                    for field in table_data['fields']:
                        print(f"  - {field.get('id', 'unknown')}: {field.get('description', 'no description')}")
                else:
                    print(f"  - No fields found")
            
            return True
    else:
        print("❌ Intugle tools not available")
        return False

def debug_etl_schema_creation(hypothesis):
    """Debug ETL schema creation process"""
    print(f"\n=== DEBUGGING ETL SCHEMA CREATION ===")
    print(f"Hypothesis: {hypothesis.get('hypothesis', 'N/A')}")
    
    # Extract hypothesis components
    target_var = hypothesis.get("target_variable", {})
    predictor_var = hypothesis.get("predictor_variable", {})
    data_req = hypothesis.get("data_requirements", {})
    
    print(f"Target Variable: {target_var.get('name', 'N/A')}")
    print(f"Predictor Variable: {predictor_var.get('name', 'N/A')}")
    print(f"Required Tables: {data_req.get('required_tables', [])}")
    print(f"Required Columns: {data_req.get('required_columns', [])}")
    
    # Create EDA Analysis Agent to test ETL schema creation
    try:
        eda_agent = EDAAnalysisAgent(
            kb=None,
            target_variable=target_var.get('name', ''),
            timeout=60,
            intugle_tools=intugle_tools
        )
        

        print(f"Running Analysis")
        result = eda_agent.analyze_hypothesis(hypothesis=hypothesis)
        print(f"Analysis result: {result}")
            
    except Exception as e:
        print(f"❌ ETL Schema creation error: {str(e)}")
        return False
    
    return True

def analyze_etl_schema_issues(etl_schema, hypothesis):
    """Analyze ETL schema for column mapping issues"""
    print("\n--- ETL Schema Analysis ---")
    
    # Check if schema has fields
    if 'fields' not in etl_schema:
        print("❌ No fields found in ETL schema")
        return
    
    fields = etl_schema['fields']
    print(f"ETL Schema has {len(fields)} fields:")
    
    # Extract user input column names
    user_columns = hypothesis.get("data_requirements", {}).get("required_columns", [])
    print(f"User input columns: {user_columns}")
    
    # Check each field in the schema
    for field in fields:
        field_id = field.get('id', '')
        field_name = field.get('name', '')
        field_category = field.get('category', '')
        
        print(f"  Field: {field_id} -> {field_name} ({field_category})")
        
        # Check if field matches user input
        matches_user_input = any(user_col in field_id for user_col in user_columns)
        if matches_user_input:
            print(f"    ✅ Matches user input")
        else:
            print(f"    ⚠️ Does not match user input")
    
    # Check for potential column mapping issues
    print("\n--- Column Mapping Analysis ---")
    
    # Get KB profiles for comparison
    if intugle_tools.is_available():
        kb_profiles = intugle_tools.get_variable_profiles()
        if "profiles" in kb_profiles:
            print("Available KB fields:")
            for table_name, table_data in kb_profiles["profiles"].items():
                if 'fields' in table_data:
                    for field in table_data['fields']:
                        field_id = field.get('id', '')
                        if any(user_col in field_id for user_col in user_columns):
                            print(f"  ✅ KB Match: {field_id} - {field.get('description', '')}")
                        elif any(user_col.lower() in field_id.lower() for user_col in user_columns):
                            print(f"  🔍 Potential Match: {field_id} - {field.get('description', '')}")

def create_dummy_hypotheses():
    """Create dummy hypotheses for testing"""
    return [
        {
      "hypothesis_id": "hyp_2",
      "hypothesis": "The percentage of returns negatively affects the sales volume, reflecting customer dissatisfaction.",
      "target_variable": {
        "name": "sales.vol",
        "alias": "sales_vol",
        "transformation": "none",
        "measure_func": "sum"
      },
      "predictor_variable": {
        "name": "sales.return_per",
        "alias": "sales_return_per",
        "transformation": "none",
        "measure_func": "average"
      },
      "relationship_type": "linear_relationship",
      "expected_impact": "negative",
      "confidence": 0.78,
      "research_support": [
        "Studies indicate a direct correlation between return rates and sales performance.",
        "High returns detract from overall sales volume, signaling consumer issues."
      ],
      "interaction_features": [],
      "test_priority": 0.85,
      "aggregate_by": {
        "product": "sales.prod_id",
        "date_sale": "sales.date"
      },
      "data_requirements": {
        "required_tables": [
          "sales"
        ],
        "required_columns": [
          "sales.vol",
          "sales.return_per",
          "sales.date",
          "sales.prod_id"
        ],
        "join_requirements": ""
      }
    }
  ]  

def test_eda_worker_loop_agent():
    """Test the EDA Worker Loop Agent with comprehensive debugging"""
    print("=== COMPREHENSIVE EDA WORKER LOOP DEBUGGING ===")
    
    # Step 1: Debug KB availability
    print("\n" + "="*60)
    kb_available = debug_kb_availability()
    
    # Step 2: Create dummy hypotheses
    print("\n" + "="*60)
    dummy_hypotheses = create_dummy_hypotheses()
    print(f"Created {len(dummy_hypotheses)} dummy hypotheses")
    
    # Step 3: Debug ETL schema creation for each hypothesis
    print("\n" + "="*60)
    for i, hypothesis in enumerate(dummy_hypotheses):
        print(f"\n--- DEBUGGING HYPOTHESIS {i+1} ---")
        etl_success = debug_etl_schema_creation(hypothesis)
        if not etl_success:
            print(f"❌ ETL schema creation failed for hypothesis {i+1}")
    
    # # Step 4: Test EDA Worker Loop Agent with proper EDAWorkflowState
    # print("\n" + "="*60)
    # print("=== Testing EDA Worker Loop Agent with EDAWorkflowState ===")
    
    # # Create proper EDAWorkflowState
    # state: EDAWorkflowState = {
    #     # Input parameters
    #     "target_variable": "sales.vol",
    #     "eda_request": "Analyze the relationship between competitive pricing and sales volume",
    #     "domain_context": "FMCG retail analytics",
    #     "hypothesis_limit": 5,
    #     "data": None,  # No data provided, will use KB
        
    #     # Agent outputs (initially None)
    #     "univariate_results": None,
    #     "generated_hypotheses": dummy_hypotheses,
    #     "hypothesis_testing_results": None,
    #     "final_summary": None,
        
    #     # Workflow metadata
    #     "execution_status": "in_progress",
    #     "error_messages": [],
    #     "performance_metrics": {},
    #     "current_agent": "eda_worker_loop",
        
    #     # Additional workflow state
    #     "start_time": datetime.now(),
    #     "end_time": None,
    #     "config": {
    #         "eda_worker_config": {
    #             "max_workers": 3,
    #             "timeout_per_hypothesis": 60
    #         }
    #     },
    #     "kb": None  # Will use fresh data loading
    # }
    
    # print("✅ EDAWorkflowState created with proper structure")
    # print(f"State keys: {list(state.keys())}")
    # print(f"Target variable: {state['target_variable']}")
    # print(f"Hypotheses count: {len(state['generated_hypotheses'])}")
    
    # # Test the LangGraph node function with proper state
    # print("\n--- Testing LangGraph integration with EDAWorkflowState ---")
    # try:
    #     # Test the LangGraph node function
    #     updated_state = eda_worker_loop_agent(state)
    #     print("✅ LangGraph integration completed successfully")
    #     print(f"Updated state keys: {list(updated_state.keys())}")
    #     print(f"Execution status: {updated_state.get('execution_status')}")
    #     print(f"Current agent: {updated_state.get('current_agent')}")
        
    #     # Check if hypothesis testing results were added
    #     if updated_state.get('hypothesis_testing_results'):
    #         print("✅ Hypothesis testing results found in state")
    #         results = updated_state['hypothesis_testing_results']
    #         print(f"  - Total hypotheses: {results.get('total_hypotheses', 0)}")
    #         print(f"  - Successful: {results.get('successful_analyses', 0)}")
    #         print(f"  - Failed: {results.get('failed_analyses', 0)}")
    #     else:
    #         print("⚠️ No hypothesis testing results in state")
        
    # except Exception as e:
    #     print(f"❌ LangGraph integration failed: {str(e)}")
    #     import traceback
    #     traceback.print_exc()
    #     return False
    
    # print("\n=== Test completed successfully! ===")
    return True

def test_edge_cases():
    """Test edge cases for the EDA Worker Loop Agent"""
    print("\n=== Testing Edge Cases ===")
    
    # Test with empty hypotheses using EDAWorkflowState
    print("\n--- Testing with empty hypotheses ---")
    try:
        empty_state: EDAWorkflowState = {
            "target_variable": "sales.vol",
            "eda_request": "Test with empty hypotheses",
            "domain_context": "FMCG retail analytics",
            "hypothesis_limit": 5,
            "data": None,
            "univariate_results": None,
            "generated_hypotheses": [],  # Empty hypotheses
            "hypothesis_testing_results": None,
            "final_summary": None,
            "execution_status": "in_progress",
            "error_messages": [],
            "performance_metrics": {},
            "current_agent": "eda_worker_loop",
            "start_time": datetime.now(),
            "end_time": None,
            "config": {"eda_worker_config": {"max_workers": 3, "timeout_per_hypothesis": 60}},
            "kb": None
        }
        
        updated_state = eda_worker_loop_agent(empty_state)
        print("✅ Empty hypotheses handled correctly")
        print(f"Execution status: {updated_state.get('execution_status')}")
        print(f"Hypothesis testing results: {updated_state.get('hypothesis_testing_results')}")
    except Exception as e:
        print(f"❌ Empty hypotheses test failed: {str(e)}")
        return False
    
    # Test with None hypotheses
    print("\n--- Testing with None hypotheses ---")
    try:
        none_state: EDAWorkflowState = {
            "target_variable": "sales.vol",
            "eda_request": "Test with None hypotheses",
            "domain_context": "FMCG retail analytics",
            "hypothesis_limit": 5,
            "data": None,
            "univariate_results": None,
            "generated_hypotheses": None,  # None hypotheses
            "hypothesis_testing_results": None,
            "final_summary": None,
            "execution_status": "in_progress",
            "error_messages": [],
            "performance_metrics": {},
            "current_agent": "eda_worker_loop",
            "start_time": datetime.now(),
            "end_time": None,
            "config": {"eda_worker_config": {"max_workers": 3, "timeout_per_hypothesis": 60}},
            "kb": None
        }
        
        updated_state = eda_worker_loop_agent(none_state)
        print("✅ None hypotheses handled correctly")
    except Exception as e:
        print(f"⚠️ None hypotheses test failed (expected): {str(e)}")
        # This is expected behavior - None should cause an error
    
    print("=== Edge case tests completed ===")
    return True

if __name__ == "__main__":
    print("COMPREHENSIVE EDA WORKER LOOP DEBUGGING")
    print("=" * 60)
    print("This test will debug column mapping issues between user input and KB")
    print("=" * 60)
    
    # Run comprehensive debugging test
    success = test_eda_worker_loop_agent()
    
    if success:
        # Run edge case tests
        print("\n" + "="*60)
        test_edge_cases()
        print("\n🎉 All debugging tests completed!")
        print("Review the output above to identify column mapping issues.")
    else:
        print("\n❌ Debugging tests failed!")
