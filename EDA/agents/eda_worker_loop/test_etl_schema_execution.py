"""
Test ETL Schema Execution and DataProduct Building

This test creates dummy ETL schemas and tests the build_dataproduct functionality
to identify and fix column mapping issues.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from EDA.tools.intugle_agent_tools import intugle_tools, IntugleAgentTools
import json
from typing import Dict, List, Any

def test_kb_structure():
    """Test and analyze KB structure"""
    print("=== TESTING KB STRUCTURE ===")
    
    if not intugle_tools.is_available():
        print("❌ Intugle tools not available")
        return False
    
    # Get variable profiles
    profiles = intugle_tools.get_variable_profiles()
    
    if "error" in profiles:
        print(f"❌ KB Error: {profiles['error']}")
        return False
    
    print(f"✅ KB Profiles loaded: {len(profiles.get('profiles', {}))} tables")
    
    # Analyze structure
    # kb_profiles = profiles.get('profiles', {})
    # tables_with_fields = 0
    
    # for table_name, table_data in kb_profiles.items():
    #     if 'fields' in table_data and table_data['fields']:
    #         tables_with_fields += 1
    #         print(f"✅ Table {table_name}: {len(table_data['fields'])} fields")
    #         for field in table_data['fields']:
    #             print(f"    - {field.get('id', 'unknown')}: {field.get('description', 'no description')}")
    #     else:
    #         print(f"❌ Table {table_name}: No fields found")
    
    # print(f"Tables with fields: {tables_with_fields}/{len(kb_profiles)}")
    return True

def create_dummy_etl_schemas():
    """Create various dummy ETL schemas for testing"""
    schemas = {'name': 'promo_campaign_sales_analysis', 'fields': [{'id': 'sales.vol', 'name': 'Sales Volume', 'category': 'measure'}, {'id': 'promo_camp.cost_promo', 'name': 'Promotional Cost', 'category': 'measure'}], 'filter': {'selections': [], 'limit': 1000}}
    
    return schemas

def test_etl_schema_execution():
    """Test ETL schema execution with various schemas"""
    print("\n=== TESTING ETL SCHEMA EXECUTION ===")
    
    if not intugle_tools.is_available():
        print("❌ Intugle tools not available")
        return False
    
    # Create dummy schemas
    schemas = create_dummy_etl_schemas()
    
    results = {}
    
    for schema_name, schema_config in schemas.items():
        print(f"\n--- Testing Schema: {schema_name} ---")
        print(f"Schema: {json.dumps(schema_config, indent=2)}")
        
        try:
            # Test build_dataproduct
            build_result = intugle_tools.build_dataproduct(schemas)
            
            print(f"Build result: {json.dumps(build_result, indent=2)}")
            
            if "error" in build_result:
                print(f"❌ Build failed: {build_result['error']}")
                results[schema_name] = {
                    "status": "failed",
                    "error": build_result['error']
                }
            else:
                print(f"✅ Build successful")
                results[schema_name] = {
                    "status": "success",
                    "result": build_result
                }
                
        except Exception as e:
            print(f"❌ Exception during build: {str(e)}")
            results[schema_name] = {
                "status": "exception",
                "error": str(e)
            }
    
    return results

def test_llm_etl_schema_generation():
    """Test LLM-based ETL schema generation"""
    print("\n=== TESTING LLM ETL SCHEMA GENERATION ===")
    
    if not intugle_tools.is_available():
        print("❌ Intugle tools not available")
        return False
    
    # Test different user inputs
    test_inputs = [
        "I need to analyze sales volume and pricing data",
        "Show me marketing spend by channel",
        "Analyze promotional campaigns and their impact on sales",
        "I want to see pricing strategies and their effects"
    ]
    
    results = {}
    
    for i, user_input in enumerate(test_inputs):
        print(f"\n--- Test Input {i+1}: {user_input} ---")
        
        try:
            # Generate ETL schema using LLM
            etl_result = intugle_tools.create_etl_schema_with_llm(user_input)
            
            print(f"ETL Generation Result: {json.dumps(etl_result, indent=2)}")
            
            if "error" in etl_result:
                print(f"❌ ETL generation failed: {etl_result['error']}")
                results[f"input_{i+1}"] = {
                    "status": "generation_failed",
                    "error": etl_result['error']
                }
            else:
                # Test the generated schema
                etl_schema = etl_result.get("etl_schema", {})
                print(f"Generated schema: {json.dumps(etl_schema, indent=2)}")
                
                # Try to build the generated schema
                build_result = intugle_tools.build_dataproduct(etl_schema)
                
                if "error" in build_result:
                    print(f"❌ Generated schema build failed: {build_result['error']}")
                    results[f"input_{i+1}"] = {
                        "status": "build_failed",
                        "etl_schema": etl_schema,
                        "error": build_result['error']
                    }
                else:
                    print(f"✅ Generated schema build successful")
                    results[f"input_{i+1}"] = {
                        "status": "success",
                        "etl_schema": etl_schema,
                        "build_result": build_result
                    }
                    
        except Exception as e:
            print(f"❌ Exception during ETL generation: {str(e)}")
            results[f"input_{i+1}"] = {
                "status": "exception",
                "error": str(e)
            }
    
    return results

def analyze_results(execution_results, generation_results = {}):
    """Analyze test results and provide recommendations"""
    print("\n=== ANALYZING RESULTS ===")
    
    # Analyze execution results
    print("\n--- ETL Schema Execution Analysis ---")
    successful_executions = 0
    failed_executions = 0
    
    for schema_name, result in execution_results.items():
        if result["status"] == "success":
            successful_executions += 1
            print(f"✅ {schema_name}: Success")
        else:
            failed_executions += 1
            print(f"❌ {schema_name}: {result['status']} - {result.get('error', 'Unknown error')}")
    
    print(f"Execution Summary: {successful_executions} successful, {failed_executions} failed")
    
    # Analyze generation results
    print("\n--- LLM ETL Generation Analysis ---")
    successful_generations = 0
    failed_generations = 0
    
    for input_name, result in generation_results.items():
        if result["status"] == "success":
            successful_generations += 1
            print(f"✅ {input_name}: Success")
        else:
            failed_generations += 1
            print(f"❌ {input_name}: {result['status']} - {result.get('error', 'Unknown error')}")
    
    print(f"Generation Summary: {successful_generations} successful, {failed_generations} failed")
    
    # Provide recommendations
    print("\n--- RECOMMENDATIONS ---")
    
    if failed_executions > 0:
        print("🔧 ETL Schema Issues:")
        print("  - Check field IDs match KB structure")
        print("  - Verify table names exist in KB")
        print("  - Ensure proper field categories (measure/dimension)")
    
    if failed_generations > 0:
        print("🔧 LLM Generation Issues:")
        print("  - Improve prompt to better map user input to KB fields")
        print("  - Add better field matching logic")
        print("  - Validate generated schemas before building")
    
    if successful_executions > 0 and successful_generations > 0:
        print("✅ System is working correctly for some cases")
        print("  - Focus on fixing the failing cases")
        print("  - Use successful patterns as templates")

def main():
    """Main test function"""
    print("COMPREHENSIVE ETL SCHEMA EXECUTION TEST")
    print("=" * 60)
    
    # Test KB structure
    kb_ok = test_kb_structure()
    
    if not kb_ok:
        print("❌ KB structure issues detected. Cannot proceed with ETL tests.")
        return
    
    # Test ETL schema execution
    execution_results = test_etl_schema_execution()
    
    # Test LLM ETL schema generation
    # generation_results = test_llm_etl_schema_generation()
    
    # Analyze results
    analyze_results(execution_results)
    
    print("\n🎉 ETL Schema testing completed!")
    print("Review the results above to identify and fix issues.")

if __name__ == "__main__":
    main()
