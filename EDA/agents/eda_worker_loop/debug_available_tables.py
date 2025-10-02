"""
Debug Available Tables and Fields in KB

This script will identify what tables and fields are actually available
in the knowledge base to fix the ETL schema generation issues.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from EDA.tools.intugle_agent_tools import intugle_tools
import json
from typing import Dict, List, Any

def debug_available_tables():
    """Debug what tables and fields are actually available in KB"""
    print("=== DEBUGGING AVAILABLE TABLES AND FIELDS ===")
    
    if not intugle_tools.is_available():
        print("❌ Intugle tools not available")
        return None
    
    # Get variable profiles
    profiles = intugle_tools.get_variable_profiles()
    
    if "error" in profiles:
        print(f"❌ KB Error: {profiles['error']}")
        return None
    
    print(f"✅ KB Profiles loaded: {len(profiles.get('profiles', {}))} tables")
    
    # Analyze structure
    kb_profiles = profiles.get('profiles', {})
    available_tables = {}
    tables_with_fields = 0
    
    print("\n--- DETAILED KB STRUCTURE ANALYSIS ---")
    
    for table_name, table_data in kb_profiles.items():
        print(f"\nTable: {table_name}")
        print(f"  Type: {type(table_data)}")
        print(f"  Keys: {list(table_data.keys()) if isinstance(table_data, dict) else 'Not a dict'}")
        
        if isinstance(table_data, dict):
            if 'fields' in table_data and table_data['fields']:
                tables_with_fields += 1
                available_tables[table_name] = table_data['fields']
                print(f"  ✅ Fields: {len(table_data['fields'])}")
                for field in table_data['fields']:
                    print(f"    - {field.get('id', 'unknown')}: {field.get('description', 'no description')}")
            else:
                print(f"  ❌ No fields found")
                # Check if it's a direct field reference
                if 'id' in table_data:
                    print(f"  🔍 Direct field: {table_data.get('id', 'unknown')}")
        else:
            print(f"  ❌ Not a dict structure")
    
    print(f"\nSummary: {tables_with_fields} tables with fields out of {len(kb_profiles)} total")
    
    return available_tables

def create_correct_etl_schemas(available_tables):
    """Create ETL schemas using only available tables and fields"""
    print("\n=== CREATING CORRECT ETL SCHEMAS ===")
    
    if not available_tables:
        print("❌ No tables with fields available")
        return {}
    
    # Create schemas using only available fields
    correct_schemas = {}
    
    # Schema 1: Sales analysis using available sales fields
    if 'sales' in available_tables or any('sales' in table for table in available_tables):
        sales_fields = []
        for table_name, fields in available_tables.items():
            if 'sales' in table_name.lower():
                sales_fields.extend(fields)
        
        if sales_fields:
            correct_schemas['sales_analysis'] = {
                "name": "sales_analysis",
                "fields": [
                    {"id": field['id'], "name": field.get('name', field['id'].split('.')[-1]), "category": "measure"}
                    for field in sales_fields[:2]  # Take first 2 fields
                ],
                "filter": {
                    "selections": [],
                    "limit": 100
                }
            }
    
    # Schema 2: Marketing analysis using available marketing fields
    marketing_tables = [table for table in available_tables.keys() if any(keyword in table.lower() for keyword in ['ad', 'marketing', 'campaign'])]
    if marketing_tables:
        marketing_fields = []
        for table_name in marketing_tables:
            marketing_fields.extend(available_tables[table_name])
        
        if marketing_fields:
            correct_schemas['marketing_analysis'] = {
                "name": "marketing_analysis",
                "fields": [
                    {"id": field['id'], "name": field.get('name', field['id'].split('.')[-1]), "category": "measure"}
                    for field in marketing_fields[:2]  # Take first 2 fields
                ],
                "filter": {
                    "selections": [],
                    "limit": 100
                }
            }
    
    # Schema 3: Pricing analysis using available pricing fields
    pricing_tables = [table for table in available_tables.keys() if any(keyword in table.lower() for keyword in ['price', 'pricing', 'cost'])]
    if pricing_tables:
        pricing_fields = []
        for table_name in pricing_tables:
            pricing_fields.extend(available_tables[table_name])
        
        if pricing_fields:
            correct_schemas['pricing_analysis'] = {
                "name": "pricing_analysis",
                "fields": [
                    {"id": field['id'], "name": field.get('name', field['id'].split('.')[-1]), "category": "measure"}
                    for field in pricing_fields[:2]  # Take first 2 fields
                ],
                "filter": {
                    "selections": [],
                    "limit": 100
                }
            }
    
    print(f"Created {len(correct_schemas)} correct schemas:")
    for schema_name, schema in correct_schemas.items():
        print(f"  - {schema_name}: {len(schema['fields'])} fields")
        for field in schema['fields']:
            print(f"    * {field['id']} -> {field['name']}")
    
    return correct_schemas

def test_correct_schemas(correct_schemas):
    """Test the correct schemas to see if they work"""
    print("\n=== TESTING CORRECT SCHEMAS ===")
    
    if not intugle_tools.is_available():
        print("❌ Intugle tools not available")
        return {}
    
    results = {}
    
    for schema_name, schema_config in correct_schemas.items():
        print(f"\n--- Testing Schema: {schema_name} ---")
        print(f"Schema: {json.dumps(schema_config, indent=2)}")
        
        try:
            # Test build_dataproduct
            build_result = intugle_tools.build_dataproduct(schema_config)
            
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

def main():
    """Main debugging function"""
    print("COMPREHENSIVE KB TABLE AND FIELD DEBUGGING")
    print("=" * 60)
    
    # Step 1: Debug available tables
    available_tables = debug_available_tables()
    
    if not available_tables:
        print("❌ No tables with fields found. Cannot proceed.")
        return
    
    # Step 2: Create correct schemas
    correct_schemas = create_correct_etl_schemas(available_tables)
    
    if not correct_schemas:
        print("❌ No correct schemas could be created.")
        return
    
    # Step 3: Test correct schemas
    test_results = test_correct_schemas(correct_schemas)
    
    # Step 4: Analyze results
    print("\n=== FINAL ANALYSIS ===")
    successful = sum(1 for result in test_results.values() if result["status"] == "success")
    failed = sum(1 for result in test_results.values() if result["status"] != "success")
    
    print(f"Schema Test Results: {successful} successful, {failed} failed")
    
    if successful > 0:
        print("✅ Some schemas work! Use these as templates.")
        for schema_name, result in test_results.items():
            if result["status"] == "success":
                print(f"  ✅ {schema_name}: Working schema")
    else:
        print("❌ All schemas failed. Need to investigate further.")
        for schema_name, result in test_results.items():
            print(f"  ❌ {schema_name}: {result.get('error', 'Unknown error')}")
    
    print("\n🎉 KB debugging completed!")
    print("Use the working schemas as templates for ETL schema generation.")

if __name__ == "__main__":
    main()
