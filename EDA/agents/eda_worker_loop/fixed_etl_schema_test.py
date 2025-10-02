"""
Fixed ETL Schema Test using actual KB structure with Dataset Setup

This test uses the correct KB structure where each entry is a field,
not a table with fields. The KB has 64 individual fields from 13 tables.
Includes dataset setup using healthcare reference to ensure data is loaded.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from EDA.tools.intugle_agent_tools import intugle_tools
import json
from typing import Dict, List, Any

def setup_healthcare_dataset():
    """Setup healthcare dataset using KnowledgeBuilder to recreate KB with data"""
    print("=== SETTING UP HEALTHCARE DATASET WITH KNOWLEDGE BUILDER ===")
    
    try:
        # Import required modules
        from intugle.knowledge_builder import KnowledgeBuilder
        import pandas as pd
        import os
        
        print("Recreating KB with actual data using KnowledgeBuilder...")
        
        # Define the data paths
        data_dir = "./data-tools/sample_data/fmcg"
        
        # Check if data directory exists
        if not os.path.exists(data_dir):
            print(f"❌ Data directory not found: {data_dir}")
            return None
        
        # List of available CSV files
        csv_files = [
            "sales.csv", "ad_spends.csv", "comp_price.csv", "cons_fb.csv",
            "disti.csv", "mkt_coverage.csv", "prd_mstr.csv", "price_sense.csv",
            "pric_strat.csv", "promo_camp.csv", "retail_audit.csv", "rnd.csv", "social.csv"
        ]
        
        # Load datasets
        datasets = {}
        for csv_file in csv_files:
            file_path = os.path.join(data_dir, csv_file)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    table_name = csv_file.replace('.csv', '')
                    datasets[table_name] = df
                    print(f"✅ Loaded {table_name}: {df.shape[0]} rows, {df.shape[1]} columns")
                except Exception as e:
                    print(f"⚠️ Failed to load {csv_file}: {str(e)}")
            else:
                print(f"⚠️ File not found: {file_path}")
        
        if not datasets:
            print("❌ No datasets loaded")
            return None
        
        print(f"\n--- Building Knowledge Base with {len(datasets)} datasets ---")
        
        # Create KnowledgeBuilder with the datasets
        print("Creating KnowledgeBuilder with datasets...")
        kb = KnowledgeBuilder(datasets)
        
        # Build the knowledge base
        print("Building knowledge base... This may take a few minutes.")
        kb.build()
        
        print("✅ Knowledge base built successfully!")
        
        # Create data product builder with the new KB
        from intugle.dp_builder import DataProductBuilder
        dpb = DataProductBuilder()
        
        # Return components with the new KB
        components = {
            'knowledge_builder': kb,
            'data_product_builder': dpb,
            'datasets': datasets
        }
        
        return components
        
    except Exception as e:
        print(f"❌ Failed to setup dataset with KnowledgeBuilder: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_semantic_model_data(components):
    """Test the semantic model data availability"""
    print("\n=== TESTING SEMANTIC MODEL DATA ===")
    
    if not components:
        print("❌ No components available")
        return False
    
    try:
        # Test accessing components
        print(f"✅ Available components: {list(components.keys())}")
        
        # Test knowledge builder
        if 'knowledge_builder' in components:
            kb = components['knowledge_builder']
            print(f"✅ KnowledgeBuilder available: {type(kb)}")
            
            # Test accessing datasets from KB
            if hasattr(kb, 'datasets'):
                print(f"✅ KB has {len(kb.datasets)} datasets")
                for table_name, dataset in kb.datasets.items():
                    print(f"  - {table_name}: {dataset.shape if hasattr(dataset, 'shape') else 'No shape'}")
        
        # Test data product builder
        if 'data_product_builder' in components:
            dpb = components['data_product_builder']
            print(f"✅ DataProductBuilder available: {type(dpb)}")
        
        # Test raw datasets
        if 'datasets' in components:
            datasets = components['datasets']
            print(f"✅ Raw datasets available: {len(datasets)} tables")
            for table_name, df in datasets.items():
                print(f"  - {table_name}: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing components: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def get_available_tables_and_fields():
    """Get the actual available tables and fields from KB"""
    print("=== GETTING AVAILABLE TABLES AND FIELDS ===")
    
    if not intugle_tools.is_available():
        print("❌ Intugle tools not available")
        return None
    
    # Get variable profiles
    profiles = intugle_tools.get_variable_profiles()
    
    if "error" in profiles:
        print(f"❌ KB Error: {profiles['error']}")
        return None
    
    # The KB structure is: each entry is a field, not a table
    kb_profiles = profiles.get('profiles', {})
    
    # Group fields by table
    tables = {}
    for field_id, field_data in kb_profiles.items():
        if '.' in field_id:
            table_name = field_id.split('.')[0]
            if table_name not in tables:
                tables[table_name] = []
            
            # Create field entry
            field_entry = {
                'id': field_id,
                'name': field_data.get('business_name', field_id.split('.')[1]),
                'description': field_data.get('business_glossary', ''),
                'datatype': field_data.get('datatype_l1', 'unknown'),
                'table_name': table_name,
                'column_name': field_data.get('column_name', field_id.split('.')[1])
            }
            tables[table_name].append(field_entry)
    
    print(f"✅ Found {len(tables)} tables with fields:")
    for table_name, fields in tables.items():
        print(f"  - {table_name}: {len(fields)} fields")
        for field in fields[:3]:  # Show first 3 fields
            print(f"    * {field['id']}: {field['name']} ({field['datatype']})")
        if len(fields) > 3:
            print(f"    ... and {len(fields) - 3} more fields")
    
    return tables

def create_working_etl_schemas(available_tables):
    """Create ETL schemas using the correct KB structure"""
    print("\n=== CREATING WORKING ETL SCHEMAS ===")
    
    working_schemas = {}
    
    # Schema 1: Sales analysis
    if 'sales' in available_tables:
        sales_fields = available_tables['sales']
        if len(sales_fields) >= 2:
            working_schemas['sales_analysis'] = {
                "name": "sales_analysis",
                "fields": [
                    {
                        "id": sales_fields[0]['id'],
                        "name": sales_fields[0]['name'],
                        "category": "measure"
                    },
                    {
                        "id": sales_fields[1]['id'],
                        "name": sales_fields[1]['name'],
                        "category": "measure"
                    }
                ],
                "filter": {
                    "selections": [],
                    "limit": 100
                }
            }
            print(f"Working schemas: {working_schemas}")
            print(f"✅ Created sales_analysis schema with {len(sales_fields)} fields")
    
    # Schema 2: Marketing analysis
    if 'ad_spends' in available_tables:
        ad_fields = available_tables['ad_spends']
        if len(ad_fields) >= 2:
            working_schemas['marketing_analysis'] = {
                "name": "marketing_analysis",
                "fields": [
                    {
                        "id": ad_fields[0]['id'],
                        "name": ad_fields[0]['name'],
                        "category": "measure"
                    },
                    {
                        "id": ad_fields[1]['id'],
                        "name": ad_fields[1]['name'],
                        "category": "dimension"
                    }
                ],
                "filter": {
                    "selections": [],
                    "limit": 100
                }
            }
            print(f"✅ Created marketing_analysis schema with {len(ad_fields)} fields")
    
    # Schema 3: Pricing analysis
    if 'price_sense' in available_tables:
        price_fields = available_tables['price_sense']
        if len(price_fields) >= 2:
            working_schemas['pricing_analysis'] = {
                "name": "pricing_analysis",
                "fields": [
                    {
                        "id": price_fields[0]['id'],
                        "name": price_fields[0]['name'],
                        "category": "measure"
                    },
                    {
                        "id": price_fields[1]['id'],
                        "name": price_fields[1]['name'],
                        "category": "measure"
                    }
                ],
                "filter": {
                    "selections": [],
                    "limit": 100
                }
            }
            print(f"✅ Created pricing_analysis schema with {len(price_fields)} fields")
    
    # Schema 4: Promotional analysis
    if 'promo_camp' in available_tables:
        promo_fields = available_tables['promo_camp']
        if len(promo_fields) >= 2:
            working_schemas['promotional_analysis'] = {
                "name": "promotional_analysis",
                "fields": [
                    {
                        "id": promo_fields[0]['id'],
                        "name": promo_fields[0]['name'],
                        "category": "measure"
                    },
                    {
                        "id": promo_fields[1]['id'],
                        "name": promo_fields[1]['name'],
                        "category": "dimension"
                    }
                ],
                "filter": {
                    "selections": [],
                    "limit": 100
                }
            }
            print(f"✅ Created promotional_analysis schema with {len(promo_fields)} fields")
    
    print(f"Working schemas: {working_schemas}")
    return working_schemas

def test_working_schemas(working_schemas, components=None):
    """Test the working schemas using the new components"""
    print("\n=== TESTING WORKING SCHEMAS ===")
    
    results = {}
    
    # Use the new components if available, otherwise fall back to intugle_tools
    if components and 'data_product_builder' in components:
        dpb = components['data_product_builder']
        print("Using new DataProductBuilder from components")
    else:
        if not intugle_tools.is_available():
            print("❌ Intugle tools not available")
            return {}
        dpb = intugle_tools._components.get('data_product_builder')
        if not dpb:
            print("❌ DataProductBuilder not available")
            return {}
        print("Using DataProductBuilder from intugle_tools")
    
    for schema_name, schema_config in working_schemas.items():
        print(f"\n--- Testing Schema: {schema_name} ---")
        print(f"Schema: {json.dumps(schema_config, indent=2)}")
        
        try:
            # Test build_dataproduct
            build_result = dpb.build(schema_config)
            
            print(f"Build result type: {type(build_result)}")
            
            # Try to get dataframe if it's a DataSet object
            if hasattr(build_result, 'to_df'):
                try:
                    df = build_result.to_df()
                    print(f"✅ Build successful - DataFrame shape: {df.shape}")
                    print(f"DataFrame columns: {list(df.columns)}")
                    print(f"First few rows:\n{df.head()}")
                    results[schema_name] = {
                        "status": "success",
                        "result": f"DataFrame with {df.shape[0]} rows, {df.shape[1]} columns"
                    }
                except Exception as df_error:
                    print(f"⚠️ Build succeeded but failed to get DataFrame: {str(df_error)}")
                    results[schema_name] = {
                        "status": "partial_success",
                        "result": str(build_result),
                        "error": str(df_error)
                    }
            else:
                print(f"✅ Build successful - Result: {build_result}")
                results[schema_name] = {
                    "status": "success",
                    "result": str(build_result)
                }
                
        except Exception as e:
            print(f"❌ Exception during build: {str(e)}")
            results[schema_name] = {
                "status": "exception",
                "error": str(e)
            }
    
    return results

def create_correct_etl_schema_for_hypothesis(hypothesis, available_tables):
    """Create a correct ETL schema for a specific hypothesis"""
    print(f"\n=== CREATING CORRECT ETL SCHEMA FOR HYPOTHESIS ===")
    print(f"Hypothesis: {hypothesis.get('hypothesis', 'N/A')}")
    
    # Extract user requirements
    data_req = hypothesis.get("data_requirements", {})
    user_columns = data_req.get("required_columns", [])
    user_tables = data_req.get("required_tables", [])
    
    print(f"User requirements:")
    print(f"  Tables: {user_tables}")
    print(f"  Columns: {user_columns}")
    
    # Find matching fields
    matching_fields = []
    
    for user_col in user_columns:
        if '.' in user_col:
            user_table, user_field = user_col.split('.', 1)
        else:
            user_table = None
            user_field = user_col
        
        # Look for exact matches
        for table_name, fields in available_tables.items():
            for field in fields:
                if user_table and table_name.lower() == user_table.lower():
                    if user_field.lower() in field['column_name'].lower() or user_field.lower() in field['name'].lower():
                        matching_fields.append(field)
                        print(f"✅ Found match: {user_col} -> {field['id']}")
                        break
    
    if not matching_fields:
        print("❌ No matching fields found. Using fallback fields.")
        # Use any available fields as fallback
        for table_name, fields in available_tables.items():
            if fields:
                matching_fields.extend(fields[:2])  # Take first 2 fields
                break
    
    # Create ETL schema
    etl_schema = {
        "name": f"corrected_{hypothesis.get('hypothesis_id', 'analysis')}",
        "fields": [
            {
                "id": field['id'],
                "name": field['name'],
                "category": "measure" if field['datatype'] in ['number', 'integer', 'float'] else "dimension"
            }
            for field in matching_fields[:3]  # Limit to 3 fields
        ],
        "filter": {
            "selections": [],
            "limit": 100
        }
    }
    
    print(f"Created ETL schema: {json.dumps(etl_schema, indent=2)}")
    return etl_schema

def main():
    """Main test function"""
    print("FIXED ETL SCHEMA TEST WITH DATASET SETUP")
    print("=" * 60)
    
    # Step 1: Setup healthcare dataset
    print("\n" + "="*60)
    components = setup_healthcare_dataset()
    
    if not components:
        print("⚠️ Dataset setup failed, but continuing with existing KB structure")
    
    # Step 2: Test semantic model data
    print("\n" + "="*60)
    if components:
        data_ok = test_semantic_model_data(components)
        if not data_ok:
            print("⚠️ Component testing failed, but continuing with KB structure")
    else:
        print("⚠️ Skipping component testing, proceeding with KB structure")
    
    # Step 3: Get available tables and fields from KB
    print("\n" + "="*60)
    available_tables = get_available_tables_and_fields()
    
    if not available_tables:
        print("❌ No tables found. Cannot proceed.")
        return
    
    # Step 4: Create working schemas
    print("\n" + "="*60)
    working_schemas = create_working_etl_schemas(available_tables)
    
    if not working_schemas:
        print("❌ No working schemas could be created.")
        return
    
    # Step 5: Test working schemas
    print("\n" + "="*60)
    test_results = test_working_schemas(working_schemas, components)
    
    # Step 6: Test with a specific hypothesis
    print("\n" + "="*60)
    test_hypothesis = {
        "hypothesis_id": "test_hyp",
        "hypothesis": "Pricing plays a significant role in influencing the units sold",
        "data_requirements": {
            "required_tables": ["sales", "price_sense"],
            "required_columns": ["sales.vol", "price_sense.price_dec"]
        }
    }
    
    correct_schema = create_correct_etl_schema_for_hypothesis(test_hypothesis, available_tables)
    
    # Test the correct schema
    print(f"\n--- Testing Correct Schema ---")
    try:
        if components and 'data_product_builder' in components:
            dpb = components['data_product_builder']
            build_result = dpb.build(correct_schema)
            print(f"Correct schema build result type: {type(build_result)}")
            
            if hasattr(build_result, 'to_df'):
                try:
                    df = build_result.to_df()
                    print(f"✅ Correct schema build successful - DataFrame shape: {df.shape}")
                    print(f"DataFrame columns: {list(df.columns)}")
                    print(f"First few rows:\n{df.head()}")
                except Exception as df_error:
                    print(f"⚠️ Build succeeded but failed to get DataFrame: {str(df_error)}")
            else:
                print(f"✅ Correct schema build successful - Result: {build_result}")
        else:
            build_result = intugle_tools.build_dataproduct(correct_schema)
            print(f"Correct schema build result: {json.dumps(build_result, indent=2)}")
            
            if "error" in build_result:
                print(f"❌ Correct schema build failed: {build_result['error']}")
            else:
                print(f"✅ Correct schema build successful")
    except Exception as e:
        print(f"❌ Exception during correct schema build: {str(e)}")
    
    # Step 7: Analyze results
    print(f"\n=== FINAL ANALYSIS ===")
    successful = sum(1 for result in test_results.values() if result["status"] == "success")
    failed = sum(1 for result in test_results.values() if result["status"] != "success")
    
    print(f"Schema Test Results: {successful} successful, {failed} failed")
    
    if successful > 0:
        print("✅ Some schemas work! The issue is in the KB structure interpretation.")
        print("🔧 SOLUTION: The KB structure needs to be interpreted correctly:")
        print("  - Each KB entry is a field, not a table with fields")
        print("  - Group fields by table name (part before the dot)")
        print("  - Use the correct field IDs in ETL schemas")
        print("  - Ensure dataset is properly loaded before testing")
    else:
        print("❌ All schemas failed. Need to investigate further.")
        print("🔧 TROUBLESHOOTING:")
        print("  - Check if datasets are properly loaded")
        print("  - Verify KB structure is correct")
        print("  - Ensure Intugle tools are properly initialized")
    
    print("\n🎉 Fixed ETL schema testing with dataset setup completed!")

if __name__ == "__main__":
    main()
