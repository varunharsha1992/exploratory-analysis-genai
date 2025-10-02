"""
Column Mapping Fix for EDA Worker Loop

This module provides comprehensive fixes for the column mapping issues
between user hypotheses and KB/ETL schema generation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from EDA.tools.intugle_agent_tools import intugle_tools
import json
import re
from typing import Dict, List, Any, Optional, Tuple

class ColumnMappingFixer:
    """Fixes column mapping issues between user input and KB"""
    
    def __init__(self):
        self.kb_profiles = None
        self.available_tables = set()
        self.available_fields = {}
        self._load_kb_structure()
    
    def _load_kb_structure(self):
        """Load and analyze KB structure"""
        if intugle_tools.is_available():
            profiles = intugle_tools.get_variable_profiles()
            if "profiles" in profiles:
                self.kb_profiles = profiles["profiles"]
                self._analyze_kb_structure()
    
    def _analyze_kb_structure(self):
        """Analyze KB structure to identify available tables and fields"""
        print("=== ANALYZING KB STRUCTURE ===")
        
        for table_name, table_data in self.kb_profiles.items():
            self.available_tables.add(table_name)
            
            # Check if table has fields
            if 'fields' in table_data and table_data['fields']:
                self.available_fields[table_name] = table_data['fields']
                print(f"✅ Table {table_name}: {len(table_data['fields'])} fields")
            else:
                print(f"❌ Table {table_name}: No fields found")
        
        print(f"Total tables: {len(self.available_tables)}")
        print(f"Tables with fields: {len(self.available_fields)}")
    
    def find_matching_fields(self, user_columns: List[str]) -> Dict[str, List[Dict]]:
        """Find KB fields that match user input columns"""
        matches = {}
        
        for user_col in user_columns:
            matches[user_col] = []
            
            # Extract table and column from user input
            if '.' in user_col:
                user_table, user_field = user_col.split('.', 1)
            else:
                user_table = None
                user_field = user_col
            
            # Search for matches in KB
            for table_name, fields in self.available_fields.items():
                for field in fields:
                    field_id = field.get('id', '')
                    field_name = field.get('name', '')
                    field_desc = field.get('description', '')
                    
                    # Check for exact matches
                    if user_field.lower() in field_id.lower() or user_field.lower() in field_name.lower():
                        matches[user_col].append({
                            'table': table_name,
                            'field_id': field_id,
                            'field_name': field_name,
                            'description': field_desc,
                            'match_type': 'exact'
                        })
                    
                    # Check for semantic matches
                    elif self._semantic_match(user_field, field_name, field_desc):
                        matches[user_col].append({
                            'table': table_name,
                            'field_id': field_id,
                            'field_name': field_name,
                            'description': field_desc,
                            'match_type': 'semantic'
                        })
        
        return matches
    
    def _semantic_match(self, user_field: str, kb_name: str, kb_desc: str) -> bool:
        """Check for semantic matches between user input and KB fields"""
        user_lower = user_field.lower()
        kb_name_lower = kb_name.lower()
        kb_desc_lower = kb_desc.lower()
        
        # Common semantic mappings
        semantic_mappings = {
            'sales_volume': ['vol', 'volume', 'units', 'quantity'],
            'discount_amount': ['discount', 'promo', 'cost', 'price'],
            'sales_amount': ['amt', 'amount', 'revenue', 'sales'],
            'product_id': ['pid', 'prod_id', 'product'],
            'date': ['date', 'time', 'period']
        }
        
        for key, values in semantic_mappings.items():
            if user_lower in key or any(val in user_lower for val in values):
                if any(val in kb_name_lower or val in kb_desc_lower for val in values):
                    return True
        
        return False
    
    def create_corrected_etl_schema(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """Create corrected ETL schema using proper column mapping"""
        print("\n=== CREATING CORRECTED ETL SCHEMA ===")
        
        # Extract user requirements
        data_req = hypothesis.get("data_requirements", {})
        user_columns = data_req.get("required_columns", [])
        user_tables = data_req.get("required_tables", [])
        
        print(f"User columns: {user_columns}")
        print(f"User tables: {user_tables}")
        
        # Find matching fields
        field_matches = self.find_matching_fields(user_columns)
        
        # Create corrected schema
        corrected_schema = {
            "name": f"corrected_{hypothesis.get('hypothesis_id', 'analysis')}",
            "fields": [],
            "filter": {
                "selections": [],
                "limit": 1000
            }
        }
        
        # Map user columns to KB fields
        for user_col, matches in field_matches.items():
            if matches:
                # Use the best match
                best_match = matches[0]
                corrected_schema["fields"].append({
                    "id": best_match["field_id"],
                    "name": best_match["field_name"],
                    "category": "measure" if "amount" in best_match["field_name"].lower() or "vol" in best_match["field_id"].lower() else "dimension"
                })
                print(f"✅ Mapped {user_col} -> {best_match['field_id']}")
            else:
                print(f"❌ No match found for {user_col}")
        
        return corrected_schema
    
    def debug_column_mapping_issues(self, hypothesis: Dict[str, Any]):
        """Comprehensive debugging of column mapping issues"""
        print("\n=== COMPREHENSIVE COLUMN MAPPING DEBUG ===")
        
        # Show KB structure
        print("\n--- KB Structure Analysis ---")
        if self.available_fields:
            for table_name, fields in self.available_fields.items():
                print(f"\nTable: {table_name}")
                for field in fields:
                    print(f"  - {field.get('id', 'unknown')}: {field.get('description', 'no description')}")
        else:
            print("❌ No fields found in KB")
        
        # Show user requirements
        print("\n--- User Requirements ---")
        data_req = hypothesis.get("data_requirements", {})
        print(f"Required tables: {data_req.get('required_tables', [])}")
        print(f"Required columns: {data_req.get('required_columns', [])}")
        
        # Find matches
        print("\n--- Column Matching Analysis ---")
        user_columns = data_req.get("required_columns", [])
        field_matches = self.find_matching_fields(user_columns)
        
        for user_col, matches in field_matches.items():
            print(f"\nUser column: {user_col}")
            if matches:
                for match in matches:
                    print(f"  ✅ {match['match_type']}: {match['field_id']} - {match['description']}")
            else:
                print(f"  ❌ No matches found")
        
        # Create corrected schema
        print("\n--- Creating Corrected Schema ---")
        corrected_schema = self.create_corrected_etl_schema(hypothesis)
        print(f"Corrected schema: {json.dumps(corrected_schema, indent=2)}")
        
        return corrected_schema

def test_column_mapping_fix():
    """Test the column mapping fix"""
    print("=== TESTING COLUMN MAPPING FIX ===")
    
    # Create test hypothesis
    test_hypothesis = {
        "hypothesis_id": "test_hyp",
        "hypothesis": "Test hypothesis for column mapping",
        "data_requirements": {
            "required_tables": ["promotions", "sales_data"],
            "required_columns": ["promotions.discount_amount", "sales_data.sales_volume"]
        }
    }
    
    # Initialize fixer
    fixer = ColumnMappingFixer()
    
    # Debug issues
    corrected_schema = fixer.debug_column_mapping_issues(test_hypothesis)
    
    return corrected_schema

if __name__ == "__main__":
    test_column_mapping_fix()
