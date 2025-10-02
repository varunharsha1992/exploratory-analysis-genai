"""
Simplified Intugle Agent Tools - Core tools for agents to use with cached Intugle components
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.setup_intugle import get_cached_intugle_components, create_intugle_agent_integration
from EDA.LLMS.llms import get_llm

class IntugleAgentTools:
    """Simplified tools for agents to interact with cached Intugle components"""
    
    def __init__(self, full_data_path: str = "", files_to_process: list = []):
        self._integration = None
        self._components = None
        self._llm = None
        self._initialize(full_data_path,files_to_process)
    
    def _initialize(self, full_data_path: str = "", files_to_process: list = []):
        """Initialize components with fresh data loading"""
        try:
            # Always load fresh data instead of using cache
            from utils.setup_intugle import setup_intugle_with_real_data
            
            # Setup with fresh data
            self._integration = setup_intugle_with_real_data(sales_data_path=full_data_path,files_to_process=files_to_process)
            if self._integration:
                self._components = {
                    'knowledge_builder': self._integration.knowledge_builder,
                    'data_product_builder': self._integration.data_product_builder,
                    'cache_dir': 'fresh_data',
                    'is_cached': {'knowledge_builder': False, 'data_product_builder': False}
                }
                self._llm = get_llm("openai", "gpt-4o-mini")
                print("✅ Intugle tools initialized with fresh data")
            else:
                print("❌ Failed to initialize Intugle with fresh data")
        except Exception as e:
            print(f"Failed to initialize Intugle agent tools: {e}")
            # Fallback to cached components if fresh data fails
            try:
                self._components = get_cached_intugle_components()
                if self._components:
                    self._integration = create_intugle_agent_integration()
                    self._llm = get_llm("openai", "gpt-4o-mini")
                    print("⚠️ Using cached components as fallback")
            except Exception as fallback_error:
                print(f"Fallback to cached components also failed: {fallback_error}")
    
    def is_available(self) -> bool:
        """Check if Intugle tools are available"""
        return self._integration is not None and self._components is not None
    
    def get_variable_profiles(self, table_name: str = None, variable_name: str = None) -> Dict[str, Any]:
        """
        Get variable profiles from knowledge base
        
        Args:
            table_name: Specific table to filter by (None for all tables)
            variable_name: Specific variable to filter by (None for all variables in table)
        
        Returns:
            Dictionary with variable profiles
        """
        if not self.is_available():
            return {"error": "Intugle tools not available", "profiles": {}}
        
        return self._integration.get_variable_profiles(table_name, variable_name)
    
    def search_variables(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Search for variables using semantic search
        
        Args:
            query: Search query for variables
            max_results: Maximum number of results to return
        
        Returns:
            Dictionary with search results
        """
        if not self.is_available():
            return {"error": "Intugle tools not available", "variables": []}
        
        try:
            search_results = self._integration.search_variables(query, max_results)
            
            return {
                "query": query,
                "variables": [
                    {
                        "variable_id": result.get("variable_id", ""),
                        "similarity_score": result.get("similarity_score", 0.0),
                        "description": result.get("description", ""),
                        "category": result.get("category", ""),
                        "table_name": result.get("table_name", ""),
                        "column_name": result.get("column_name", "")
                    }
                    for result in search_results
                ],
                "total_results": len(search_results),
                "status": "success"
            }
            
        except Exception as e:
            return {"error": str(e), "variables": []}
    
    def create_etl_schema_with_llm(self, user_input: str) -> Dict[str, Any]:
        """
        Create ETL schema using LLM based on user input and available fields from KB
        
        Args:
            user_input: User's natural language input describing what they want to analyze
        
        Returns:
            Dictionary with generated ETL schema
        """
        if not self.is_available() or not self._llm:
            return {"error": "Intugle tools or LLM not available", "etl_schema": {}}
        
        try:
            # Get all variable profiles for LLM context
            kb_profiles = self.get_variable_profiles()
            
            if "error" in kb_profiles:
                return {"error": "Failed to load KB profiles", "etl_schema": {}}
            
            # Create prompt for LLM based on Intugle's create_dp_prompt.md
            prompt = """
You are an expert data engineer creating ETL schemas for Intugle DataProduct. Based on the user's input, create a matching ETL schema using available fields from the knowledge base.

User Input: {user_input}

Available Fields from Knowledge Base:
{knowledge_base_profiles}

TASKS:
1. Create a valid ETL schema based on the user's input and available fields from the knowledge base.
2. Refer to target variable and predictor variables [can be multiple predictor variables] from the user input and match them to the fields from the knowledge base to create the ETL schema
3. If there are aggregation fields present in the user input, match them to the fields from the knowledge base to create the ETL schema and include them in the ETL schema. Aggregation fields have alias as the key and the field name as the value.
2. Return the ETL schema in the format specified in the examples below.
3. Return only a valid JSON object following the GOOD examples below.
4. Include the measure function for the target and predictor variables in the ETL schema. Don't have measure function for strings. Default to sum incase measure function is not present for a numeric field.

## STRICT RULES - MUST FOLLOW:

1. **NO JOINS**: Never include "join" field in the schema. Use only single table analysis.
2. **NO COMPLEX SORTING**: Avoid sort_by that causes SQL GROUP BY conflicts. Only sort by fields that are in GROUP BY.
3. **SIMPLE SCHEMAS**: Keep schemas simple and focused on single table analysis.
4. **PROPER CATEGORIES**: Use "measure" for numeric fields, "dimension" for categorical fields.
5. **LIMIT ROWS**: Always include a reasonable limit (100-1000 rows).
6. **ALL THE FIELD IDs MUST BE FROM THE knowledge base**. Never make up fields and table names or use the field names from the user input. Match the closest field name from the knowledge base.
7. **THE FIELD NAMES MUST BE THE SAME AS THE USER INPUT**. Never make up field names or use the field names from the knowledge base. If there is an alias present in the user input, prioritize the alias as the field name.
8. **NEVER SKIP AGGREGATION FIELDS IF PRESENT IN THE USER INPUT, ESPECIALLY FOR DATE FIELDS AS THEY ARE IMPORTANT FOR TIME SERIES ANALYSIS**.

## WHAT TO DO:
✅ Use fields from the same table
✅ Include target and predictor variables
✅ Use simple filters and limits
✅ Focus on correlation analysis
✅ Keep schemas minimal and functional
✅ Include aggregation fields if present in the user input. The schema of aggregation fields from user input is typically in the form of a dictionary with alias as the key and the field id as the value.


## WHAT NOT TO DO:
❌ Never use "join" field
❌ Never sort by fields not in GROUP BY
❌ Never create complex multi-table schemas
❌ Never use complex aggregations with sorting conflicts
❌ Never exceed 1000 rows limit
❌ Never skip aggregation fields if present in the user input, especially for date fields as they are important for time series analysis.
❌ Never have dots or spaces in the name


## SCHEMA STRUCTURE (glossary):
{{
  "name": "descriptive_snake_case_name",
  "fields": [
    {{"id": "table.field", "name": "variable alias present as mention in user input", "category": "measure|dimension", "measure_func": "count|sum|average|median|mode|min|max"}}, (id: the field id from the knowledge base, name: the alias present as mention in user input, category: measure or dimension, measure_func: the measure function for the field)
    {{"id": "table.field", "name": "variable alias present as mention in user input", "category": "measure|dimension", "measure_func": "count|sum|average|median|mode|min|max"}}
  ],
  "filter": {{
    "selections": [],
    "limit": 1000
  }}
}}

## EXAMPLES:

### ✅ GOOD EXAMPLE - Simple Analysis:
user input: "I need a list of the 10 highest sold units from Boston.
Target Name Alias: sales_volume
Target Name Alias: sales_volume
Aggregation Fields: {{"product_category":"sales.Product_Category", "city":"sales.city"}}
KB PROFILES:
{{
  "name": "sales",
  "fields": [
    {{"id": "sales.Units_Sold", "datatype": "number", "measure_func": "sum"}},
    {{"id": "sales.Product_Category", "datatype: "string"}},
    {{"id": "sales.city", "datatype": "string"}}
  ]
}}
ETL SCHEMA:{{
  "name": "units_sold_analysis",
  "fields": [
    {{"id": "sales.Units_Sold", "name": "sales_volume", "category": "measure"}}, # Ensure the field ids match the knowledge base and name matches the user input
    {{"id": "sales.Product_Category", "name": "product_category", "category": "dimension"}} # As this is an aggregation field, the name exactly matches the key of the aggregation field.
    {{"id": "sales.city", "name": "city", "category": "dimension"}}
  ],
  "filter": {{
    "selections": [],
    "limit": 1000
  }}
}}

### ✅ GOOD EXAMPLE - With Filter:
{{
  "name": "marketing_spend_analysis",
  "fields": [
    {{"id": "marketing.Marketing_Spend", "name": "marketing_spend", "category": "measure"}}, # Ensure the field ids match the knowledge base and name matches the user input
    {{"id": "marketing.Campaign_Type", "name": "campaign_type", "category": "dimension"}}
  ],
  "filter": {{
    "selections": [
      {{"id": "marketing.Campaign_Type", "values": ["Digital", "TV"]}}
    ],
    "limit": 500
  }}
}}

### ✅ GOOD EXAMPLE - With Aggregation Fields:
User Input: I need a list of the 10 highest sold units from Boston.
Target Name Alias: marketing.marketing_spend
Target Name Alias: marketing_spend
Aggregation Fields: {{"campaign_type":"marketing.Campaign_Type", "date_sale":"marketing.Date_Sale"}}
KB PROFILES:
{{
  "name": "marketing",
  "fields": [
    {{"id": "marketing.Marketing_Spend", "name": "marketing_spend", "datatype": "number"}},
    {{"id": "marketing.Campaign_Type", "name": "campaign_type", "datatype: "string"}}
    {{"id": "marketing.Date_Sale", "name": "date_sale", "datatype": "date"}}
  ]
}}
ETL SCHEMA:
{{
  "name": "marketing_spend_analysis",
  "fields": [
    {{"id": "marketing.Marketing_Spend", "name": "marketing_spend", "category": "measure"}}, # Ensure the field ids match the knowledge base and name matches the user input
    {{"id": "marketing.Campaign_Type", "name": "campaign_type", "category": "dimension"}}
    {{"id": "marketing.Date_Sale", "name": "date_sale", "category": "dimension"}}
  ],
  "filter": {{
    "selections": [],
    "limit": 1000
  }}
}}
### ❌ BAD EXAMPLE - With Joins (DON'T DO THIS):
{{
  "name": "complex_analysis",
  "fields": [...],
  "join": {{"left_table": "sales", "right_table": "marketing"}}  // ❌ NEVER USE JOINS
}}

### ❌ BAD EXAMPLE - Complex Sorting (DON'T DO THIS):
{{
  "name": "problematic_analysis",
  "fields": [...],
  "filter": {{
    "sort_by": [{{"id": "sales.Units_Sold", "direction": "desc"}}]  // ❌ CAUSES SQL CONFLICTS
  }}
}}
## ✅GOOD EXAMPLE - With FILED NAMES FROM THE KNOWLEDGE BASE NOT MATCHING USER INPUT:
USER INPUT: "I need a list of the 10 highest sold units from Boston.
Column Names: units_sold"
Aggregation Fields: {{"date_sale":"sales.date","product_category":"sales.Product_Category", "city":"sales.city"}}
KB PROFILES:
{{
  "name": "sales",
  "fields": [
    {{"id": "sales.vol", "name": "units_sold", "datatype": "number"}},
    {{"id": "sales.Product_Category", "name": "product_category", "datatype: "string"}},
    {{"id": "sales.city", "name": "city", "datatype": "string"}},
    {{"id": "sales.date", "name": "date_sale", "datatype": "date"}}
  ]
}}
OUTPUT (ETL SCHEMA THAT MATCHES KB PROFILES):
{{
  "name": "sales_volume_analysis",
  "fields": [
    {{"id": "sales.vol", "name": "units_sold", "category": "measure"}}, # Ensure the field ids match the knowledge base and name matches the user input
    {{"id": "sales.Product_Category", "name": "product_category", "category": "dimension"}}
  ],
  "filter": {{
    "selections": [{{"id": "sales.city", "values": ["Boston"]}}],
    "limit": 1000
  }}
}}


## ❌ BAD EXAMPLE - With FILED NAMES FROM THE KNOWLEDGE BASE NOT MATCHING USER INPUT - NEVER DO THIS:
USER INPUT: "I need a list of the 10 highest sold units from Boston.
Column Names: units_sold"
KB PROFILES:
{{
  "name": "sales",
  "fields": [
    {{"id": "sales.vol", "datatype": "number"}},
    {{"id": "sales.Product_Category", "datatype: "string"}},
    {{"id": "sales.city", "datatype": "string"}}
  ]
}}
OUTPUT (ETL SCHEMA THAT MATCHES KB PROFILES):
{{
  "name": "sales_volume_analysis",
  "fields": [
    {{"id": "sales_units", "name": "units_sold", "category": "measure"}},
    {{"id": "sales.Product_Category", "name": "product_category", "category": "dimension"}}
  ],
  "filter": {{
    "selections": [{{"id": "city_name", "values": ["Boston"]}}],
    "limit": 1000
  }}
}}

Return only a valid JSON object following the GOOD examples above.

"""
            
            # Format the prompt with user input and KB profiles
            formatted_prompt = prompt.format(
                user_input=user_input,
                knowledge_base_profiles=json.dumps(kb_profiles["profiles"], indent=2)
            )
            
            # Get LLM response
            response = self._llm.invoke(formatted_prompt)
            
            # Extract JSON from markdown code blocks if present
            content = response.content.strip()
            if content.startswith('```json') and content.endswith('```'):
                # Remove markdown code block formatting
                content = content[7:-3].strip()  # Remove ```json and ```
            elif content.startswith('```') and content.endswith('```'):
                # Remove generic code block formatting
                content = content[3:-3].strip()  # Remove ``` and ```
            
            etl_schema = json.loads(content)
            
            return {
                "user_input": user_input,
                "etl_schema": etl_schema,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": str(e), "etl_schema": {}}
    
    def build_dataproduct(self, data_product_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build DataProduct using the provided configuration
        
        Args:
            data_product_config: DataProduct configuration following Intugle format
        
        Returns:
            Dictionary with build results
        """
        if not self.is_available():
            return {"error": "Intugle tools not available", "build_result": {}}
        
        try:
            if not self._components.get('data_product_builder'):
                return {"error": "DataProductBuilder not available", "build_result": {}}
            
            dpb = self._components['data_product_builder']
            
            try:
                build_result = dpb.build(data_product_config)
                print(f"Query Generated: {build_result.sql_query}")
                return {
                    "data_product_config": data_product_config,
                    "build_result": build_result,
                    "status": "success"
                }
            except Exception as build_error:
                print(f"Build error: {build_error}")
                return {
                    "data_product_config": data_product_config,
                    "build_result": {
                        "status": "error",
                        "error": str(build_error)
                    },
                    "status": "failed"
                }
            
        except Exception as e:
            return {"error": str(e), "build_result": {}}

# Convenience functions for direct use
def get_variable_profiles(table_name: str = None, variable_name: str = None) -> Dict[str, Any]:
    """Get variable profiles from knowledge base"""
    return IntugleAgentTools().get_variable_profiles(table_name, variable_name)

def search_variables(query: str, max_results: int = 10) -> Dict[str, Any]:
    """Search for variables using semantic search"""
    return search_variables(query, max_results)

def create_etl_schema_with_llm(user_input: str) -> Dict[str, Any]:
    """Create ETL schema using LLM based on user input and available fields from KB"""
    return create_etl_schema_with_llm(user_input)

def build_dataproduct(data_product_config: Dict[str, Any]) -> Dict[str, Any]:
    """Build DataProduct using the provided configuration"""
    return build_dataproduct(data_product_config)

def is_intugle_available() -> bool:
    """Check if Intugle tools are available"""
    return is_available()
