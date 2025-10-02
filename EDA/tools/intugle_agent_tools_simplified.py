import json
import logging
from typing import Dict, Any, Optional, List
from utils.setup_intugle import setup_intugle_with_real_data, get_cached_intugle_components
from LLMS.llms import get_llm
from utils.config_loader import AgentConfigLoader

class IntugleAgentTools:
    def __init__(self):
        self.config_loader = AgentConfigLoader()
        self._components = None
        self._llm = None
        self._initialized = False
    
    def _initialize(self):
        """Initialize Intugle components and LLM"""
        if self._initialized:
            return
        
        try:
            # Always load fresh data
            self._components = setup_intugle_with_real_data()
            self._initialized = True
            logging.info("✅ IntugleAgentTools initialized with fresh data")
        except Exception as e:
            logging.warning(f"Failed to load fresh data, using cached components: {e}")
            # Fallback to cached components
            self._components = get_cached_intugle_components()
            self._initialized = True
    
    def get_variable_profiles(self) -> Dict[str, Any]:
        """Get variable profiles from the knowledge base"""
        self._initialize()
        
        try:
            if not self._components or "kb" not in self._components:
                return {"error": "Knowledge base not available"}
            
            kb = self._components["kb"]
            profiles = kb.get_profiles()
            return {"profiles": profiles}
        except Exception as e:
            logging.error(f"Failed to get variable profiles: {e}")
            return {"error": str(e)}
    
    def search_variables(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search for variables in the knowledge base"""
        self._initialize()
        
        try:
            if not self._components or "kb" not in self._components:
                return {"error": "Knowledge base not available"}
            
            kb = self._components["kb"]
            results = kb.search(query, limit=limit)
            return {"results": results}
        except Exception as e:
            logging.error(f"Failed to search variables: {e}")
            return {"error": str(e)}
    
    def create_etl_schema_with_llm(self, user_input: str) -> Dict[str, Any]:
        """Create ETL schema using LLM based on user input and KB profiles"""
        self._initialize()
        
        try:
            if not self._components or "kb" not in self._components:
                return {"error": "Knowledge base not available"}
            
            # Get KB profiles
            kb_profiles = self.get_variable_profiles()
            
            if "error" in kb_profiles:
                return {"error": "Failed to load KB profiles", "etl_schema": {}}
            
            # Create simplified prompt for LLM
            prompt = """
You are an expert data engineer creating ETL schemas for Intugle DataProduct.

User Input: {user_input}

Available Fields from Knowledge Base:
{kb_profiles}

## RULES:
1. Use ONLY fields from the knowledge base
2. NO joins - single table analysis only
3. Keep schemas simple
4. Use "measure" for numeric fields, "dimension" for categorical fields
5. Limit to 1000 rows maximum

## SCHEMA FORMAT:
{
  "name": "analysis_name",
  "fields": [
    {"id": "table.field", "name": "alias", "category": "measure|dimension"}
  ],
  "filter": {
    "selections": [],
    "limit": 1000
  }
}

## EXAMPLE:
{
  "name": "sales_analysis",
  "fields": [
    {"id": "sales.vol", "name": "sales_volume", "category": "measure"},
    {"id": "sales.Product_Category", "name": "product_category", "category": "dimension"}
  ],
  "filter": {
    "selections": [],
    "limit": 1000
  }
}

Return only a valid JSON object.
"""
            
            # Format the prompt
            formatted_prompt = prompt.format(
                user_input=user_input,
                kb_profiles=json.dumps(kb_profiles["profiles"], indent=2)
            )
            
            # Get LLM response
            if not self._llm:
                model_config = self.config_loader.get_model_config("eda_analysis")
                self._llm = get_llm(model_config['provider'], model_config['model'])
            
            response = self._llm.invoke(formatted_prompt)
            
            # Parse response
            try:
                etl_schema = json.loads(response.content)
                return {"etl_schema": etl_schema}
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse LLM response as JSON: {e}")
                return {"error": f"Invalid JSON response: {response.content}", "etl_schema": {}}
                
        except Exception as e:
            logging.error(f"ETL schema creation failed: {e}")
            return {"error": str(e), "etl_schema": {}}
    
    def build_dataproduct(self, etl_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Build data product using ETL schema"""
        self._initialize()
        
        try:
            if not self._components or "dp_builder" not in self._components:
                return {"error": "Data product builder not available"}
            
            dp_builder = self._components["dp_builder"]
            result = dp_builder.build(etl_schema)
            
            return {"status": "success", "result": result}
        except Exception as e:
            logging.error(f"Data product build failed: {e}")
            return {"error": str(e), "status": "failed"}

# Create global instance
intugle_tools = IntugleAgentTools()
