"""
Simplified Intugle Setup Script

This script provides a simplified setup and integration for Intugle Knowledge Base and Semantic Search
for the Data Querying AI project.

Usage:
    python setup_intugle.py
"""

import os
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import json
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

# Add Intugle to path if needed
intugle_path = os.path.join(os.path.dirname(__file__), '..', 'Intugle', 'data-tools', 'src')
if intugle_path not in sys.path:
    sys.path.append(intugle_path)

try:
    from intugle import KnowledgeBuilder
    from intugle.dp_builder import DataProductBuilder
    INTRUGLE_AVAILABLE = True
except ImportError as e:
    print(f"Intugle not available: {e}")
    INTRUGLE_AVAILABLE = False

def setup_environment():
    """Setup environment variables"""
    # Set default environment variables if not set
    if not os.getenv("LLM_PROVIDER"):
        os.environ["LLM_PROVIDER"] = "openai:gpt-4o-mini"
    
    if not os.getenv("QDRANT_URL"):
        os.environ["QDRANT_URL"] = "http://localhost:6333"
    
    if not os.getenv("EMBEDDING_MODEL_NAME"):
        os.environ["EMBEDDING_MODEL_NAME"] = "openai:text-embedding-3-small"
    
    # Fix Qdrant collection name issue
    if not os.getenv("VECTOR_COLLECTION_NAME"):
        cwd_name = os.getcwd().split(os.sep)[-1]
        safe_name = cwd_name.replace(" ", "_").replace(":", "_").lower()
        os.environ["VECTOR_COLLECTION_NAME"] = safe_name


def load_sales_forecast_data(data_path: str = None, files_to_process: list = []) -> Dict[str, Any]:
    """Load sales forecast data from CSV files"""
    if data_path is None:
        # Use the project's data directory
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data-tools', 'sample_data', 'fmcg')
    
    data_input = {}
    
    csv_files = files_to_process if files_to_process else [
"ad_spends",
"comp_price",
"cons_fb",
"disti",
"mkt_coverage",
"prd_mstr",
"price_sense",
"pric_strat",
"retail_audit",
"rnd",
"sales",
"social"
    ]
    
    print(f"Loading data from: {data_path}")
    
    for csv_file in csv_files:
        file_path = os.path.join(data_path, f"{csv_file}.csv")
        if os.path.exists(file_path):
            data_input[csv_file] = {
                "path": file_path,
                "type": "csv"
            }
            print(f"✓ Loaded {csv_file}")
        else:
            print(f"✗ File not found: {file_path}")
    
    return data_input

class IntugleSetup:
    """Simplified Intugle setup manager"""
    
    def __init__(self, project_base: str = None):
        self.project_base = Path(project_base) if project_base else Path.cwd()
        self.models_dir = self.project_base / "models"
        self.cache_dir = self.models_dir / "intugle_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.knowledge_builder = None
        self.data_product_builder = None
        
    def setup_environment(self) -> bool:
        """Setup environment variables"""
        try:
            setup_environment()
            self.models_dir.mkdir(exist_ok=True)
            return True
        except Exception as e:
            print(f"Environment setup failed: {e}")
            return False
    
    def initialize_knowledge_builder(self, data_input: Union[Dict[str, Any], List[Any]], force_rebuild: bool = False) -> bool:
        """Initialize Knowledge Builder with data"""
        try:
            if not INTRUGLE_AVAILABLE:
                raise ImportError("Intugle is not available")
            
            # Try to load from cache first
            if not force_rebuild and self.load_knowledge_builder():
                return True
            
            # Initialize KnowledgeBuilder
            self.knowledge_builder = KnowledgeBuilder(data_input)
            self.knowledge_builder.build()
            
            # Save to cache
            self.save_knowledge_builder()
            return True
            
        except Exception as e:
            print(f"Knowledge Builder initialization failed: {e}")
            return False
    
    def initialize_data_product_builder(self, force_rebuild: bool = False) -> bool:
        """Initialize Data Product Builder"""
        try:
            if not INTRUGLE_AVAILABLE:
                raise ImportError("Intugle is not available")
            
            # Try to load from cache first
            if not force_rebuild and self.load_data_product_builder():
                return True
            
            # Initialize DataProductBuilder with the same data input as KnowledgeBuilder
            # This is the key fix - DPB needs to be initialized with data sources
            if not self.knowledge_builder:
                raise Exception("KnowledgeBuilder must be initialized first")
            
            # Initialize DataProductBuilder with the same data input
            self.data_product_builder = DataProductBuilder()
            
            # Save to cache
            self.save_data_product_builder()
            return True
            
        except Exception as e:
            print(f"Data Product Builder initialization failed: {e}")
            return False
    
    def save_knowledge_builder(self) -> bool:
        """Save KnowledgeBuilder to disk"""
        try:
            if not self.knowledge_builder:
                return False
            
            import pickle
            kb_cache_file = self.cache_dir / "knowledge_builder.pkl"
            
            with open(kb_cache_file, 'wb') as f:
                pickle.dump(self.knowledge_builder, f)
            
            # Save metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "datasets_count": len(self.knowledge_builder.datasets),
                "variables_count": sum(len(dataset.source_table_model.columns) for dataset in self.knowledge_builder.datasets.values())
            }
            
            metadata_file = self.cache_dir / "kb_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Failed to save KnowledgeBuilder: {e}")
            return False
    
    def load_knowledge_builder(self) -> bool:
        """Load KnowledgeBuilder from disk"""
        try:
            kb_cache_file = self.cache_dir / "knowledge_builder.pkl"
            
            if not kb_cache_file.exists():
                return False
            
            import pickle
            with open(kb_cache_file, 'rb') as f:
                self.knowledge_builder = pickle.load(f)
            
            return True
            
        except Exception as e:
            print(f"Failed to load KnowledgeBuilder: {e}")
            return False
    
    def save_data_product_builder(self) -> bool:
        """Save DataProductBuilder to disk"""
        try:
            if not self.data_product_builder:
                return False
            
            import pickle
            dpb_cache_file = self.cache_dir / "data_product_builder.pkl"
            
            with open(dpb_cache_file, 'wb') as f:
                pickle.dump(self.data_product_builder, f)
            
            return True
            
        except Exception as e:
            print(f"Failed to save DataProductBuilder: {e}")
            return False
    
    def load_data_product_builder(self) -> bool:
        """Load DataProductBuilder from disk"""
        try:
            dpb_cache_file = self.cache_dir / "data_product_builder.pkl"
            
            if not dpb_cache_file.exists():
                return False
            
            import pickle
            with open(dpb_cache_file, 'rb') as f:
                self.data_product_builder = pickle.load(f)
            
            return True
            
        except Exception as e:
            print(f"Failed to load DataProductBuilder: {e}")
            return False
    
    def get_cached_components(self) -> Dict[str, Any]:
        """Get cached components for agent use"""
        return {
            "knowledge_builder": self.knowledge_builder,
            "data_product_builder": self.data_product_builder,
            "cache_dir": str(self.cache_dir),
            "is_cached": {
                "knowledge_builder": (self.cache_dir / "knowledge_builder.pkl").exists(),
                "data_product_builder": (self.cache_dir / "data_product_builder.pkl").exists()
            }
        }

class IntugleKnowledgeIntegration:
    """Integration layer for Intugle Knowledge Builder"""
    
    def __init__(self, intugle_setup: IntugleSetup):
        self.setup = intugle_setup
        self.knowledge_builder = intugle_setup.knowledge_builder
        self.data_product_builder = intugle_setup.data_product_builder
    
    def get_available_variables(self) -> List[str]:
        """Get list of all available variables"""
        if not self.knowledge_builder:
            return []
        
        variables = []
        try:
            for dataset in self.knowledge_builder.datasets.values():
                for column in dataset.source_table_model.columns:
                    variable_id = f"{dataset.name}.{column.name}"
                    variables.append(variable_id)
        except Exception as e:
            print(f"Error getting available variables: {e}")
        
        return variables
    
    def search_variables(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for variables using semantic search"""
        if not self.knowledge_builder:
            return []
        
        try:
            # Use KnowledgeBuilder's built-in search
            search_results = self.knowledge_builder.search(query)
            
            variables = []
            for _, row in search_results.head(limit).iterrows():
                variables.append({
                    "variable_id": row.get("column_id", ""),
                    "table_name": row.get("table_name", ""),
                    "column_name": row.get("column_name", ""),
                    "similarity_score": row.get("score", 0.0),
                    "description": row.get("column_glossary", ""),
                    "tags": row.get("column_tags", []),
                    "category": row.get("category", "")
                })
            
            return variables
            
        except Exception as e:
            print(f"Semantic search failed: {e}")
            return []
    
    def get_variable_details(self, variable_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific variable"""
        if not self.knowledge_builder:
            return {}
        
        try:
            # Parse table and column from variable_id
            if '.' not in variable_id:
                return {}
            
            table_name, column_name = variable_id.split('.', 1)
            
            # Find dataset
            dataset = self.knowledge_builder.datasets.get(table_name)
            if not dataset:
                return {}
            
            # Find column in the dataset's source_table_model
            column = None
            for col in dataset.source_table_model.columns:
                if col.name == column_name:
                    column = col
                    break
            
            if not column:
                return {}
            
            # Extract profiling metrics if available
            profiling_metrics = {}
            if hasattr(column, 'profiling_metrics') and column.profiling_metrics:
                profiling_metrics = column.profiling_metrics.model_dump() if hasattr(column.profiling_metrics, 'model_dump') else {}
            
            return {
                "variable_id": variable_id,
                "table_name": table_name,
                "column_name": column_name,
                "data_type": getattr(column, 'type', 'unknown'),
                "category": getattr(column, 'category', 'unknown'),
                "description": getattr(column, 'description', ''),
                "tags": getattr(column, 'tags', []),
                "profiling_metrics": profiling_metrics,
                "is_pii": getattr(column, 'is_pii', False)
            }
            
        except Exception as e:
            print(f"Error getting variable details for {variable_id}: {e}")
            return {}
    
    def get_variable_profiles(self, table_name: str = None, variable_name: str = None) -> Dict[str, Any]:
        """
        Get variable profiles from knowledge base
        
        Args:
            table_name: Specific table to filter by (None for all tables)
            variable_name: Specific variable to filter by (None for all variables in table)
        
        Returns:
            Dictionary with variable profiles
        """
        if not self.knowledge_builder:
            return {"error": "Knowledge builder not available", "profiles": {}}
        
        try:
            profiles = {}
            
            # If table_name is None, get all available datasets
            if table_name is None:
                available_datasets = list(self.knowledge_builder.datasets.keys())
            else:
                # If table_name is specified, only process that table
                if table_name not in self.knowledge_builder.datasets:
                    return {"error": f"Table '{table_name}' not found", "profiles": {}}
                available_datasets = [table_name]
            
            # For each dataset, get the profiling_df
            for dataset_name in available_datasets:
                dataset = self.knowledge_builder.datasets[dataset_name]
                profiling_df = dataset.profiling_df
                
                if profiling_df is not None and not profiling_df.empty:
                    # If variable_name is specified, filter for that variable
                    if variable_name is not None:
                        # Find rows where column_name matches the variable_name
                        matching_rows = profiling_df[profiling_df['column_name'] == variable_name]
                        if not matching_rows.empty:
                            var_id = f"{dataset_name}.{variable_name}"
                            profiles[var_id] = matching_rows.iloc[0].to_dict()
                    else:
                        # Get all variables in the dataset
                        for idx, row in profiling_df.iterrows():
                            var_name = row['column_name']
                            var_id = f"{dataset_name}.{var_name}"
                            profiles[var_id] = row.to_dict()
            
            return {
                "profiles": profiles,
                "total_count": len(profiles),
                "table_filter": table_name,
                "variable_filter": variable_name,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": str(e), "profiles": {}}
    
    def get_related_variables(self, target_variable: str, max_results: int = 10, min_similarity: float = 0.3) -> List[Dict[str, Any]]:
        """Get variables semantically related to target variable"""
        try:
            related_vars = self.search_variables(target_variable, max_results * 2)
            
            filtered_vars = [
                var for var in related_vars 
                if var["variable_id"] != target_variable and var["similarity_score"] >= min_similarity
            ]
            
            return filtered_vars[:max_results]
            
        except Exception as e:
            print(f"Error getting related variables: {e}")
            return []

def setup_intugle_complete(project_base: str = None, sales_data_path: str = None, files_to_process: list = []) -> IntugleKnowledgeIntegration:
    """Complete setup of Intugle for the project with fresh data"""
    setup = IntugleSetup(project_base)
    
    # Setup environment
    if not setup.setup_environment():
        raise Exception("Environment setup failed")
    
    # Load real data - always use fresh data
    data_input = load_sales_forecast_data(sales_data_path, files_to_process)
    
    if not data_input:
        raise Exception("No real data loaded - please check data path")
    
    print(f"✅ Loaded {len(data_input)} datasets for fresh setup")
    
    # Initialize components with force rebuild to ensure fresh data
    kb_success = setup.initialize_knowledge_builder(data_input, force_rebuild=True)
    dpb_success = setup.initialize_data_product_builder(force_rebuild=True)
    
    if not kb_success:
        raise Exception("Knowledge Builder initialization failed")
    
    print("✅ Fresh Intugle setup completed with real data")
    return IntugleKnowledgeIntegration(setup)

def setup_intugle_with_real_data(project_base: str = None, sales_data_path: str = None, files_to_process: list = []) -> IntugleKnowledgeIntegration:
    """Setup Intugle with real sales forecast data"""
    return setup_intugle_complete(
        project_base=project_base,
        sales_data_path=sales_data_path,
        files_to_process=files_to_process
    )

def get_cached_intugle_components(project_base: str = None) -> Dict[str, Any]:
    """Get cached Intugle components for agent use"""
    try:
        setup = IntugleSetup(project_base)
        
        kb_loaded = setup.load_knowledge_builder()
        dpb_loaded = setup.load_data_product_builder()
        
        if kb_loaded:
            return setup.get_cached_components()
        else:
            return None
            
    except Exception as e:
        print(f"Failed to get cached Intugle components: {e}")
        return None

def create_intugle_agent_integration(project_base: str = None) -> IntugleKnowledgeIntegration:
    """Create Intugle integration for agents with fresh data"""
    try:
        # Always use fresh data instead of cache
        return setup_intugle_with_real_data(project_base)
        
    except Exception as e:
        print(f"Failed to create Intugle agent integration: {e}")
        return None

def main():
    """Main setup function"""
    print("INTUGLE SETUP FOR DATA QUERYING AI")
    print("="*60)
    
    try:
        print("Setting up Intugle with real sales forecast data...")
        integration = setup_intugle_with_real_data()
        
        print("✓ Intugle setup completed successfully!")
        
        # Test basic functionality
        variables = integration.get_available_variables()
        print(f"✓ Found {len(variables)} variables in knowledge base")
        
        if integration.knowledge_builder:
            search_results = integration.search_variables("sales", limit=3)
            print(f"✓ Semantic search working - found {len(search_results)} results")
        
        return True
        
    except Exception as e:
        print(f"✗ Intugle setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)