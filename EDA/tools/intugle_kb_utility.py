"""
Intugle KnowledgeBuilder Utility - Load cached KB once for agent setup
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from EDA.tools.intugle_setup import get_cached_intugle_components, create_intugle_agent_integration

class IntugleKBUtility:
    """Utility for loading cached KnowledgeBuilder once during agent setup"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._kb = None
        self._integration = None
        self._components = None
        self._is_loaded = False
        self._load_kb()
    
    def _load_kb(self):
        """Load cached KnowledgeBuilder once"""
        try:
            self._components = get_cached_intugle_components()
            if self._components and self._components.get('knowledge_builder'):
                self._kb = self._components['knowledge_builder']
                self._integration = create_intugle_agent_integration()
                self._is_loaded = True
                self.logger.info("KnowledgeBuilder loaded successfully from cache")
            else:
                self.logger.warning("No cached KnowledgeBuilder found")
        except Exception as e:
            self.logger.error(f"Failed to load KnowledgeBuilder: {e}")
    
    def is_loaded(self) -> bool:
        """Check if KnowledgeBuilder is loaded"""
        return self._is_loaded and self._kb is not None
    
    def get_kb(self):
        """Get the loaded KnowledgeBuilder instance"""
        return self._kb
    
    def get_integration(self):
        """Get the loaded integration instance"""
        return self._integration
    
    def get_components(self) -> Dict[str, Any]:
        """Get all loaded components"""
        return self._components or {}
    
    def get_available_variables(self) -> list:
        """Get list of available variables from the loaded KB"""
        if not self.is_loaded():
            return []
        
        try:
            return self._integration.get_available_variables()
        except Exception as e:
            self.logger.error(f"Error getting available variables: {e}")
            return []
    
    def get_datasets_info(self) -> Dict[str, Any]:
        """Get information about datasets in the loaded KB"""
        if not self.is_loaded():
            return {"error": "KnowledgeBuilder not loaded", "datasets": {}}
        
        try:
            datasets_info = {}
            for dataset_name, dataset in self._kb.datasets.items():
                datasets_info[dataset_name] = {
                    "variables": [str(col) for col in dataset.source_table_model.columns],
                    "variable_count": len(dataset.source_table_model.columns),
                    "description": getattr(dataset, 'description', '')
                }
            return {"datasets": datasets_info, "status": "success"}
        except Exception as e:
            self.logger.error(f"Error getting datasets info: {e}")
            return {"error": str(e), "datasets": {}}

# Global instance for easy access
kb_utility = IntugleKBUtility()

# Convenience functions
def get_loaded_kb():
    """Get the loaded KnowledgeBuilder instance"""
    return kb_utility.get_kb()

def get_loaded_integration():
    """Get the loaded integration instance"""
    return kb_utility.get_integration()

def is_kb_loaded() -> bool:
    """Check if KnowledgeBuilder is loaded"""
    return kb_utility.is_loaded()

def get_available_variables() -> list:
    """Get list of available variables"""
    return kb_utility.get_available_variables()

def get_datasets_info() -> Dict[str, Any]:
    """Get information about datasets"""
    return kb_utility.get_datasets_info()
