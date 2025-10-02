"""
EDA Workflow Configuration

This module defines configuration classes for the EDA workflow.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum

class WorkflowMode(Enum):
    """Workflow execution modes"""
    FULL = "full"
    SIMPLIFIED = "simplified"
    TESTING = "testing"

class DomainType(Enum):
    """Domain types for customization"""
    RETAIL = "retail"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    MANUFACTURING = "manufacturing"
    GENERAL = "general"

@dataclass
class AgentConfig:
    """Configuration for individual agents"""
    timeout: int = 300
    max_retries: int = 3
    enable_caching: bool = True
    custom_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}

@dataclass
class WorkflowConfig:
    """Main configuration for the EDA workflow"""
    
    # Workflow settings
    mode: WorkflowMode = WorkflowMode.FULL
    domain: DomainType = DomainType.GENERAL
    hypothesis_limit: int = 10
    max_workers: int = 5
    timeout_per_hypothesis: int = 300
    
    # Agent configurations
    univariate_config: AgentConfig = None
    hypothesis_config: AgentConfig = None
    eda_worker_config: AgentConfig = None
    summarizer_config: AgentConfig = None
    
    # Intugle settings
    intugle_project_base: str = "models"
    enable_intugle: bool = True
    
    # Performance settings
    enable_parallel_processing: bool = True
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    
    # Output settings
    save_visualizations: bool = True
    visualization_format: str = "png"
    save_intermediate_results: bool = False
    
    def __post_init__(self):
        if self.univariate_config is None:
            self.univariate_config = AgentConfig()
        if self.hypothesis_config is None:
            self.hypothesis_config = AgentConfig()
        if self.eda_worker_config is None:
            self.eda_worker_config = AgentConfig()
        if self.summarizer_config is None:
            self.summarizer_config = AgentConfig()

class EDAWorkflowConfig:
    """Configuration manager for EDA workflow"""
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        self.config = config or WorkflowConfig()
    
    @classmethod
    def create_for_domain(cls, domain: DomainType) -> 'EDAWorkflowConfig':
        """Create configuration optimized for specific domain"""
        
        config = WorkflowConfig(domain=domain)
        
        if domain == DomainType.RETAIL:
            config.hypothesis_limit = 15
            config.timeout_per_hypothesis = 400
            config.hypothesis_config.custom_params = {
                "focus_areas": ["seasonality", "promotions", "pricing"],
                "correlation_threshold": 0.3
            }
            
        elif domain == DomainType.HEALTHCARE:
            config.hypothesis_limit = 12
            config.timeout_per_hypothesis = 500
            config.hypothesis_config.custom_params = {
                "focus_areas": ["patient_demographics", "treatment_effects", "outcomes"],
                "correlation_threshold": 0.2
            }
            
        elif domain == DomainType.FINANCE:
            config.hypothesis_limit = 20
            config.timeout_per_hypothesis = 350
            config.hypothesis_config.custom_params = {
                "focus_areas": ["market_indicators", "risk_factors", "performance_metrics"],
                "correlation_threshold": 0.4
            }
            
        elif domain == DomainType.MANUFACTURING:
            config.hypothesis_limit = 18
            config.timeout_per_hypothesis = 450
            config.hypothesis_config.custom_params = {
                "focus_areas": ["quality_metrics", "production_efficiency", "supply_chain"],
                "correlation_threshold": 0.35
            }
        
        return cls(config)
    
    @classmethod
    def create_for_testing(cls) -> 'EDAWorkflowConfig':
        """Create configuration optimized for testing"""
        
        config = WorkflowConfig(
            mode=WorkflowMode.TESTING,
            hypothesis_limit=3,
            max_workers=2,
            timeout_per_hypothesis=60,
            enable_intugle=False,
            save_visualizations=False,
            save_intermediate_results=True
        )
        
        # Reduce timeouts for testing
        config.univariate_config.timeout = 60
        config.hypothesis_config.timeout = 60
        config.eda_worker_config.timeout = 60
        config.summarizer_config.timeout = 60
        
        return cls(config)
    
    @classmethod
    def create_simplified(cls) -> 'EDAWorkflowConfig':
        """Create simplified configuration"""
        
        config = WorkflowConfig(
            mode=WorkflowMode.SIMPLIFIED,
            hypothesis_limit=5,
            max_workers=3,
            timeout_per_hypothesis=200
        )
        
        return cls(config)
    
    def get_agent_config(self, agent_name: str) -> AgentConfig:
        """Get configuration for specific agent"""
        
        config_map = {
            "univariate_analysis": self.config.univariate_config,
            "hypothesis_generation": self.config.hypothesis_config,
            "eda_worker_loop": self.config.eda_worker_config,
            "summarizer": self.config.summarizer_config
        }
        
        return config_map.get(agent_name, AgentConfig())
    
    def get_intugle_config(self) -> Dict[str, Any]:
        """Get Intugle-specific configuration"""
        
        return {
            "project_base": self.config.intugle_project_base,
            "enabled": self.config.enable_intugle,
            "cache_ttl": self.config.cache_ttl
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance-related configuration"""
        
        return {
            "parallel_processing": self.config.enable_parallel_processing,
            "caching": self.config.enable_caching,
            "max_workers": self.config.max_workers,
            "timeout_per_hypothesis": self.config.timeout_per_hypothesis
        }
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output-related configuration"""
        
        return {
            "save_visualizations": self.config.save_visualizations,
            "visualization_format": self.config.visualization_format,
            "save_intermediate_results": self.config.save_intermediate_results
        }
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return validation results"""
        
        validation = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Validate hypothesis limit
        if self.config.hypothesis_limit < 1:
            validation["errors"].append("hypothesis_limit must be at least 1")
            validation["valid"] = False
        elif self.config.hypothesis_limit > 50:
            validation["warnings"].append("hypothesis_limit > 50 may cause performance issues")
        
        # Validate max workers
        if self.config.max_workers < 1:
            validation["errors"].append("max_workers must be at least 1")
            validation["valid"] = False
        elif self.config.max_workers > 10:
            validation["warnings"].append("max_workers > 10 may cause resource issues")
        
        # Validate timeouts
        if self.config.timeout_per_hypothesis < 30:
            validation["warnings"].append("timeout_per_hypothesis < 30 seconds may be too short")
        
        # Validate domain-specific settings
        if self.config.domain != DomainType.GENERAL:
            domain_config = self.get_agent_config("hypothesis_generation")
            if not domain_config.custom_params:
                validation["warnings"].append(f"No domain-specific parameters for {self.config.domain.value}")
        
        return validation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        
        return {
            "mode": self.config.mode.value,
            "domain": self.config.domain.value,
            "hypothesis_limit": self.config.hypothesis_limit,
            "max_workers": self.config.max_workers,
            "timeout_per_hypothesis": self.config.timeout_per_hypothesis,
            "intugle_project_base": self.config.intugle_project_base,
            "enable_intugle": self.config.enable_intugle,
            "enable_parallel_processing": self.config.enable_parallel_processing,
            "enable_caching": self.config.enable_caching,
            "cache_ttl": self.config.cache_ttl,
            "save_visualizations": self.config.save_visualizations,
            "visualization_format": self.config.visualization_format,
            "save_intermediate_results": self.config.save_intermediate_results,
            "agent_configs": {
                "univariate": {
                    "timeout": self.config.univariate_config.timeout,
                    "max_retries": self.config.univariate_config.max_retries,
                    "enable_caching": self.config.univariate_config.enable_caching,
                    "custom_params": self.config.univariate_config.custom_params
                },
                "hypothesis": {
                    "timeout": self.config.hypothesis_config.timeout,
                    "max_retries": self.config.hypothesis_config.max_retries,
                    "enable_caching": self.config.hypothesis_config.enable_caching,
                    "custom_params": self.config.hypothesis_config.custom_params
                },
                "eda_worker": {
                    "timeout": self.config.eda_worker_config.timeout,
                    "max_retries": self.config.eda_worker_config.max_retries,
                    "enable_caching": self.config.eda_worker_config.enable_caching,
                    "custom_params": self.config.eda_worker_config.custom_params
                },
                "summarizer": {
                    "timeout": self.config.summarizer_config.timeout,
                    "max_retries": self.config.summarizer_config.max_retries,
                    "enable_caching": self.config.summarizer_config.enable_caching,
                    "custom_params": self.config.summarizer_config.custom_params
                }
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EDAWorkflowConfig':
        """Create configuration from dictionary"""
        
        # Create base config
        config = WorkflowConfig()
        
        # Set basic properties
        if "mode" in config_dict:
            config.mode = WorkflowMode(config_dict["mode"])
        if "domain" in config_dict:
            config.domain = DomainType(config_dict["domain"])
        if "hypothesis_limit" in config_dict:
            config.hypothesis_limit = config_dict["hypothesis_limit"]
        if "max_workers" in config_dict:
            config.max_workers = config_dict["max_workers"]
        if "timeout_per_hypothesis" in config_dict:
            config.timeout_per_hypothesis = config_dict["timeout_per_hypothesis"]
        
        # Set other properties
        for key in ["intugle_project_base", "enable_intugle", "enable_parallel_processing", 
                   "enable_caching", "cache_ttl", "save_visualizations", 
                   "visualization_format", "save_intermediate_results"]:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        # Set agent configs
        if "agent_configs" in config_dict:
            agent_configs = config_dict["agent_configs"]
            
            for agent_name, agent_config in agent_configs.items():
                if agent_name == "univariate":
                    config.univariate_config = AgentConfig(**agent_config)
                elif agent_name == "hypothesis":
                    config.hypothesis_config = AgentConfig(**agent_config)
                elif agent_name == "eda_worker":
                    config.eda_worker_config = AgentConfig(**agent_config)
                elif agent_name == "summarizer":
                    config.summarizer_config = AgentConfig(**agent_config)
        
        return cls(config)
