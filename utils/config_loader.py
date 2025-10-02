import yaml
import os
import importlib
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from langchain_core.tools import Tool

class AgentConfigLoader:
    """Loads and manages agent configurations from YAML files"""
    
    def __init__(self, config_dir: str = "EDA/agents"):
        self.config_dir = Path(config_dir)
        self._configs = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all agent configurations from the config directory"""
        for agent_dir in self.config_dir.iterdir():
            if agent_dir.is_dir() and (agent_dir / "config.yaml").exists():
                agent_name = agent_dir.name
                config_path = agent_dir / "config.yaml"
                self._configs[agent_name] = self._load_config(config_path)
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """Load a single configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            raise ValueError(f"Error loading config from {config_path}: {e}")
    
    def get_agent_config(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific agent"""
        return self._configs.get(agent_name)
    
    def get_model_config(self, agent_name: str) -> Optional[Dict[str, str]]:
        """Get model configuration for a specific agent"""
        config = self.get_agent_config(agent_name)
        if config:
            return {
                'provider': config.get('model_provider'),
                'model': config.get('model_name')
            }
        return None
    
    def load_agent_class(self, agent_name: str):
        """Dynamically load the agent class"""
        config = self.get_agent_config(agent_name)
        if not config or 'agent' not in config:
            raise ValueError(f"No agent class info found for {agent_name}")
        
        agent_info = config['agent']
        try:
            # Import the module
            module_path = f"agents.{agent_name}.{agent_info['file'].replace('.py', '')}"
            module = importlib.import_module(module_path)
            
            # Get the class
            agent_class = getattr(module, agent_info['class'])
            return agent_class
        except Exception as e:
            raise ValueError(f"Error loading agent class for {agent_name}: {e}")
    
    def load_prompt(self, agent_name: str):
        """Load the prompt from the specified file"""
        config = self.get_agent_config(agent_name)
        if not config or 'prompt' not in config:
            raise ValueError(f"No prompt info found for {agent_name}")
        
        prompt_info = config['prompt']
        try:
            # Import the module
            module_path = f"EDA.agents.{agent_name}.{prompt_info['file'].replace('.py', '')}"
            module = importlib.import_module(module_path)
            
            # Get the prompt (variable or function)
            if 'variable' in prompt_info:
                prompt = getattr(module, prompt_info['variable'])
            elif 'function' in prompt_info:
                prompt_function = getattr(module, prompt_info['function'])
                prompt = prompt_function()
            else:
                raise ValueError(f"Prompt config must have 'variable' or 'function'")
            
            return prompt
        except Exception as e:
            raise ValueError(f"Error loading prompt for {agent_name}: {e}")
    
    def load_tool_function(self, tool_config: Dict[str, Any]):
        """Load a tool function from the specified file"""
        try:
            # Import the module
            module_path = tool_config['file'].replace('.py', '').replace('/', '.')
            module = importlib.import_module(module_path)
            
            # Get the function
            tool_function = getattr(module, tool_config['function'])
            return tool_function
        except Exception as e:
            raise ValueError(f"Error loading tool function {tool_config['name']}: {e}")
    
    def create_tools_for_agent(self, agent_name: str, state: Any) -> List[Tool]:
        """Create tools for an agent, handling state-aware tools"""
        config = self.get_agent_config(agent_name)
        if not config or 'tools' not in config:
            return []
        
        tools = []
        for tool_config in config['tools']:
            try:
                tool_function = self.load_tool_function(tool_config)
                
                # Check if tool needs state
                if tool_config.get('state_required', False):
                    # Create state-aware tool
                    tool = tool_function(state)
                else:
                    # Create regular tool
                    tool = tool_function
                
                tools.append(tool)
                
            except Exception as e:
                print(f"Warning: Could not load tool {tool_config['name']}: {e}")
        
        return tools
    
    def create_agent_instance(self, agent_name: str, state: Any = None, **kwargs):
        """Create an instance of an agent with its configuration"""
        config = self.get_agent_config(agent_name)
        if not config:
            raise ValueError(f"No configuration found for agent {agent_name}")
        
        # Load agent class
        agent_class = self.load_agent_class(agent_name)
        
        # Load prompt
        prompt = self.load_prompt(agent_name)
        
        # Get model config
        model_config = self.get_model_config(agent_name)
        
        # Create tools (if state is provided)
        tools = []
        if state is not None:
            tools = self.create_tools_for_agent(agent_name, state)
        
        # Create agent instance
        agent_instance = agent_class(
            model_config=model_config,
            prompt=prompt,
            tools=tools,
            **kwargs
        )
        
        return agent_instance
    
    def create_ports_for_agent(self, agent_name: str) -> dict:
        """Create ports for an agent based on configuration"""
        config = self.get_agent_config(agent_name)
        if not config or 'ports' not in config:
            return {}
        
        ports = {}
        for port_name, port_config in config['ports'].items():
            try:
                adapter_class_name = port_config['adapter']
                adapter_config = port_config.get('config', {})
                
                # Import the adapter class
                if adapter_class_name == "MongoDBVectorSearchAdapter":
                    from adapters.vector_search.mongodb_adapter import MongoDBVectorSearchAdapter
                    adapter_class = MongoDBVectorSearchAdapter
                else:
                    raise ValueError(f"Unknown adapter: {adapter_class_name}")
                
                # Create adapter instance with config
                if adapter_class_name == "MongoDBVectorSearchAdapter":
                    import os
                    mongo_uri = os.getenv(adapter_config.get('mongo_uri_env', 'MONGO_ATLAS_SEARCH_INDEX_ENABLED_DB'))
                    if not mongo_uri:
                        raise ValueError(f"Environment variable {adapter_config.get('mongo_uri_env')} not found")
                    
                    adapter_instance = adapter_class(
                        mongo_uri=mongo_uri,
                        embedding_provider=adapter_config.get('embedding_provider', 'gemini'),
                        embedding_model=adapter_config.get('embedding_model', 'models/embedding-001'),
                        database_name=adapter_config.get('database_name', 'cmms')
                    )
                
                ports[port_name] = adapter_instance
                
            except Exception as e:
                print(f"Warning: Could not create port {port_name}: {e}")
        
        return ports
    
    def get_agent_processor(self, agent_name: str):
        """Get the agent processor function (like insights_agent)"""
        try:
            # Import the module
            module_path = f"agents.{agent_name}.{agent_name}"
            module = importlib.import_module(module_path)
            
            # Get the processor function
            processor_function = getattr(module, f"{agent_name}_agent")
            return processor_function
        except Exception as e:
            raise ValueError(f"Error loading agent processor for {agent_name}: {e}")
    
    def initialize_agent_functions(self, agent_names: List[str]) -> Dict[str, Callable]:
        """
        Initialize multiple agent functions at once.
        
        Args:
            agent_names: List of agent names to initialize.
            
        Returns:
            Dict mapping agent names to their processor functions.
        """
        agent_functions = {}
        
        for agent_name in agent_names:
            try:
                config_name = agent_name.replace('_agent', '')
                processor_function = self.get_agent_processor(config_name)
                agent_functions[agent_name] = processor_function
            except Exception as e:
                print(f"Error loading agent processor for {agent_name}: {str(e)}")
        
        return agent_functions

# Usage example
if __name__ == "__main__":
    loader = AgentConfigLoader()
    
    # Get all agents
    all_agents = loader.get_all_agents()
    print(f"Loaded {len(all_agents)} agents: {list(all_agents.keys())}")
    
    # Example: Get insights agent processor
    try:
        insights_processor = loader.get_agent_processor("insights")
        print("Successfully loaded insights agent processor")
    except Exception as e:
        print(f"Error loading insights agent processor: {e}") 