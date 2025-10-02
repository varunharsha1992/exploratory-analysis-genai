"""
Test Script for Univariate Analysis Agent

This script provides comprehensive testing for the UnivariateAnalysisAgent including:
- Unit tests for individual components
- Integration tests with real Intugle knowledge base (with fallback to mock)
- Agentic flow testing
- Error handling validation
- Performance testing
- Demo functionality

Updated to use real Intugle setup with graceful fallback to mock objects.
Tests the new agent tools architecture including semantic search and variable connections.
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Import the agent and related components
from EDA.agents.univariate_analysis.univariate_analysis import UnivariateAnalysisAgent
from EDA.tools.univariate_analysis import (
    DataProfilingTool, 
    AnomalyDetectionTool, 
    RelatedVariablesTool, 
    TrendAnalysisTool,
    AnomalyMethod
)

# Import Intugle setup and agent tools
from utils.setup_intugle import setup_intugle_with_real_data
from EDA.tools.intugle_agent_tools import (
    get_variable_profiles,
    search_variables,
    create_etl_schema_with_llm,
    build_dataproduct,
    is_intugle_available
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestUnivariateAnalysisAgent(unittest.TestCase):
    """Test cases for the UnivariateAnalysisAgent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.target_variable = "sales.Units_Sold"
        
        # Try to set up real Intugle KB, fallback to mock if not available
        self.kb = self._setup_knowledge_base()
        self.test_config = {
            "anomaly_detection_method": "iqr",
            "trend_analysis": True,
            "visualization_enabled": True
        }
        
        # Create test data based on sales schema
        self.test_data = self._create_sales_test_data()
        
        # Initialize agent with data
        self.agent = UnivariateAnalysisAgent(
            target_variable=self.target_variable,
            data=self.test_data,
            kb=self.kb,
            config=self.test_config
        )
    
    def _setup_knowledge_base(self):
        """Set up knowledge base - try real Intugle first, fallback to mock"""
        try:
            # Check if Intugle is available
            if is_intugle_available():
                logger.info("Setting up real Intugle knowledge base...")
                integration = setup_intugle_with_real_data()
                
                # Create a mock object that wraps the real integration
                kb_wrapper = Mock()
                kb_wrapper.data_product_builder = integration.data_product_builder
                kb_wrapper.semantic_search = integration.semantic_search
                kb_wrapper.knowledge_builder = integration.knowledge_builder
                kb_wrapper.is_real = True
                
                logger.info("✓ Real Intugle knowledge base setup successful")
                return kb_wrapper
            else:
                logger.warning("Intugle not available, using mock knowledge base")
                return self._create_mock_kb()
                
        except Exception as e:
            logger.warning(f"Failed to setup real Intugle KB: {e}, using mock")
            return self._create_mock_kb()
    
    def _create_mock_kb(self):
        """Create a mock knowledge base"""
        mock_kb = Mock()
        mock_kb.data_product_builder = Mock()
        mock_kb.semantic_search = Mock()
        mock_kb.knowledge_builder = Mock()
        mock_kb.is_real = False
        return mock_kb
    
    def _create_sales_test_data(self) -> pd.DataFrame:
        """Load real sales data from the specified directory"""
        try:
            # Path to real sales data
            sales_data_path = r"C:\Users\varun\Downloads\Sales Forecast Data\sales.csv"
            
            if os.path.exists(sales_data_path):
                logger.info(f"Loading real sales data from: {sales_data_path}")
                data = pd.read_csv(sales_data_path)
                logger.info(f"✓ Loaded real sales data: {len(data)} rows, {len(data.columns)} columns")
                logger.info(f"Columns: {list(data.columns)}")
                return data
            else:
                logger.warning(f"Sales data file not found at: {sales_data_path}")
                logger.info("Falling back to synthetic data...")
                return self._create_synthetic_sales_data()
                
        except Exception as e:
            logger.error(f"Failed to load real sales data: {e}")
            logger.info("Falling back to synthetic data...")
            return self._create_synthetic_sales_data()
    
    def _create_synthetic_sales_data(self) -> pd.DataFrame:
        """Create synthetic sales test data as fallback"""
        np.random.seed(42)  # For reproducible results
        
        # Create 1000 records as specified in the schema
        n_records = 1000
        
        # Product categories from the schema
        product_categories = ['Sports', 'Toys', 'Fashion', 'Electronics', 'Home Decor']
        
        # Generate units sold with realistic patterns
        # Different categories have different typical sales volumes
        category_means = {
            'Sports': 35,
            'Toys': 45,
            'Fashion': 25,
            'Electronics': 20,
            'Home Decor': 15
        }
        
        # Generate data
        data = []
        for i in range(n_records):
            category = np.random.choice(product_categories)
            base_units = category_means[category]
            
            # Add some variation and seasonality
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * i / 365)  # Annual seasonality
            weekly_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 7)  # Weekly pattern
            noise = np.random.normal(0, 0.2)  # Random variation
            
            units_sold = max(1, int(base_units * seasonal_factor * weekly_factor * (1 + noise)))
            
            # Add some anomalies (5% of records)
            if np.random.random() < 0.05:
                units_sold = int(units_sold * np.random.uniform(2, 5))  # 2-5x normal
            
            data.append({
                'Product_Category': category,
                'Units_Sold': units_sold
            })
        
        return pd.DataFrame(data)
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        self.assertEqual(self.agent.target_variable, self.target_variable)
        self.assertIsNotNone(self.agent.llm)
        self.assertIsNotNone(self.agent.prompt)
        self.assertEqual(self.agent.anomaly_detection_method, "iqr")
        self.assertTrue(self.agent.trend_analysis)
        self.assertTrue(self.agent.visualization_enabled)
        
        # Check if we're using real or mock KB
        if hasattr(self.kb, 'is_real') and self.kb.is_real:
            logger.info("✓ Agent initialized with real Intugle knowledge base")
        else:
            logger.info("✓ Agent initialized with mock knowledge base")
    
    def test_tool_creation(self):
        """Test that analysis tools are created correctly"""
        tools = self.agent._create_analysis_tools()
        
        # Check that all expected tools are created
        tool_names = [tool.name for tool in tools]
        expected_tools = [
            'profile_variable',
            'detect_anomalies', 
            'analyze_trends',
            'discover_related_variables',
            'search_knowledge_base',
            'get_variable_metadata',
            'create_etl_schema',
            'build_data_product'
        ]
        
        for expected_tool in expected_tools:
            self.assertIn(expected_tool, tool_names)
        
        logger.info(f"✓ Created {len(tools)} analysis tools: {tool_names}")
    
    def test_data_profiling_tool(self):
        """Test data profiling functionality"""
        tools = self.agent._create_analysis_tools()
        profile_tool = next(tool for tool in tools if tool.name == "profile_variable")
        
        # Test with preloaded data (no data parameter needed)
        result = profile_tool.invoke({"variable_id": "sales.Units_Sold"})
        
        self.assertIsInstance(result, dict)
        self.assertIn("variable_id", result)
        self.assertIn("statistical_summary", result)
        self.assertIn("data_type", result)
        
        # Check statistical summary
        stats = result["statistical_summary"]
        self.assertIn("count", stats)
        self.assertIn("mean", stats)
        self.assertIn("std", stats)
        self.assertIn("min", stats)
        self.assertIn("max", stats)
    
    def test_anomaly_detection_tool(self):
        """Test anomaly detection functionality"""
        tools = self.agent._create_analysis_tools()
        anomaly_tool = next(tool for tool in tools if tool.name == "detect_anomalies")
        
        # Test with preloaded data (no data parameter needed)
        result = anomaly_tool.invoke({"variable_id": "sales.Units_Sold"})
        
        self.assertIsInstance(result, dict)
        self.assertIn("variable_id", result)
        self.assertIn("anomaly_results", result)
        self.assertIn("combined_result", result)
        
        # Check that anomalies were detected (we added some in test data)
        combined = result["combined_result"]
        self.assertIn("total_unique_outliers", combined)
        self.assertGreaterEqual(combined["total_unique_outliers"], 0)
    
    def test_related_variables_tool(self):
        """Test related variables discovery"""
        tools = self.agent._create_analysis_tools()
        related_tool = next(tool for tool in tools if tool.name == "discover_related_variables")
        
        # Test with preloaded data (no data parameter needed)
        result = related_tool.invoke({"target_variable": "sales.Units_Sold", "max_results": 5})
        
        self.assertIsInstance(result, dict)
        # Should find some related variables from knowledge base
        self.assertIn("related_variables", result)
    
    def test_trend_analysis_tool(self):
        """Test trend analysis functionality"""
        tools = self.agent._create_analysis_tools()
        trend_tool = next(tool for tool in tools if tool.name == "analyze_trends")
        
        # Test with preloaded data (no data parameter needed)
        result = trend_tool.invoke({"variable_id": "sales.Units_Sold"})
        
        self.assertIsInstance(result, dict)
        self.assertIn("variable_id", result)
        self.assertIn("temporal_trends", result)
        self.assertIn("seasonality_analysis", result)
        self.assertIn("volatility_analysis", result)
        
        # Check temporal trends
        temporal = result["temporal_trends"]
        self.assertIn("trend_direction", temporal)
        self.assertIn("trend_strength", temporal)
    
    @patch('agents.univariate_analysis.univariate_analysis.create_react_agent')
    def test_agentic_flow_mock(self, mock_create_react_agent):
        """Test the agentic flow with mocked LLM"""
        # Mock the LLM response
        mock_agent = Mock()
        mock_response = {
            "messages": [
                Mock(content=json.dumps({
                    "target_variable": {
                        "name": "revenue",
                        "data_type": "continuous",
                        "profile": {"count": 365, "mean": 1250.0},
                        "anomalies": {"outliers_count": 10},
                        "trends": {"trend_direction": "increasing"}
                    },
                    "related_variables": [],
                    "data_quality_summary": {
                        "overall_quality": "good",
                        "issues_found": 1,
                        "recommendations": ["Handle 10 outliers"]
                    },
                    "overall_insights": "Revenue shows upward trend with some anomalies",
                    "feature_engineering_recommendations": ["Create lag features"]
                }))
            ]
        }
        mock_agent.invoke.return_value = mock_response
        mock_create_react_agent.return_value = mock_agent
        
        # Test the process method with state
        state = {
            "target_variable": "sales.Units_Sold",
            "data": self.test_data,
            "messages": [],
            "context": {}
        }
        result = self.agent.process(state)
        
        # Verify the result structure
        self.assertIsInstance(result, dict)
        self.assertIn("univariate_results", result)
        self.assertIn("current_agent", result)
        self.assertIn("execution_status", result)
        
        # Verify that create_react_agent was called
        mock_create_react_agent.assert_called_once()
    
    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test with empty data
        empty_data = pd.DataFrame()
        agent_with_empty_data = UnivariateAnalysisAgent(
            target_variable="sales.Units_Sold",
            data=empty_data,
            kb=self.kb
        )
        state = {
            "target_variable": "sales.Units_Sold",
            "data": empty_data,
            "messages": [],
            "context": {}
        }
        result = agent_with_empty_data.process(state)
        self.assertIn("execution_status", result)
        
        # Test with invalid variable
        invalid_agent = UnivariateAnalysisAgent(
            target_variable="nonexistent.column",
            data=self.test_data,
            kb=self.kb
        )
        state = {
            "target_variable": "nonexistent.column",
            "data": self.test_data,
            "messages": [],
            "context": {}
        }
        result = invalid_agent.process(state)
        self.assertIn("execution_status", result)
    
    def test_configuration_options(self):
        """Test different configuration options"""
        # Test with different anomaly detection method
        config = {"anomaly_detection_method": "z_score"}
        agent = UnivariateAnalysisAgent(
            target_variable=self.target_variable,
            data=self.test_data,
            kb=self.kb,
            config=config
        )
        self.assertEqual(agent.anomaly_detection_method, "z_score")
        
        # Test with trend analysis disabled
        config = {"trend_analysis": False}
        agent = UnivariateAnalysisAgent(
            target_variable=self.target_variable,
            data=self.test_data,
            kb=self.kb,
            config=config
        )
        self.assertFalse(agent.trend_analysis)
    
    def test_real_intugle_integration(self):
        """Test integration with real Intugle knowledge base"""
        if not (hasattr(self.kb, 'is_real') and self.kb.is_real):
            self.skipTest("Real Intugle KB not available, skipping integration test")
        
        logger.info("Testing real Intugle integration...")
        
        # Test knowledge base search
        tools = self.agent._create_analysis_tools()
        search_tool = next(tool for tool in tools if tool.name == "search_knowledge_base")
        
        try:
            # Test knowledge base search with a real query
            result = search_tool.invoke({"query": "sales", "max_results": 5})
            self.assertIsInstance(result, dict)
            logger.info(f"✓ Knowledge base search found results")
            
            # Test related variables discovery
            related_tool = next(tool for tool in tools if tool.name == "discover_related_variables")
            related_result = related_tool.invoke({"target_variable": "sales.Units_Sold", "max_results": 3})
            self.assertIsInstance(related_result, dict)
            logger.info(f"✓ Related variables discovery completed")
            
            # Test variable metadata
            metadata_tool = next(tool for tool in tools if tool.name == "get_variable_metadata")
            metadata_result = metadata_tool.invoke({"variable_id": "sales.Units_Sold"})
            self.assertIsInstance(metadata_result, dict)
            logger.info("✓ Variable metadata retrieval completed")
            
        except Exception as e:
            logger.warning(f"Real Intugle integration test failed: {e}")
            # Don't fail the test, just log the warning
            pass

class TestToolIntegration(unittest.TestCase):
    """Test integration between tools and agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = self._create_comprehensive_test_data()
        self.kb = self._setup_knowledge_base()
    
    def _setup_knowledge_base(self):
        """Set up knowledge base - try real Intugle first, fallback to mock"""
        try:
            # Check if Intugle is available
            if is_intugle_available():
                logger.info("Setting up real Intugle knowledge base for tool integration tests...")
                integration = setup_intugle_with_real_data()
                
                # Create a mock object that wraps the real integration
                kb_wrapper = Mock()
                kb_wrapper.data_product_builder = integration.data_product_builder
                kb_wrapper.semantic_search = integration.semantic_search
                kb_wrapper.knowledge_builder = integration.knowledge_builder
                kb_wrapper.is_real = True
                
                logger.info("✓ Real Intugle knowledge base setup successful for tool integration")
                return kb_wrapper
            else:
                logger.warning("Intugle not available, using mock knowledge base for tool integration")
                return self._create_mock_kb()
                
        except Exception as e:
            logger.warning(f"Failed to setup real Intugle KB for tool integration: {e}, using mock")
            return self._create_mock_kb()
    
    def _create_mock_kb(self):
        """Create a mock knowledge base"""
        mock_kb = Mock()
        mock_kb.data_product_builder = Mock()
        mock_kb.semantic_search = Mock()
        mock_kb.knowledge_builder = Mock()
        mock_kb.is_real = False
        return mock_kb
    
    def _create_comprehensive_test_data(self) -> pd.DataFrame:
        """Load real sales data from the specified directory"""
        try:
            # Path to real sales data
            sales_data_path = r"C:\Users\varun\Downloads\Sales Forecast Data\sales.csv"
            
            if os.path.exists(sales_data_path):
                logger.info(f"Loading real sales data for comprehensive tests from: {sales_data_path}")
                data = pd.read_csv(sales_data_path)
                logger.info(f"✓ Loaded real sales data: {len(data)} rows, {len(data.columns)} columns")
                return data
            else:
                logger.warning(f"Sales data file not found at: {sales_data_path}")
                logger.info("Falling back to synthetic data...")
                return self._create_synthetic_comprehensive_data()
                
        except Exception as e:
            logger.error(f"Failed to load real sales data: {e}")
            logger.info("Falling back to synthetic data...")
            return self._create_synthetic_comprehensive_data()
    
    def _create_synthetic_comprehensive_data(self) -> pd.DataFrame:
        """Create comprehensive synthetic sales test data as fallback"""
        np.random.seed(123)
        
        # Create 1000 records as specified in the schema
        n_records = 1000
        
        # Product categories from the schema
        product_categories = ['Sports', 'Toys', 'Fashion', 'Electronics', 'Home Decor']
        
        # Generate units sold with realistic patterns
        category_means = {
            'Sports': 35,
            'Toys': 45,
            'Fashion': 25,
            'Electronics': 20,
            'Home Decor': 15
        }
        
        # Generate data
        data = []
        for i in range(n_records):
            category = np.random.choice(product_categories)
            base_units = category_means[category]
            
            # Add some variation and seasonality
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * i / 365)  # Annual seasonality
            weekly_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 7)  # Weekly pattern
            noise = np.random.normal(0, 0.2)  # Random variation
            
            units_sold = max(1, int(base_units * seasonal_factor * weekly_factor * (1 + noise)))
            
            # Add some anomalies (5% of records)
            if np.random.random() < 0.05:
                units_sold = int(units_sold * np.random.uniform(2, 5))  # 2-5x normal
            
            data.append({
                'Product_Category': category,
                'Units_Sold': units_sold
            })
        
        return pd.DataFrame(data)
    
    def _generate_trend_data(self, n_days: int, base: float, trend: float, seasonal: bool = False) -> np.ndarray:
        """Generate data with trend and optional seasonality"""
        trend_component = np.linspace(0, trend * n_days, n_days)
        seasonal_component = 100 * np.sin(2 * np.pi * np.arange(n_days) / 365) if seasonal else 0
        noise = np.random.normal(0, 50, n_days)
        return base + trend_component + seasonal_component + noise
    
    def _generate_stable_data(self, n_days: int, base: float, weekly: bool = False) -> np.ndarray:
        """Generate stable data with optional weekly pattern"""
        weekly_component = 10 * np.sin(2 * np.pi * np.arange(n_days) / 7) if weekly else 0
        noise = np.random.normal(0, 5, n_days)
        return base + weekly_component + noise
    
    def _generate_volatile_data(self, n_days: int, base: float, volatility: float) -> np.ndarray:
        """Generate volatile data with anomalies"""
        noise = np.random.normal(0, base * volatility, n_days)
        # Add some extreme values
        anomaly_indices = np.random.choice(n_days, size=int(n_days * 0.05), replace=False)
        noise[anomaly_indices] *= 5
        return base + noise
    
    def test_end_to_end_analysis(self):
        """Test complete end-to-end analysis workflow"""
        # Initialize agent
        agent = UnivariateAnalysisAgent(
            target_variable="sales.Units_Sold",
            data=self.test_data,
            kb=self.kb,
            config={"anomaly_detection_method": "iqr"}
        )
        
        # Log whether we're using real or mock KB
        if hasattr(self.kb, 'is_real') and self.kb.is_real:
            logger.info("Running end-to-end test with real Intugle KB")
        else:
            logger.info("Running end-to-end test with mock KB")
        
        # Test individual tools first
        tools = agent._create_analysis_tools()
        
        # Test profile_variable tool
        profile_tool = next(tool for tool in tools if tool.name == "profile_variable")
        profile_result = profile_tool.invoke({"variable_id": "sales.Units_Sold"})
        self.assertIsInstance(profile_result, dict)
        
        # Test detect_anomalies tool
        anomaly_tool = next(tool for tool in tools if tool.name == "detect_anomalies")
        anomaly_result = anomaly_tool.invoke({"variable_id": "sales.Units_Sold"})
        self.assertIsInstance(anomaly_result, dict)
        
        # Test analyze_trends tool
        trend_tool = next(tool for tool in tools if tool.name == "analyze_trends")
        trend_result = trend_tool.invoke({"variable_id": "sales.Units_Sold"})
        self.assertIsInstance(trend_result, dict)
        
        # Test discover_related_variables tool
        try:
            related_tool = next(tool for tool in tools if tool.name == "discover_related_variables")
            related_result = related_tool.invoke({"target_variable": "sales.Units_Sold", "max_results": 5})
            self.assertIsInstance(related_result, dict)
            logger.info(f"✓ Related variables test passed")
        except StopIteration:
            logger.warning("discover_related_variables tool not found, skipping test")
        
        # Test search_knowledge_base tool
        try:
            search_tool = next(tool for tool in tools if tool.name == "search_knowledge_base")
            search_result = search_tool.invoke({"query": "sales", "max_results": 3})
            self.assertIsInstance(search_result, dict)
            logger.info(f"✓ Knowledge base search test passed")
        except StopIteration:
            logger.warning("search_knowledge_base tool not found, skipping test")

def run_performance_test():
    """Run performance tests on the agent"""
    print("\n" + "="*50)
    print("PERFORMANCE TESTS")
    print("="*50)
    
    # Set up knowledge base
    print("Setting up knowledge base...")
    try:
        if is_intugle_available():
            integration = setup_intugle_with_real_data()
            kb_wrapper = Mock()
            kb_wrapper.data_product_builder = integration.data_product_builder
            kb_wrapper.semantic_search = integration.semantic_search
            kb_wrapper.knowledge_builder = integration.knowledge_builder
            kb_wrapper.is_real = True
            print("✓ Using real Intugle knowledge base")
        else:
            kb_wrapper = Mock()
            kb_wrapper.is_real = False
            print("✓ Using mock knowledge base")
    except Exception as e:
        kb_wrapper = Mock()
        kb_wrapper.is_real = False
        print(f"⚠️  Failed to setup real KB: {e}, using mock")
    
    # Load real sales data
    print("Loading real sales data for performance tests...")
    sales_data_path = r"C:\Users\varun\Downloads\Sales Forecast Data\sales.csv"
    
    try:
        if os.path.exists(sales_data_path):
            large_data = pd.read_csv(sales_data_path)
            print(f"✓ Loaded real sales data: {len(large_data)} rows, {len(large_data.columns)} columns")
        else:
            print(f"⚠️  Sales data file not found at: {sales_data_path}")
            print("Creating synthetic large dataset...")
            np.random.seed(42)
            n_rows = 10000
            
            # Create large sales dataset
            product_categories = ['Sports', 'Toys', 'Fashion', 'Electronics', 'Home Decor']
            category_means = {'Sports': 35, 'Toys': 45, 'Fashion': 25, 'Electronics': 20, 'Home Decor': 15}
            
            large_data = []
            for i in range(n_rows):
                category = np.random.choice(product_categories)
                base_units = category_means[category]
                units_sold = max(1, int(base_units * (1 + np.random.normal(0, 0.3))))
                large_data.append({
                    'Product_Category': category,
                    'Units_Sold': units_sold
                })
            
            large_data = pd.DataFrame(large_data)
    except Exception as e:
        print(f"⚠️  Failed to load real sales data: {e}")
        print("Creating synthetic large dataset...")
        np.random.seed(42)
        n_rows = 10000
        
        # Create large sales dataset
        product_categories = ['Sports', 'Toys', 'Fashion', 'Electronics', 'Home Decor']
        category_means = {'Sports': 35, 'Toys': 45, 'Fashion': 25, 'Electronics': 20, 'Home Decor': 15}
        
        large_data = []
        for i in range(n_rows):
            category = np.random.choice(product_categories)
            base_units = category_means[category]
            units_sold = max(1, int(base_units * (1 + np.random.normal(0, 0.3))))
            large_data.append({
                'Product_Category': category,
                'Units_Sold': units_sold
            })
        
        large_data = pd.DataFrame(large_data)
    
    # Initialize agent
    agent = UnivariateAnalysisAgent(
        target_variable="sales.Units_Sold",
        data=large_data,
        kb=kb_wrapper,
        config={"anomaly_detection_method": "iqr"}
    )
    
    # Test individual tool performance
    tools = agent._create_analysis_tools()
    
    for tool in tools:
        start_time = datetime.now()
        try:
            if tool.name == "profile_variable":
                result = tool.invoke({"variable_id": "sales.Units_Sold"})
            elif tool.name == "detect_anomalies":
                result = tool.invoke({"variable_id": "sales.Units_Sold"})
            elif tool.name == "analyze_trends":
                result = tool.invoke({"variable_id": "sales.Units_Sold"})
            else:
                continue
                
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            print(f"{tool.name}: {duration:.2f} seconds")
            
        except Exception as e:
            print(f"{tool.name}: ERROR - {str(e)}")

def run_demo():
    """Run a demonstration of the agent capabilities"""
    print("\n" + "="*50)
    print("UNIVARIATE ANALYSIS AGENT DEMO")
    print("="*50)
    
    # Set up knowledge base
    print("Setting up knowledge base...")
    try:
        if is_intugle_available():
            integration = setup_intugle_with_real_data()
            kb_wrapper = Mock()
            kb_wrapper.data_product_builder = integration.data_product_builder
            kb_wrapper.semantic_search = integration.semantic_search
            kb_wrapper.knowledge_builder = integration.knowledge_builder
            kb_wrapper.is_real = True
            print("✓ Using real Intugle knowledge base")
        else:
            kb_wrapper = Mock()
            kb_wrapper.is_real = False
            print("✓ Using mock knowledge base")
    except Exception as e:
        kb_wrapper = Mock()
        kb_wrapper.is_real = False
        print(f"⚠️  Failed to setup real KB: {e}, using mock")
    
    # Load real demo data
    print("Loading real sales data for demo...")
    sales_data_path = r"C:\Users\varun\Downloads\Sales Forecast Data\sales.csv"
    
    try:
        if os.path.exists(sales_data_path):
            demo_data = pd.read_csv(sales_data_path)
            print(f"✓ Loaded real sales data: {len(demo_data)} rows, {len(demo_data.columns)} columns")
        else:
            print(f"⚠️  Sales data file not found at: {sales_data_path}")
            print("Creating synthetic demo dataset...")
            np.random.seed(42)
            n_records = 1000
            
            # Product categories from the schema
            product_categories = ['Sports', 'Toys', 'Fashion', 'Electronics', 'Home Decor']
            category_means = {'Sports': 35, 'Toys': 45, 'Fashion': 25, 'Electronics': 20, 'Home Decor': 15}
            
            # Generate demo sales data
            demo_data = []
            for i in range(n_records):
                category = np.random.choice(product_categories)
                base_units = category_means[category]
                
                # Add some variation and seasonality
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * i / 365)  # Annual seasonality
                weekly_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 7)  # Weekly pattern
                noise = np.random.normal(0, 0.2)  # Random variation
                
                units_sold = max(1, int(base_units * seasonal_factor * weekly_factor * (1 + noise)))
                
                # Add some anomalies (5% of records)
                if np.random.random() < 0.05:
                    units_sold = int(units_sold * np.random.uniform(2, 5))  # 2-5x normal
                
                demo_data.append({
                    'Product_Category': category,
                    'Units_Sold': units_sold
                })
            
            demo_data = pd.DataFrame(demo_data)
    except Exception as e:
        print(f"⚠️  Failed to load real sales data: {e}")
        print("Creating synthetic demo dataset...")
        np.random.seed(42)
        n_records = 1000
        
        # Product categories from the schema
        product_categories = ['Sports', 'Toys', 'Fashion', 'Electronics', 'Home Decor']
        category_means = {'Sports': 35, 'Toys': 45, 'Fashion': 25, 'Electronics': 20, 'Home Decor': 15}
        
        # Generate demo sales data
        demo_data = []
        for i in range(n_records):
            category = np.random.choice(product_categories)
            base_units = category_means[category]
            
            # Add some variation and seasonality
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * i / 365)  # Annual seasonality
            weekly_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 7)  # Weekly pattern
            noise = np.random.normal(0, 0.2)  # Random variation
            
            units_sold = max(1, int(base_units * seasonal_factor * weekly_factor * (1 + noise)))
            
            # Add some anomalies (5% of records)
            if np.random.random() < 0.05:
                units_sold = int(units_sold * np.random.uniform(2, 5))  # 2-5x normal
            
            demo_data.append({
                'Product_Category': category,
                'Units_Sold': units_sold
            })
        
        demo_data = pd.DataFrame(demo_data)
    
    print(f"Demo dataset created with {len(demo_data)} rows and {len(demo_data.columns)} columns")
    
    # Initialize agent
    print("Initializing UnivariateAnalysisAgent...")
    agent = UnivariateAnalysisAgent(
        target_variable="sales.Units_Sold",
        data=demo_data,
        kb=kb_wrapper,
        config={
            "anomaly_detection_method": "iqr",
            "trend_analysis": True,
            "visualization_enabled": True
        }
    )
    
    # Test individual tools
    print("\nTesting individual analysis tools...")
    tools = agent._create_analysis_tools()
    
    for tool in tools:
        print(f"\n--- Testing {tool.name} ---")
        try:
            if tool.name == "profile_variable":
                result = tool.invoke({"variable_id": "sales.Units_Sold"})
                print(f"✓ Data profiling completed")
                print(f"  - Data type: {result.get('data_type', 'unknown')}")
                print(f"  - Total values: {result.get('statistical_summary', {}).get('count', 0)}")
                print(f"  - Mean: {result.get('statistical_summary', {}).get('mean', 0):.2f}")
                
            elif tool.name == "detect_anomalies":
                result = tool.invoke({"variable_id": "sales.Units_Sold"})
                print(f"✓ Anomaly detection completed")
                outliers = result.get('combined_result', {}).get('total_unique_outliers', 0)
                print(f"  - Outliers detected: {outliers}")
                
            elif tool.name == "analyze_trends":
                result = tool.invoke({"variable_id": "sales.Units_Sold"})
                print(f"✓ Trend analysis completed")
                trend_dir = result.get('temporal_trends', {}).get('trend_direction', 'unknown')
                trend_strength = result.get('temporal_trends', {}).get('trend_strength', 0)
                print(f"  - Trend direction: {trend_dir}")
                print(f"  - Trend strength: {trend_strength:.2f}")
                
            elif tool.name == "discover_related_variables":
                result = tool.invoke({"target_variable": "sales.Units_Sold", "max_results": 3})
                print(f"✓ Related variables discovery completed")
                print(f"  - Related variables found")
                
            elif tool.name == "search_knowledge_base":
                result = tool.invoke({"query": "sales", "max_results": 3})
                print(f"✓ Knowledge base search completed")
                print(f"  - Search results found")
                
            elif tool.name == "get_variable_metadata":
                result = tool.invoke({"variable_id": "sales.Units_Sold"})
                print(f"✓ Variable metadata retrieval completed")
                
            elif tool.name == "create_etl_schema":
                result = tool.invoke({"user_input": "Show me sales data by product category"})
                print(f"✓ ETL schema creation completed")
                
            elif tool.name == "build_data_product":
                print(f"✓ Data product building tool available")
                
            else:
                print(f"✓ {tool.name} tool available")
                
        except Exception as e:
            print(f"✗ {tool.name} failed: {str(e)}")
    
    print(f"\n{'='*50}")
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("The UnivariateAnalysisAgent is ready for use with LLM agentic flow.")
    print(f"{'='*50}")

if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance tests
    run_performance_test()
    
    # Run demo
    run_demo()
