from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage
from EDA.LLMS.llms import get_llm
from EDA.workflow.eda_workflow_state import EDAWorkflowState
import json
import logging
from typing import Dict, Any, Optional, List
from EDA.agents.eda_analysis.eda_analysis_prompt import prompt
from utils.config_loader import AgentConfigLoader
# from utils.helper import clean_messages_for_agent, msg_to_dict  # Not used
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

class EDAAnalysisAgent:
    def __init__(self, kb=None, intugle_tools=None, target_variable: str = "", timeout: int = 300):
        """
        Initialize the EDA Analysis Agent
        
        Args:
            hypothesis: Individual hypothesis to test
            kb: Knowledge base instance (Intugle integration)
            target_variable: The variable being predicted
            timeout: Analysis timeout in seconds
        """
        self.kb = kb
        self.intugle_tools = intugle_tools
        self.target_variable = target_variable
        self.timeout = timeout
        self.config_loader = AgentConfigLoader()
        
        # Load model configuration
        model_config = self.config_loader.get_model_config("eda_analysis")
        self.llm = get_llm(model_config['provider'], model_config['model'])
        self.prompt = self.config_loader.load_prompt("eda_analysis")
        
        # Agent-specific configuration
        self.analysis_config = self.config_loader.get_agent_config("eda_analysis")
    
    def analyze_hypothesis(self, transformation_config: Dict = None, hypothesis: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze a single hypothesis with statistical analysis and visualizations
        
        Args:
            transformation_config: Configuration for data transformations
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        try:
            results = {
                "hypothesis": hypothesis,
                "etl_schema": {},
                "correlation_analysis": {},
                "visualizations": {},
                "feature_engineering_insights": {},
                "statistical_summary": {},
                "status": "completed"
            }
            
            # Build ETL schema using LLM and Intugle tools
            etl_schema = self._build_etl_schema(hypothesis)
            if not etl_schema:
                results["status"] = "failed"
                results["error"] = "Failed to build ETL schema"
                return results
            
            #results["etl_schema"] = etl_schema
            
            # Fetch data using ETL schema
            data = self._fetch_data(etl_schema)
            if data is None or data.empty:
                results["status"] = "failed"
                results["error"] = "Failed to fetch data"
                return results
            
            # Apply transformations
            transformed_data = self._apply_transformations(data, transformation_config, hypothesis)
            print(f"Transformed data: {transformed_data.head()}")
            transformed_data.to_csv("transformed_data.csv", index=False)
            
            # Perform correlation analysis
            correlation_results = self._perform_correlation_analysis(transformed_data, hypothesis)
            results["correlation_analysis"] = correlation_results
            
            # Generate visualizations
            visualizations = self._generate_visualizations(transformed_data, hypothesis)
            print(f"Visualizations completed")
            results["visualizations"] = visualizations
            
            # Generate feature engineering insights
            # insights = self._generate_feature_insights(correlation_results, transformed_data)
            # results["feature_engineering_insights"] = insights
            
            # Compile statistical summary
            # summary = self._compile_statistical_summary(correlation_results, transformed_data, hypothesis)
            # results["statistical_summary"] = summary
            
            return results
            
        except Exception as e:
            logging.error(f"EDA Analysis failed: {str(e)}")
            return {
                "hypothesis": hypothesis,
                "error": str(e),
                "status": "failed"
            }
    
    def _build_etl_schema(self, hypothesis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build ETL schema using LLM and Intugle tools"""
        if not self.intugle_tools.is_available():
            return None
        
        try:
            # Create query for ETL schema based on hypothesis
            query = self._create_etl_query(hypothesis)
            if not query:
                return None
            
            # Use Intugle tools to create ETL schema
            etl_result = self.intugle_tools.create_etl_schema_with_llm(query)
            
            if "error" in etl_result:
                logging.error(f"ETL schema creation failed: {etl_result['error']}")
                return None

            return etl_result.get("etl_schema", {})

            
        except Exception as e:
            logging.error(f"ETL schema building failed: {str(e)}")
            return None
    
    def _create_etl_query(self, hypothesis: Dict[str, Any]) -> str:
        """Create natural language query for ETL schema based on hypothesis"""
        # Extract information from new hypothesis format
        hypothesis_text = hypothesis.get("hypothesis", "")
        target_var = hypothesis.get("target_variable", {})
        predictor_var = hypothesis.get("predictor_variable", {})
        data_req = hypothesis.get("data_requirements", {})
        
        target_name = target_var.get("name", self.target_variable)
        predictor_name = predictor_var.get("name", "unknown")
        target_alias = target_var.get("alias", "unknown")
        predictor_alias = predictor_var.get("alias", "unknown")
        required_tables = data_req.get("required_tables", [])
        aggregate_by = hypothesis.get("aggregate_by")
        target_measure_func = target_var.get("measure_func", "unknown")
        predictor_measure_func = predictor_var.get("measure_func", "unknown")

        query = f"""
        I need to test the hypothesis: {hypothesis_text}
        
        Target Variable: {target_name}
        Target Variable Alias: {target_alias}
        Predictor Variable: {predictor_name}
        Predictor Variable Alias: {predictor_alias}
        Required Tables: {', '.join(required_tables)}
        Aggregate By: {aggregate_by}
        
        Please create an ETL schema that includes:
        1. {target_name} as the target variable
        1. {target_alias} as the target variable alias
        1. Target Variable Measure Function: {target_measure_func}
        2. {predictor_name} as the predictor variable
        2. {predictor_alias} as the predictor variable alias
        3. Predictor Variable Measure Function: {predictor_measure_func}
        3. All required tables: {', '.join(required_tables)}
        6. Appropriate time periods and filters if needed
        7. {aggregate_by} as the aggregate by fields
        
        The schema should support correlation analysis, statistical testing, and visualization.
        These are indicative fields, you can use profiles from knowledge base to identify the fields and table names accurately. Never make up field and table ids and use alias as the name.
        """
        return query
    
    def _fetch_data(self, etl_schema: Dict) -> Optional[pd.DataFrame]:
        """Fetch data using ETL schema and DataProductBuilder"""
        if not etl_schema:
            return None
        
        try:
            # Use Intugle tools to build data product
            print(f"ETL schema: {etl_schema}")
            if not self.intugle_tools.is_available():
                return "Intugle tools not available"
            build_result = self.intugle_tools.build_dataproduct(etl_schema)
            print(f"Build result: {build_result}")
            if "error" in build_result:
                logging.error(f"Data product build failed: {build_result['error']}")
                return None
            
            # Extract dataframe from build result
            data_product = build_result.get("build_result")
            if data_product and hasattr(data_product, 'to_df'):
                print(data_product.to_df().head())
                return data_product.to_df()
            else:
                logging.error("Unable to extract dataframe from data product")
                return None
                
        except Exception as e:
            logging.error(f"Data fetching failed: {str(e)}")
            return None
    
    def _apply_transformations(self, data: pd.DataFrame, config: Dict, hypothesis: Dict[str, Any]) -> pd.DataFrame:
        """Apply transformations to the data based on hypothesis specifications"""
        transformed_data = data.copy()
        
        # Get transformation specifications from hypothesis
        target_var = hypothesis.get("target_variable", {})
        predictor_var =hypothesis.get("predictor_variable", {})
        
        target_name = target_var.get("alias", self.target_variable)
        predictor_name = predictor_var.get("alias", "unknown")
        target_transformation = target_var.get("transformation", "none")
        predictor_transformation = predictor_var.get("transformation", "none")
        
        # Apply target variable transformation
        if target_name in transformed_data.columns and target_transformation != "none":
            transformed_data[target_name] = self._apply_single_transformation(
                transformed_data[target_name], target_transformation
            )
        
        # Apply predictor variable transformation
        if predictor_name in transformed_data.columns and predictor_transformation != "none":
            transformed_data[predictor_name] = self._apply_single_transformation(
                transformed_data[predictor_name], predictor_transformation
            )
        
        # Apply any additional transformations from config
        if config:
            for var, transformation in config.items():
                if var in transformed_data.columns:
                    transformed_data[var] = self._apply_single_transformation(
                        transformed_data[var], transformation
                    )
        
        return transformed_data
    
    def _apply_single_transformation(self, series: pd.Series, transformation: str) -> pd.Series:
        """Apply a single transformation to a pandas Series"""
        # These need to be updated to be done by aggregation groups - especially lagged and seasonal effects
        if transformation == "log" or transformation == "log_log" or transformation == "log_linear":
            return np.log1p(series)
        elif transformation == "sqrt":
            return np.sqrt(series)
        elif transformation == "standardize":
            return (series - series.mean()) / series.std()
        elif transformation == "lag_1":
            return series.shift(1)
        elif transformation == "polynomial_2":
            return series ** 2
        elif transformation == "exponential":
            return np.exp(series)
        elif transformation == "seasonal_decomposition":
            # Simple seasonal adjustment (could be enhanced and replaced wth stl decomposition)
            return series - series.rolling(window=12, center=True).mean()
        else:
            return series
    
    def _perform_correlation_analysis(self, data: pd.DataFrame, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive correlation analysis"""
        # Extract variable names from hypothesis
        target_var = hypothesis.get("target_variable", {})
        predictor_var = hypothesis.get("predictor_variable", {})
        #Extract aggregate by from hypothesis
        aggregate_by = hypothesis.get("aggregate_by", {})
        #create a list of aggregate by fields & seperate out non datetime fields
        aggregate_by_fields = [field for field in aggregate_by.keys() if not field.startswith("date")]
        print(f"Aggregate by fields: {aggregate_by_fields}")

        #min-max scaling of the data by the aggregate by fields
        def safe_min_max_scaling(x):
            """Safe min-max scaling with NaN imputation"""
            
            if x.empty or len(x) == 0:
                return x
            
            # Impute NaN values using forward/backward fill
            if x.isna().any():
                x = x.fillna(method='ffill').fillna(method='bfill')
                # If still NaN (all values were NaN), use 0
                x = x.fillna(0)

            # Apply scaling
            if len(x) <= 1:
                return x * 0
            
            x_min, x_max = x.min(), x.max()
            if x_max == x_min:
                return x * 0
            return (x - x_min) / (x_max - x_min)
        target_name = target_var.get("alias", self.target_variable)
        predictor_name = predictor_var.get("alias", "unknown")
        if len(aggregate_by_fields) > 0:
            print(f"Applying minmax scaling to {target_name} and {predictor_name} by {aggregate_by_fields}")
            data[[target_name,predictor_name]] = data.groupby(aggregate_by_fields)[[target_name,predictor_name]].transform(safe_min_max_scaling)
        if predictor_name not in data.columns or target_name not in data.columns:
            return {"error": f"Required variables not found in data: {target_name}, {predictor_name}"}
        
        data.to_csv("data_post_minmax_scaling.csv", index=False)

        # Calculate different correlation types
        try:
            #ignore na and nan value rows before calculating correlation
            # Drop rows where either target or predictor has missing values
            data = data.dropna(subset=[target_name, predictor_name])
            
            # Additional check for infinite values
            data = data.replace([np.inf, -np.inf], np.nan).dropna()
            
            # Check if we have enough data after cleaning
            if len(data) < 2:
                return {"error": f"Insufficient data after cleaning: {len(data)} rows remaining"}
            data.to_csv("data_for_correlation.csv", index=False)
            pearson_corr, pearson_p = stats.pearsonr(data[target_name], data[predictor_name])
            spearman_corr, spearman_p = stats.spearmanr(data[target_name], data[predictor_name])
            
            # Calculate additional statistics
            n_samples = len(data)
            target_mean = data[target_name].mean()
            predictor_mean = data[predictor_name].mean()
            target_std = data[target_name].std()
            predictor_std = data[predictor_name].std()
            
            return {
                "correlations": {
                    "pearson": {
                        "correlation": float(pearson_corr),
                        "p_value": float(pearson_p),
                        "significant": pearson_p < 0.05
                    },
                    "spearman": {
                        "correlation": float(spearman_corr),
                        "p_value": float(spearman_p),
                        "significant": spearman_p < 0.05
                    }
                },
                "variables": {
                    "target": target_name,
                    "predictor": predictor_name
                },
                "statistics": {
                    "sample_size": n_samples,
                    "target_mean": float(target_mean),
                    "predictor_mean": float(predictor_mean),
                    "target_std": float(target_std),
                    "predictor_std": float(predictor_std)
                },
                "hypothesis_support": self._evaluate_hypothesis_support(pearson_corr, pearson_p, hypothesis = hypothesis)
            }
        except Exception as e:
            return {"error": f"Correlation analysis failed: {str(e)}"}
    
    def _evaluate_hypothesis_support(self, correlation: float, p_value: float, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate how well the data supports the hypothesis"""
        expected_impact = hypothesis.get("expected_impact", "unknown")
        confidence = hypothesis.get("confidence", 0.5)
        
        # Determine if correlation direction matches expected impact
        direction_match = False
        if expected_impact == "positive" and correlation > 0:
            direction_match = True
        elif expected_impact == "negative" and correlation < 0:
            direction_match = True
        elif expected_impact == "unknown":
            direction_match = True  # Neutral for unknown
        
        # Calculate support score
        significance = p_value < 0.05
        strength = abs(correlation)
        
        support_score = 0.0
        if significance and direction_match:
            support_score = min(1.0, strength + 0.3)  # Bonus for significance and direction
        elif direction_match:
            support_score = strength * 0.7  # Partial credit for direction match
        elif significance:
            support_score = strength * 0.5  # Partial credit for significance
        
        return {
            "direction_match": direction_match,
            "significant": significance,
            "correlation_strength": strength,
            "support_score": support_score,
            "expected_impact": expected_impact,
            "actual_correlation": correlation
        }
    
    def _generate_visualizations(self, data: pd.DataFrame, hypothesis: Dict[str, Any]) -> Dict[str, str]:
        """Generate visualizations for the analysis"""
        # Extract variable names from hypothesis
        target_var = hypothesis.get("target_variable", {})
        predictor_var = hypothesis.get("predictor_variable", {})
        
        target_name = target_var.get("alias", self.target_variable)
        predictor_name = predictor_var.get("alias", "unknown")
        
        if predictor_name not in data.columns or target_name not in data.columns:
            return {"error": f"Required variables not found for visualization: {target_name}, {predictor_name}"}
        
        visualizations = {}
        
        # Create visualizations directory if it doesn't exist
        import os
        os.makedirs("visualizations", exist_ok=True)
        
        try:
            # Scatter plot with trend line
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=data, x=predictor_name, y=target_name)
            sns.regplot(data=data, x=predictor_name, y=target_name, scatter=False, color='red')
            plt.title(f"Relationship between {predictor_name} and {target_name}")
            scatter_path = f"visualizations/scatter_{predictor_name}_{target_name}.png"
            plt.savefig(scatter_path)
            plt.close()
            visualizations["scatter_plot"] = scatter_path
            
            # Correlation heatmap
            plt.figure(figsize=(8, 6))
            corr_matrix = data[[target_name, predictor_name]].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title("Correlation Heatmap")
            heatmap_path = f"visualizations/heatmap_{predictor_name}_{target_name}.png"
            plt.savefig(heatmap_path)
            plt.close()
            visualizations["correlation_heatmap"] = heatmap_path
            
        except Exception as e:
            logging.error(f"Visualization generation failed: {str(e)}")
            visualizations["error"] = str(e)
        
        return visualizations
    
    def _generate_feature_insights(self, correlation_results: Dict, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate feature engineering recommendations"""
        insights = {
            "engineering_recommendations": [],
            "relationship_strength": "weak",
            "feature_importance": "low"
        }
        
        if "correlations" in correlation_results:
            pearson_corr = abs(correlation_results["correlations"]["pearson"]["correlation"])
            
            if pearson_corr > 0.7:
                insights["relationship_strength"] = "strong"
                insights["feature_importance"] = "high"
                insights["engineering_recommendations"].append("Consider as primary feature")
            elif pearson_corr > 0.3:
                insights["relationship_strength"] = "moderate"
                insights["feature_importance"] = "medium"
                insights["engineering_recommendations"].append("Consider as secondary feature")
            else:
                insights["engineering_recommendations"].append("Consider feature engineering or interaction terms")
        
        return insights
    
    def _compile_statistical_summary(self, correlation_results: Dict, data: pd.DataFrame,hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """Compile comprehensive statistical summary"""
        influencing_var = hypothesis.get("variable", "")
        
        summary = {
            "sample_size": len(data),
            "target_variable_stats": {},
            "influencing_variable_stats": {},
            "correlation_summary": {}
        }
        
        if self.target_variable in data.columns:
            summary["target_variable_stats"] = {
                "mean": float(data[self.target_variable].mean()),
                "std": float(data[self.target_variable].std()),
                "min": float(data[self.target_variable].min()),
                "max": float(data[self.target_variable].max())
            }
        
        if influencing_var in data.columns:
            summary["influencing_variable_stats"] = {
                "mean": float(data[influencing_var].mean()),
                "std": float(data[influencing_var].std()),
                "min": float(data[influencing_var].min()),
                "max": float(data[influencing_var].max())
            }
        
        if "correlations" in correlation_results:
            summary["correlation_summary"] = {
                "primary_correlation": correlation_results["correlations"]["pearson"]["correlation"],
                "significance": correlation_results["correlations"]["pearson"]["significant"]
            }
        
        return summary
    
    def process(self, state: EDAWorkflowState):
        """
        Process method for LangGraph integration
        
        Args:
            state: EDAWorkflowState containing workflow state
            
        Returns:
            Updated state with EDA analysis results
        """
        try:
            # Extract parameters from state
            hypothesis = state.get("current_hypothesis")
            transformation_config = state.get("transformation_config", {})
            target_variable = state.get("target_variable", "")
            
            # Update agent with current hypothesis
            hypothesis = hypothesis
            self.target_variable = target_variable
            
            # Execute analysis
            analysis_results = self.analyze_hypothesis(transformation_config, hypothesis)
            
            # Update state with results
            updated_state = state.copy()
            
            # Append to existing results instead of overwriting
            existing_results = updated_state.get("eda_analysis_results", [])
            if not isinstance(existing_results, list):
                existing_results = [existing_results] if existing_results else []
            
            existing_results.append(analysis_results)
            updated_state["eda_analysis_results"] = existing_results
            updated_state["current_agent"] = "eda_analysis"
            updated_state["execution_status"] = "completed"
            return updated_state
            
        except Exception as e:
            logging.error(f"EDA Analysis agent processing failed: {str(e)}")
            updated_state = state.copy()
            updated_state["error_messages"] = updated_state.get("error_messages", [])
            updated_state["error_messages"].append(f"EDA Analysis failed: {str(e)}")
            updated_state["execution_status"] = "failed"
            return updated_state

def eda_analysis_agent(state: EDAWorkflowState):
    """
    LangGraph node function for EDA analysis
    
    Args:
        state: EDAWorkflowState containing workflow state
        
    Returns:
        Updated state with EDA analysis results
    """
    # Extract configuration from state
    config = state.get("config", {})
    kb = state.get("kb")
    
    # Initialize agent
    agent = EDAAnalysisAgent(
        hypothesis=state.get("current_hypothesis", {}),
        kb=kb,
        target_variable=state.get("target_variable", ""),
        timeout=config.get("eda_analysis_config", {}).get("timeout", 300)
    )
    
    # Process state
    return agent.process(state)