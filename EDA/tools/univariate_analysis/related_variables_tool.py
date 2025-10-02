"""
Related Variables Discovery Tool for Univariate Analysis

This tool provides capabilities to discover variables related to a target variable
using Intugle's semantic search and relationship prediction features.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum

class RelationshipType(Enum):
    STRONG_CORRELATION = "strong_correlation"
    MODERATE_CORRELATION = "moderate_correlation"
    WEAK_CORRELATION = "weak_correlation"
    CAUSAL_RELATIONSHIP = "causal_relationship"
    HIERARCHICAL_RELATIONSHIP = "hierarchical_relationship"
    TEMPORAL_RELATIONSHIP = "temporal_relationship"
    SEMANTIC_SIMILARITY = "semantic_similarity"

@dataclass
class RelatedVariable:
    """Information about a related variable"""
    variable_id: str
    table_name: str
    column_name: str
    relationship_type: RelationshipType
    similarity_score: float
    correlation_score: Optional[float] = None
    description: str = ""
    tags: List[str] = None
    category: str = ""
    data_type: str = ""
    uniqueness: float = 0.0
    completeness: float = 0.0
    is_pii: bool = False

class RelatedVariablesTool:
    """Tool for discovering variables related to a target variable"""
    
    def __init__(self, kb=None, semantic_search=None, data_product_builder=None):
        """
        Initialize the related variables discovery tool
        
        Args:
            kb: Intugle KnowledgeBuilder instance
            semantic_search: Intugle SemanticSearch instance
            data_product_builder: Intugle DataProductBuilder instance
        """
        self.kb = kb
        self.semantic_search = semantic_search
        self.data_product_builder = data_product_builder
        self.logger = logging.getLogger(__name__)
    
    def discover_related_variables(self, 
                                 target_variable: str,
                                 search_queries: Optional[List[str]] = None,
                                 max_results: int = 20,
                                 min_similarity: float = 0.3) -> Dict[str, Any]:
        """
        Discover variables related to the target variable
        
        Args:
            target_variable: The target variable to find related variables for
            search_queries: Optional custom search queries
            max_results: Maximum number of related variables to return
            min_similarity: Minimum similarity score threshold
            
        Returns:
            Dictionary containing related variables and analysis results
        """
        try:
            # Generate search queries if not provided
            if search_queries is None:
                search_queries = self._generate_search_queries(target_variable)
            
            # Discover related variables using different methods
            semantic_results = self._discover_semantic_relationships(target_variable, search_queries, max_results, min_similarity)
            predicted_relationships = self._discover_predicted_relationships(target_variable)
            correlation_analysis = self._analyze_correlations(target_variable, semantic_results)
            
            # Combine and rank results
            combined_results = self._combine_relationship_results(
                semantic_results, 
                predicted_relationships, 
                correlation_analysis
            )
            
            # Analyze relationship patterns
            relationship_patterns = self._analyze_relationship_patterns(combined_results)
            
            # Generate recommendations
            recommendations = self._generate_relationship_recommendations(combined_results, target_variable)
            
            return {
                "target_variable": target_variable,
                "related_variables": combined_results,
                "relationship_patterns": relationship_patterns,
                "recommendations": recommendations,
                "discovery_metadata": {
                    "total_variables_found": len(combined_results),
                    "search_queries_used": search_queries,
                    "min_similarity_threshold": min_similarity,
                    "discovery_timestamp": pd.Timestamp.now().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Related variables discovery failed for {target_variable}: {str(e)}")
            return self._create_error_result(target_variable, str(e))
    
    def _generate_search_queries(self, target_variable: str) -> List[str]:
        """Generate search queries for finding related variables"""
        # Extract base variable name (remove table prefix if present)
        base_name = target_variable.split('.')[-1] if '.' in target_variable else target_variable
        
        # Generate various search queries
        queries = [
            f"variables related to {base_name}",
            f"factors that influence {base_name}",
            f"predictors of {base_name}",
            f"drivers of {base_name}",
            f"determinants of {base_name}",
            f"variables correlated with {base_name}",
            f"features similar to {base_name}",
            f"metrics related to {base_name}",
            f"KPIs for {base_name}",
            f"indicators of {base_name}"
        ]
        
        # Add domain-specific queries if we can infer domain from variable name
        domain_queries = self._generate_domain_specific_queries(base_name)
        queries.extend(domain_queries)
        
        return queries
    
    def _generate_domain_specific_queries(self, base_name: str) -> List[str]:
        """Generate domain-specific search queries based on variable name"""
        queries = []
        
        # Sales/Revenue related
        if any(keyword in base_name.lower() for keyword in ['sales', 'revenue', 'income', 'profit']):
            queries.extend([
                "customer metrics",
                "marketing spend",
                "product features",
                "pricing variables",
                "seasonal factors"
            ])
        
        # Customer related
        elif any(keyword in base_name.lower() for keyword in ['customer', 'user', 'client', 'churn']):
            queries.extend([
                "customer behavior",
                "demographic variables",
                "engagement metrics",
                "satisfaction scores",
                "lifetime value"
            ])
        
        # Operational metrics
        elif any(keyword in base_name.lower() for keyword in ['efficiency', 'performance', 'productivity', 'quality']):
            queries.extend([
                "operational metrics",
                "resource utilization",
                "process variables",
                "quality indicators",
                "performance drivers"
            ])
        
        # Financial metrics
        elif any(keyword in base_name.lower() for keyword in ['cost', 'expense', 'budget', 'financial']):
            queries.extend([
                "financial indicators",
                "cost drivers",
                "budget variables",
                "expense categories",
                "financial ratios"
            ])
        
        return queries
    
    def _discover_semantic_relationships(self, 
                                       target_variable: str, 
                                       search_queries: List[str], 
                                       max_results: int, 
                                       min_similarity: float) -> List[RelatedVariable]:
        """Discover related variables using semantic search"""
        related_variables = []
        
        try:
            if not self.semantic_search:
                self.logger.warning("Semantic search not available")
                return related_variables
            
            # Search for each query
            for query in search_queries:
                try:
                    search_results = self.semantic_search.search(query)
                    
                    if search_results is not None and not search_results.empty:
                        for _, row in search_results.iterrows():
                            # Extract variable information
                            variable_id = row.get("column_id", "")
                            table_name = row.get("table_name", "")
                            column_name = row.get("column_name", "")
                            similarity_score = row.get("score", 0.0)
                            
                            # Skip if similarity is too low
                            if similarity_score < min_similarity:
                                continue
                            
                            # Skip the target variable itself
                            if variable_id == target_variable:
                                continue
                            
                            # Create RelatedVariable object
                            related_var = RelatedVariable(
                                variable_id=variable_id,
                                table_name=table_name,
                                column_name=column_name,
                                relationship_type=RelationshipType.SEMANTIC_SIMILARITY,
                                similarity_score=similarity_score,
                                description=row.get("column_glossary", ""),
                                tags=row.get("column_tags", []),
                                category=row.get("category", ""),
                                data_type=row.get("data_type", ""),
                                uniqueness=row.get("uniqueness", 0.0),
                                completeness=row.get("completeness", 0.0),
                                is_pii=row.get("is_pii", False)
                            )
                            
                            related_variables.append(related_var)
                            
                except Exception as e:
                    self.logger.error(f"Semantic search failed for query '{query}': {str(e)}")
                    continue
            
            # Remove duplicates and sort by similarity score
            unique_variables = {}
            for var in related_variables:
                if var.variable_id not in unique_variables or var.similarity_score > unique_variables[var.variable_id].similarity_score:
                    unique_variables[var.variable_id] = var
            
            # Sort by similarity score and limit results
            sorted_variables = sorted(unique_variables.values(), key=lambda x: x.similarity_score, reverse=True)
            return sorted_variables[:max_results]
            
        except Exception as e:
            self.logger.error(f"Semantic relationship discovery failed: {str(e)}")
            return []
    
    def _discover_predicted_relationships(self, target_variable: str) -> List[RelatedVariable]:
        """Discover related variables using predicted relationships from knowledge base"""
        related_variables = []
        
        try:
            if not self.kb or not hasattr(self.kb, 'links'):
                self.logger.warning("Knowledge base or links not available")
                return related_variables
            
            # Find relationships involving the target variable
            for link in self.kb.links:
                source_field = str(link.source_field_id)
                target_field = str(link.target_field_id)
                
                # Check if target variable is involved in the relationship
                if target_variable in source_field or target_variable in target_field:
                    # Determine the related variable
                    if target_variable in source_field:
                        related_var_id = target_field
                        relationship_strength = self._calculate_relationship_strength(link)
                    else:
                        related_var_id = source_field
                        relationship_strength = self._calculate_relationship_strength(link)
                    
                    # Skip if it's the same variable
                    if related_var_id == target_variable:
                        continue
                    
                    # Get variable details
                    var_details = self._get_variable_details(related_var_id)
                    
                    # Determine relationship type
                    relationship_type = self._classify_relationship_type(link)
                    
                    related_var = RelatedVariable(
                        variable_id=related_var_id,
                        table_name=var_details.get("table_name", ""),
                        column_name=var_details.get("column_name", ""),
                        relationship_type=relationship_type,
                        similarity_score=relationship_strength,
                        description=var_details.get("description", ""),
                        tags=var_details.get("tags", []),
                        category=var_details.get("category", ""),
                        data_type=var_details.get("data_type", ""),
                        uniqueness=var_details.get("uniqueness", 0.0),
                        completeness=var_details.get("completeness", 0.0),
                        is_pii=var_details.get("is_pii", False)
                    )
                    
                    related_variables.append(related_var)
            
            return related_variables
            
        except Exception as e:
            self.logger.error(f"Predicted relationship discovery failed: {str(e)}")
            return []
    
    def _analyze_correlations(self, target_variable: str, related_variables: List[RelatedVariable]) -> Dict[str, float]:
        """Analyze correlations between target variable and related variables"""
        correlations = {}
        
        try:
            if not self.data_product_builder or not related_variables:
                return correlations
            
            # Get target variable data
            target_data = self._fetch_variable_data(target_variable)
            if target_data is None or target_data.empty:
                return correlations
            
            target_column = target_variable.split('.')[-1] if '.' in target_variable else target_variable
            if target_column not in target_data.columns:
                return correlations
            
            target_series = target_data[target_column]
            
            # Calculate correlations with related variables
            for related_var in related_variables:
                try:
                    related_data = self._fetch_variable_data(related_var.variable_id)
                    if related_data is None or related_data.empty:
                        continue
                    
                    related_column = related_var.column_name
                    if related_column not in related_data.columns:
                        continue
                    
                    related_series = related_data[related_column]
                    
                    # Align data (handle different lengths)
                    aligned_data = pd.concat([target_series, related_series], axis=1, join='inner')
                    if aligned_data.empty:
                        continue
                    
                    # Calculate correlation
                    if pd.api.types.is_numeric_dtype(aligned_data.iloc[:, 0]) and pd.api.types.is_numeric_dtype(aligned_data.iloc[:, 1]):
                        correlation = aligned_data.corr().iloc[0, 1]
                        if not pd.isna(correlation):
                            correlations[related_var.variable_id] = abs(correlation)
                            related_var.correlation_score = correlation
                    
                except Exception as e:
                    self.logger.error(f"Correlation analysis failed for {related_var.variable_id}: {str(e)}")
                    continue
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Correlation analysis failed: {str(e)}")
            return {}
    
    def _combine_relationship_results(self, 
                                    semantic_results: List[RelatedVariable],
                                    predicted_results: List[RelatedVariable],
                                    correlations: Dict[str, float]) -> List[RelatedVariable]:
        """Combine results from different discovery methods"""
        combined_results = {}
        
        # Add semantic search results
        for var in semantic_results:
            combined_results[var.variable_id] = var
        
        # Add predicted relationship results
        for var in predicted_results:
            if var.variable_id in combined_results:
                # Update with predicted relationship information
                existing_var = combined_results[var.variable_id]
                if var.similarity_score > existing_var.similarity_score:
                    existing_var.similarity_score = var.similarity_score
                existing_var.relationship_type = var.relationship_type
            else:
                combined_results[var.variable_id] = var
        
        # Add correlation information
        for var_id, correlation in correlations.items():
            if var_id in combined_results:
                combined_results[var_id].correlation_score = correlation
                # Update relationship type based on correlation strength
                if correlation > 0.7:
                    combined_results[var_id].relationship_type = RelationshipType.STRONG_CORRELATION
                elif correlation > 0.4:
                    combined_results[var_id].relationship_type = RelationshipType.MODERATE_CORRELATION
                else:
                    combined_results[var_id].relationship_type = RelationshipType.WEAK_CORRELATION
        
        # Sort by combined score (similarity + correlation)
        def calculate_combined_score(var):
            base_score = var.similarity_score
            if var.correlation_score is not None:
                base_score += abs(var.correlation_score) * 0.5
            return base_score
        
        sorted_results = sorted(combined_results.values(), key=calculate_combined_score, reverse=True)
        return sorted_results
    
    def _analyze_relationship_patterns(self, related_variables: List[RelatedVariable]) -> Dict[str, Any]:
        """Analyze patterns in the discovered relationships"""
        try:
            patterns = {
                "relationship_types": {},
                "table_distribution": {},
                "category_distribution": {},
                "data_type_distribution": {},
                "strength_distribution": {
                    "high": 0,
                    "medium": 0,
                    "low": 0
                }
            }
            
            for var in related_variables:
                # Count relationship types
                rel_type = var.relationship_type.value
                patterns["relationship_types"][rel_type] = patterns["relationship_types"].get(rel_type, 0) + 1
                
                # Count table distribution
                table = var.table_name
                patterns["table_distribution"][table] = patterns["table_distribution"].get(table, 0) + 1
                
                # Count category distribution
                category = var.category
                patterns["category_distribution"][category] = patterns["category_distribution"].get(category, 0) + 1
                
                # Count data type distribution
                data_type = var.data_type
                patterns["data_type_distribution"][data_type] = patterns["data_type_distribution"].get(data_type, 0) + 1
                
                # Categorize by strength
                if var.similarity_score > 0.7 or (var.correlation_score and abs(var.correlation_score) > 0.7):
                    patterns["strength_distribution"]["high"] += 1
                elif var.similarity_score > 0.4 or (var.correlation_score and abs(var.correlation_score) > 0.4):
                    patterns["strength_distribution"]["medium"] += 1
                else:
                    patterns["strength_distribution"]["low"] += 1
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Relationship pattern analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def _generate_relationship_recommendations(self, 
                                             related_variables: List[RelatedVariable], 
                                             target_variable: str) -> List[str]:
        """Generate recommendations based on discovered relationships"""
        recommendations = []
        
        try:
            # High-priority variables
            high_priority = [var for var in related_variables[:5] if var.similarity_score > 0.6]
            if high_priority:
                recommendations.append(f"Focus on {len(high_priority)} high-priority variables with strong relationships to {target_variable}")
            
            # Correlation recommendations
            strong_correlations = [var for var in related_variables if var.correlation_score and abs(var.correlation_score) > 0.7]
            if strong_correlations:
                recommendations.append(f"Consider {len(strong_correlations)} variables with strong correlations for feature engineering")
            
            # Table diversity
            unique_tables = set(var.table_name for var in related_variables)
            if len(unique_tables) > 1:
                recommendations.append(f"Explore relationships across {len(unique_tables)} different tables")
            
            # Data type diversity
            unique_types = set(var.data_type for var in related_variables)
            if len(unique_types) > 1:
                recommendations.append(f"Consider variables of different data types: {', '.join(unique_types)}")
            
            # PII considerations
            pii_variables = [var for var in related_variables if var.is_pii]
            if pii_variables:
                recommendations.append(f"Be cautious with {len(pii_variables)} PII variables in analysis")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {str(e)}")
            return [f"Unable to generate recommendations: {str(e)}"]
    
    def _calculate_relationship_strength(self, link) -> float:
        """Calculate relationship strength based on link metrics"""
        try:
            if link.records_mapped == 0:
                return 0.0
            
            source_coverage = link.source_count_distinct / link.source_count if link.source_count > 0 else 0
            target_coverage = link.target_count_distinct / link.target_count if link.target_count > 0 else 0
            
            # Average coverage as relationship strength
            strength = (source_coverage + target_coverage) / 2
            return min(strength, 1.0)
            
        except Exception as e:
            self.logger.error(f"Relationship strength calculation failed: {str(e)}")
            return 0.0
    
    def _classify_relationship_type(self, link) -> RelationshipType:
        """Classify the type of relationship"""
        try:
            # Simple classification based on counts
            if link.source_count == link.target_count:
                return RelationshipType.HIERARCHICAL_RELATIONSHIP
            elif link.source_count > link.target_count:
                return RelationshipType.CAUSAL_RELATIONSHIP
            else:
                return RelationshipType.TEMPORAL_RELATIONSHIP
                
        except Exception as e:
            self.logger.error(f"Relationship type classification failed: {str(e)}")
            return RelationshipType.SEMANTIC_SIMILARITY
    
    def _get_variable_details(self, variable_id: str) -> Dict[str, Any]:
        """Get variable details from knowledge base"""
        try:
            if not self.kb:
                return {}
            
            return self.kb.get_variable_details(variable_id)
            
        except Exception as e:
            self.logger.error(f"Variable details retrieval failed: {str(e)}")
            return {}
    
    def _fetch_variable_data(self, variable_id: str) -> Optional[pd.DataFrame]:
        """Fetch data for a variable using Intugle DataProductBuilder"""
        try:
            if not self.data_product_builder:
                return None
            
            # Parse table and column from variable_id
            if '.' not in variable_id:
                return None
            
            table_name, column_name = variable_id.split('.', 1)
            
            # Create a simple ETL model to fetch the data
            from intugle.libs.smart_query_generator.models.models import ETLModel, FieldsModel
            
            field = FieldsModel(
                id=column_name,
                name=column_name,
                category="dimension"
            )
            
            etl_model = ETLModel(
                name=f"related_{variable_id.replace('.', '_')}",
                fields=[field]
            )
            
            # Generate and execute query
            dataset = self.data_product_builder.build(etl_model)
            return dataset.to_df()
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {variable_id}: {str(e)}")
            return None
    
    def _create_error_result(self, target_variable: str, error_message: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            "target_variable": target_variable,
            "related_variables": [],
            "relationship_patterns": {},
            "recommendations": [],
            "error": error_message,
            "discovery_timestamp": pd.Timestamp.now().isoformat()
        }
