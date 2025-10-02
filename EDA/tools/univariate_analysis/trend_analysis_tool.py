"""
Trend Analysis Tool for Univariate Analysis

This tool provides comprehensive trend analysis capabilities for target variables
including temporal trends, seasonality detection, and volatility analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class TrendType(Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    CYCLICAL = "cyclical"
    SEASONAL = "seasonal"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

class SeasonalityType(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    NONE = "none"

@dataclass
class TrendResult:
    """Result of trend analysis"""
    trend_type: TrendType
    trend_strength: float
    trend_direction: str
    trend_significance: float
    growth_rate: Optional[float] = None
    volatility: Optional[float] = None
    seasonality: Optional[SeasonalityType] = None
    seasonal_strength: Optional[float] = None
    cyclical_period: Optional[int] = None
    confidence_level: float = 0.95

class TrendAnalysisTool:
    """Tool for analyzing trends and patterns in variables"""
    
    def __init__(self, kb=None, data_product_builder=None):
        """
        Initialize the trend analysis tool
        
        Args:
            kb: Intugle KnowledgeBuilder instance
            data_product_builder: Intugle DataProductBuilder instance
        """
        self.kb = kb
        self.data_product_builder = data_product_builder
        self.logger = logging.getLogger(__name__)
    
    def analyze_trends(self, 
                      variable_id: str, 
                      data: Optional[pd.DataFrame] = None,
                      time_column: Optional[str] = None,
                      confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Analyze trends and patterns in a variable
        
        Args:
            variable_id: ID of the variable to analyze
            data: Optional DataFrame containing the data
            time_column: Optional time column for temporal analysis
            confidence_level: Confidence level for statistical tests
            
        Returns:
            Dictionary containing trend analysis results
        """
        try:
            # Get data if not provided
            if data is None:
                data = self._fetch_variable_data(variable_id)
            
            if data is None or data.empty:
                return self._create_empty_trend_result(variable_id)
            
            # Extract column name from variable_id
            column_name = variable_id.split('.')[-1] if '.' in variable_id else variable_id
            
            if column_name not in data.columns:
                self.logger.error(f"Column {column_name} not found in data")
                return self._create_empty_trend_result(variable_id)
            
            column_data = data[column_name]
            
            # Analyze temporal trends
            temporal_trends = self._analyze_temporal_trends(column_data, time_column, confidence_level)
            
            # Analyze seasonality
            seasonality_analysis = self._analyze_seasonality(column_data, time_column)
            
            # Analyze volatility
            volatility_analysis = self._analyze_volatility(column_data)
            
            # Analyze cyclical patterns
            cyclical_analysis = self._analyze_cyclical_patterns(column_data)
            
            # Analyze growth patterns
            growth_analysis = self._analyze_growth_patterns(column_data, time_column)
            
            # Combine results
            combined_trend = self._combine_trend_results(
                temporal_trends, 
                seasonality_analysis, 
                volatility_analysis,
                cyclical_analysis,
                growth_analysis
            )
            
            # Generate trend recommendations
            recommendations = self._generate_trend_recommendations(combined_trend, variable_id)
            
            return {
                "variable_id": variable_id,
                "column_name": column_name,
                "temporal_trends": temporal_trends,
                "seasonality_analysis": seasonality_analysis,
                "volatility_analysis": volatility_analysis,
                "cyclical_analysis": cyclical_analysis,
                "growth_analysis": growth_analysis,
                "combined_trend": combined_trend,
                "recommendations": recommendations,
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
                "confidence_level": confidence_level
            }
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed for {variable_id}: {str(e)}")
            return self._create_error_trend_result(variable_id, str(e))
    
    def _analyze_temporal_trends(self, 
                               column_data: pd.Series, 
                               time_column: Optional[str] = None,
                               confidence_level: float = 0.95) -> Dict[str, Any]:
        """Analyze temporal trends in the data"""
        try:
            # Remove null values
            clean_data = column_data.dropna()
            
            if len(clean_data) < 3:
                return {
                    "trend_type": TrendType.UNKNOWN.value,
                    "trend_strength": 0.0,
                    "trend_direction": "unknown",
                    "trend_significance": 0.0,
                    "insufficient_data": True
                }
            
            # Create time index if not provided
            if time_column is None:
                time_index = pd.RangeIndex(len(clean_data))
            else:
                time_index = clean_data.index
            
            # Perform linear regression to detect trend
            x = np.arange(len(clean_data))
            y = clean_data.values
            
            # Calculate linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Determine trend type and strength
            trend_strength = abs(r_value)
            trend_direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            
            # Determine trend significance
            trend_significance = 1 - p_value if p_value is not None else 0.0
            
            # Classify trend type
            if trend_significance < (1 - confidence_level):
                trend_type = TrendType.STABLE.value
            elif trend_strength > 0.7:
                trend_type = TrendType.INCREASING.value if slope > 0 else TrendType.DECREASING.value
            elif trend_strength > 0.3:
                trend_type = "weak_" + trend_direction
            else:
                trend_type = TrendType.STABLE.value
            
            # Calculate growth rate
            growth_rate = None
            if len(clean_data) > 1:
                first_value = clean_data.iloc[0]
                last_value = clean_data.iloc[-1]
                if first_value != 0:
                    growth_rate = ((last_value - first_value) / first_value) * 100
            
            return {
                "trend_type": trend_type,
                "trend_strength": trend_strength,
                "trend_direction": trend_direction,
                "trend_significance": trend_significance,
                "slope": slope,
                "r_squared": r_value ** 2,
                "p_value": p_value,
                "growth_rate": growth_rate,
                "insufficient_data": False
            }
            
        except Exception as e:
            self.logger.error(f"Temporal trend analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_seasonality(self, 
                           column_data: pd.Series, 
                           time_column: Optional[str] = None) -> Dict[str, Any]:
        """Analyze seasonality patterns in the data"""
        try:
            # Remove null values
            clean_data = column_data.dropna()
            
            if len(clean_data) < 12:  # Need at least 12 data points for seasonality
                return {
                    "seasonality_type": SeasonalityType.NONE.value,
                    "seasonal_strength": 0.0,
                    "seasonal_period": None,
                    "insufficient_data": True
                }
            
            # Create time index
            if time_column is None:
                time_index = pd.RangeIndex(len(clean_data))
            else:
                time_index = clean_data.index
            
            # Convert to time series if possible
            if hasattr(time_index, 'to_pydatetime'):
                ts = pd.Series(clean_data.values, index=time_index)
            else:
                ts = pd.Series(clean_data.values, index=pd.date_range(start='2020-01-01', periods=len(clean_data), freq='D'))
            
            # Analyze different seasonal periods
            seasonal_periods = [7, 30, 90, 365]  # Weekly, monthly, quarterly, yearly
            seasonal_results = {}
            
            for period in seasonal_periods:
                if len(clean_data) >= period * 2:  # Need at least 2 full periods
                    try:
                        # Calculate seasonal strength using autocorrelation
                        autocorr = ts.autocorr(lag=period)
                        seasonal_results[period] = abs(autocorr) if not pd.isna(autocorr) else 0.0
                    except:
                        seasonal_results[period] = 0.0
            
            # Find strongest seasonal pattern
            if seasonal_results:
                strongest_period = max(seasonal_results, key=seasonal_results.get)
                strongest_strength = seasonal_results[strongest_period]
                
                # Determine seasonality type
                if strongest_strength > 0.5:
                    if strongest_period == 7:
                        seasonality_type = SeasonalityType.WEEKLY.value
                    elif strongest_period == 30:
                        seasonality_type = SeasonalityType.MONTHLY.value
                    elif strongest_period == 90:
                        seasonality_type = SeasonalityType.QUARTERLY.value
                    elif strongest_period == 365:
                        seasonality_type = SeasonalityType.YEARLY.value
                    else:
                        seasonality_type = "custom"
                else:
                    seasonality_type = SeasonalityType.NONE.value
                    strongest_strength = 0.0
            else:
                seasonality_type = SeasonalityType.NONE.value
                strongest_strength = 0.0
                strongest_period = None
            
            return {
                "seasonality_type": seasonality_type,
                "seasonal_strength": strongest_strength,
                "seasonal_period": strongest_period,
                "all_periods": seasonal_results,
                "insufficient_data": False
            }
            
        except Exception as e:
            self.logger.error(f"Seasonality analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_volatility(self, column_data: pd.Series) -> Dict[str, Any]:
        """Analyze volatility in the data"""
        try:
            # Remove null values
            clean_data = column_data.dropna()
            
            if len(clean_data) < 2:
                return {
                    "volatility": 0.0,
                    "volatility_type": "unknown",
                    "coefficient_of_variation": 0.0,
                    "insufficient_data": True
                }
            
            # Calculate basic volatility metrics
            mean_value = clean_data.mean()
            std_value = clean_data.std()
            
            # Coefficient of variation
            cv = (std_value / mean_value) * 100 if mean_value != 0 else 0.0
            
            # Calculate rolling volatility if enough data points
            rolling_volatility = None
            if len(clean_data) >= 10:
                try:
                    rolling_std = clean_data.rolling(window=5).std()
                    rolling_volatility = rolling_std.mean()
                except:
                    rolling_volatility = std_value
            
            # Classify volatility
            if cv < 10:
                volatility_type = "low"
            elif cv < 30:
                volatility_type = "moderate"
            elif cv < 50:
                volatility_type = "high"
            else:
                volatility_type = "very_high"
            
            # Calculate volatility trend
            volatility_trend = None
            if len(clean_data) >= 20:
                try:
                    # Split data into two halves and compare volatility
                    mid_point = len(clean_data) // 2
                    first_half = clean_data[:mid_point]
                    second_half = clean_data[mid_point:]
                    
                    first_vol = first_half.std()
                    second_vol = second_half.std()
                    
                    if second_vol > first_vol * 1.2:
                        volatility_trend = "increasing"
                    elif second_vol < first_vol * 0.8:
                        volatility_trend = "decreasing"
                    else:
                        volatility_trend = "stable"
                except:
                    volatility_trend = "unknown"
            
            return {
                "volatility": std_value,
                "volatility_type": volatility_type,
                "coefficient_of_variation": cv,
                "rolling_volatility": rolling_volatility,
                "volatility_trend": volatility_trend,
                "mean": mean_value,
                "std": std_value,
                "insufficient_data": False
            }
            
        except Exception as e:
            self.logger.error(f"Volatility analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_cyclical_patterns(self, column_data: pd.Series) -> Dict[str, Any]:
        """Analyze cyclical patterns in the data"""
        try:
            # Remove null values
            clean_data = column_data.dropna()
            
            if len(clean_data) < 20:  # Need sufficient data for cyclical analysis
                return {
                    "cyclical_period": None,
                    "cyclical_strength": 0.0,
                    "cyclical_type": "none",
                    "insufficient_data": True
                }
            
            # Convert to numpy array for analysis
            data_array = clean_data.values
            
            # Find peaks and troughs
            peaks, _ = find_peaks(data_array, distance=3)
            troughs, _ = find_peaks(-data_array, distance=3)
            
            # Calculate cycle lengths
            cycle_lengths = []
            if len(peaks) > 1:
                peak_cycles = np.diff(peaks)
                cycle_lengths.extend(peak_cycles)
            
            if len(troughs) > 1:
                trough_cycles = np.diff(troughs)
                cycle_lengths.extend(trough_cycles)
            
            # Determine dominant cycle
            if cycle_lengths:
                dominant_cycle = int(np.median(cycle_lengths))
                cycle_strength = len(cycle_lengths) / len(clean_data) * 100
                
                # Classify cycle type
                if dominant_cycle <= 5:
                    cycle_type = "short_term"
                elif dominant_cycle <= 20:
                    cycle_type = "medium_term"
                else:
                    cycle_type = "long_term"
            else:
                dominant_cycle = None
                cycle_strength = 0.0
                cycle_type = "none"
            
            # Calculate cyclical strength using autocorrelation
            autocorr_strength = 0.0
            if dominant_cycle and len(clean_data) > dominant_cycle * 2:
                try:
                    autocorr = pd.Series(data_array).autocorr(lag=dominant_cycle)
                    autocorr_strength = abs(autocorr) if not pd.isna(autocorr) else 0.0
                except:
                    autocorr_strength = 0.0
            
            return {
                "cyclical_period": dominant_cycle,
                "cyclical_strength": max(cycle_strength, autocorr_strength * 100),
                "cyclical_type": cycle_type,
                "cycle_lengths": cycle_lengths,
                "peaks_count": len(peaks),
                "troughs_count": len(troughs),
                "autocorr_strength": autocorr_strength,
                "insufficient_data": False
            }
            
        except Exception as e:
            self.logger.error(f"Cyclical pattern analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_growth_patterns(self, 
                               column_data: pd.Series, 
                               time_column: Optional[str] = None) -> Dict[str, Any]:
        """Analyze growth patterns in the data"""
        try:
            # Remove null values
            clean_data = column_data.dropna()
            
            if len(clean_data) < 3:
                return {
                    "growth_rate": 0.0,
                    "growth_type": "unknown",
                    "growth_acceleration": 0.0,
                    "insufficient_data": True
                }
            
            # Calculate overall growth rate
            first_value = clean_data.iloc[0]
            last_value = clean_data.iloc[-1]
            
            if first_value != 0:
                overall_growth_rate = ((last_value - first_value) / first_value) * 100
            else:
                overall_growth_rate = 0.0
            
            # Calculate period-over-period growth rates
            period_growth_rates = []
            for i in range(1, len(clean_data)):
                if clean_data.iloc[i-1] != 0:
                    growth_rate = ((clean_data.iloc[i] - clean_data.iloc[i-1]) / clean_data.iloc[i-1]) * 100
                    period_growth_rates.append(growth_rate)
            
            # Analyze growth acceleration
            growth_acceleration = 0.0
            if len(period_growth_rates) >= 3:
                # Calculate second derivative (acceleration)
                first_half = np.mean(period_growth_rates[:len(period_growth_rates)//2])
                second_half = np.mean(period_growth_rates[len(period_growth_rates)//2:])
                growth_acceleration = second_half - first_half
            
            # Classify growth type
            if abs(overall_growth_rate) < 1:
                growth_type = "stable"
            elif overall_growth_rate > 10:
                growth_type = "rapid_growth"
            elif overall_growth_rate > 0:
                growth_type = "moderate_growth"
            elif overall_growth_rate > -10:
                growth_type = "moderate_decline"
            else:
                growth_type = "rapid_decline"
            
            # Calculate growth volatility
            growth_volatility = np.std(period_growth_rates) if period_growth_rates else 0.0
            
            return {
                "growth_rate": overall_growth_rate,
                "growth_type": growth_type,
                "growth_acceleration": growth_acceleration,
                "growth_volatility": growth_volatility,
                "period_growth_rates": period_growth_rates,
                "growth_consistency": 1.0 / (1.0 + growth_volatility) if growth_volatility > 0 else 1.0,
                "insufficient_data": False
            }
            
        except Exception as e:
            self.logger.error(f"Growth pattern analysis failed: {str(e)}")
            return {"error": str(e)}
    
    def _combine_trend_results(self, 
                             temporal_trends: Dict[str, Any],
                             seasonality_analysis: Dict[str, Any],
                             volatility_analysis: Dict[str, Any],
                             cyclical_analysis: Dict[str, Any],
                             growth_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from different trend analyses"""
        try:
            # Determine overall trend type
            overall_trend_type = temporal_trends.get("trend_type", TrendType.UNKNOWN.value)
            
            # Adjust trend type based on other analyses
            if seasonality_analysis.get("seasonal_strength", 0) > 0.5:
                overall_trend_type = "seasonal_" + overall_trend_type
            
            if cyclical_analysis.get("cyclical_strength", 0) > 30:
                overall_trend_type = "cyclical_" + overall_trend_type
            
            if volatility_analysis.get("volatility_type") == "very_high":
                overall_trend_type = "volatile_" + overall_trend_type
            
            # Calculate overall trend strength
            trend_strength = temporal_trends.get("trend_strength", 0.0)
            seasonal_strength = seasonality_analysis.get("seasonal_strength", 0.0)
            cyclical_strength = cyclical_analysis.get("cyclical_strength", 0.0) / 100
            
            overall_strength = max(trend_strength, seasonal_strength, cyclical_strength)
            
            # Determine confidence level
            confidence = min(
                temporal_trends.get("trend_significance", 0.0),
                0.95  # Default confidence
            )
            
            return {
                "overall_trend_type": overall_trend_type,
                "overall_trend_strength": overall_strength,
                "trend_confidence": confidence,
                "primary_driver": self._identify_primary_driver(
                    temporal_trends, seasonality_analysis, cyclical_analysis, volatility_analysis
                ),
                "trend_summary": self._generate_trend_summary(
                    temporal_trends, seasonality_analysis, cyclical_analysis, volatility_analysis, growth_analysis
                )
            }
            
        except Exception as e:
            self.logger.error(f"Trend result combination failed: {str(e)}")
            return {"error": str(e)}
    
    def _identify_primary_driver(self, 
                               temporal_trends: Dict[str, Any],
                               seasonality_analysis: Dict[str, Any],
                               cyclical_analysis: Dict[str, Any],
                               volatility_analysis: Dict[str, Any]) -> str:
        """Identify the primary driver of the trend"""
        try:
            # Compare strengths
            trend_strength = temporal_trends.get("trend_strength", 0.0)
            seasonal_strength = seasonality_analysis.get("seasonal_strength", 0.0)
            cyclical_strength = cyclical_analysis.get("cyclical_strength", 0.0) / 100
            volatility_strength = volatility_analysis.get("coefficient_of_variation", 0.0) / 100
            
            strengths = {
                "temporal_trend": trend_strength,
                "seasonality": seasonal_strength,
                "cyclical_pattern": cyclical_strength,
                "volatility": volatility_strength
            }
            
            primary_driver = max(strengths, key=strengths.get)
            return primary_driver
            
        except Exception as e:
            self.logger.error(f"Primary driver identification failed: {str(e)}")
            return "unknown"
    
    def _generate_trend_summary(self, 
                              temporal_trends: Dict[str, Any],
                              seasonality_analysis: Dict[str, Any],
                              cyclical_analysis: Dict[str, Any],
                              volatility_analysis: Dict[str, Any],
                              growth_analysis: Dict[str, Any]) -> str:
        """Generate a human-readable trend summary"""
        try:
            summary_parts = []
            
            # Temporal trend
            trend_direction = temporal_trends.get("trend_direction", "unknown")
            trend_strength = temporal_trends.get("trend_strength", 0.0)
            if trend_strength > 0.3:
                summary_parts.append(f"Overall {trend_direction} trend (strength: {trend_strength:.2f})")
            
            # Seasonality
            seasonal_type = seasonality_analysis.get("seasonality_type", "none")
            seasonal_strength = seasonality_analysis.get("seasonal_strength", 0.0)
            if seasonal_strength > 0.3:
                summary_parts.append(f"{seasonal_type} seasonality (strength: {seasonal_strength:.2f})")
            
            # Cyclical patterns
            cyclical_type = cyclical_analysis.get("cyclical_type", "none")
            cyclical_strength = cyclical_analysis.get("cyclical_strength", 0.0)
            if cyclical_strength > 30:
                summary_parts.append(f"{cyclical_type} cyclical pattern (strength: {cyclical_strength:.1f}%)")
            
            # Volatility
            volatility_type = volatility_analysis.get("volatility_type", "unknown")
            if volatility_type in ["high", "very_high"]:
                summary_parts.append(f"{volatility_type} volatility")
            
            # Growth
            growth_type = growth_analysis.get("growth_type", "unknown")
            growth_rate = growth_analysis.get("growth_rate", 0.0)
            if abs(growth_rate) > 1:
                summary_parts.append(f"{growth_type} (rate: {growth_rate:.1f}%)")
            
            if not summary_parts:
                return "No significant trends detected"
            
            return "; ".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Trend summary generation failed: {str(e)}")
            return "Unable to generate trend summary"
    
    def _generate_trend_recommendations(self, 
                                      combined_trend: Dict[str, Any], 
                                      variable_id: str) -> List[str]:
        """Generate recommendations based on trend analysis"""
        recommendations = []
        
        try:
            trend_type = combined_trend.get("overall_trend_type", "unknown")
            trend_strength = combined_trend.get("overall_trend_strength", 0.0)
            primary_driver = combined_trend.get("primary_driver", "unknown")
            
            # Trend-specific recommendations
            if "increasing" in trend_type:
                recommendations.append("Consider the upward trend when making predictions")
                recommendations.append("Monitor for potential saturation or inflection points")
            elif "decreasing" in trend_type:
                recommendations.append("Investigate causes of the downward trend")
                recommendations.append("Consider intervention strategies if decline is undesirable")
            
            # Seasonality recommendations
            if "seasonal" in trend_type:
                recommendations.append("Account for seasonality in predictive models")
                recommendations.append("Consider seasonal decomposition for better analysis")
            
            # Volatility recommendations
            if "volatile" in trend_type:
                recommendations.append("High volatility detected - consider smoothing techniques")
                recommendations.append("Use robust statistical methods for analysis")
            
            # Cyclical recommendations
            if "cyclical" in trend_type:
                recommendations.append("Incorporate cyclical patterns in forecasting models")
                recommendations.append("Monitor cycle length and amplitude changes")
            
            # Strength-based recommendations
            if trend_strength > 0.7:
                recommendations.append("Strong trend detected - reliable for forecasting")
            elif trend_strength > 0.3:
                recommendations.append("Moderate trend - use with caution in predictions")
            else:
                recommendations.append("Weak trend - consider other factors for predictions")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendation generation failed: {str(e)}")
            return [f"Unable to generate recommendations: {str(e)}"]
    
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
                name=f"trend_{variable_id.replace('.', '_')}",
                fields=[field]
            )
            
            # Generate and execute query
            dataset = self.data_product_builder.build(etl_model)
            return dataset.to_df()
            
        except Exception as e:
            self.logger.error(f"Failed to fetch data for {variable_id}: {str(e)}")
            return None
    
    def _create_empty_trend_result(self, variable_id: str) -> Dict[str, Any]:
        """Create empty trend result for missing data"""
        return {
            "variable_id": variable_id,
            "column_name": variable_id.split('.')[-1] if '.' in variable_id else variable_id,
            "temporal_trends": {"insufficient_data": True},
            "seasonality_analysis": {"insufficient_data": True},
            "volatility_analysis": {"insufficient_data": True},
            "cyclical_analysis": {"insufficient_data": True},
            "growth_analysis": {"insufficient_data": True},
            "combined_trend": {"overall_trend_type": "unknown"},
            "recommendations": [],
            "error": "No data available"
        }
    
    def _create_error_trend_result(self, variable_id: str, error_message: str) -> Dict[str, Any]:
        """Create error trend result"""
        return {
            "variable_id": variable_id,
            "column_name": variable_id.split('.')[-1] if '.' in variable_id else variable_id,
            "error": error_message,
            "analysis_timestamp": pd.Timestamp.now().isoformat()
        }
