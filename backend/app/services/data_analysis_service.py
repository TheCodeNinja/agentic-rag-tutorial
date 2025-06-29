import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import json
import io
import base64
import logging
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)

class DataAnalysisService:
    """Service for performing statistical analysis and generating visualizations."""
    
    def __init__(self):
        # Set up matplotlib and seaborn styles
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configure plot settings
        plt.rcParams.update({
            'figure.figsize': (10, 6),
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9
        })
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None  # Convert NaN and infinity to None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        elif isinstance(obj, (float, np.float64, np.float32)):
            if np.isnan(obj) or np.isinf(obj):
                return None  # Handle regular Python floats that are NaN/inf
            return float(obj)
        return obj
    
    def analyze_dataframe(self, df: pd.DataFrame, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Perform comprehensive statistical analysis on a DataFrame."""
        try:
            logger.info(f"Starting {analysis_type} analysis on DataFrame with shape {df.shape}")
            
            analysis = {
                "basic_info": self._get_basic_info(df),
                "statistical_summary": self._get_statistical_summary(df),
                "data_quality": self._assess_data_quality(df),
                "column_types": self._analyze_column_types(df)
            }
            
            if analysis_type == "comprehensive":
                analysis.update({
                    "correlations": self._calculate_correlations(df),
                    "distributions": self._analyze_distributions(df),
                    "outliers": self._detect_outliers(df)
                })
            
            logger.info("Analysis completed successfully")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in analyze_dataframe: {str(e)}")
            raise
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic information about the DataFrame."""
        return {
            "shape": df.shape,
            "column_count": int(len(df.columns)),
            "row_count": int(len(df)),
            "memory_usage": int(df.memory_usage(deep=True).sum()),
            "columns": list(df.columns)
        }
    
    def _get_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistical summary of numerical columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {"message": "No numerical columns found for statistical analysis"}
        
        summary = numeric_df.describe()
        # Convert numpy types to native Python types
        converted_summary = {}
        for col, col_stats in summary.to_dict().items():
            converted_summary[col] = {stat: float(value) if not pd.isna(value) else None 
                                    for stat, value in col_stats.items()}
        
        return {
            "numerical_summary": converted_summary,
            "total_numeric_columns": int(len(numeric_df.columns))
        }
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality by checking for missing values, duplicates, etc."""
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df) * 100)
        
        quality_report = {
            "missing_values": {col: int(val) for col, val in missing_values.to_dict().items()},
            "missing_percentage": {col: float(val) for col, val in missing_percentage.to_dict().items()},
            "duplicate_rows": int(df.duplicated().sum()),
            "total_missing_values": int(df.isnull().sum().sum()),
            "completeness_score": float((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100)
        }
        
        return quality_report
    
    def _analyze_column_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the data types and characteristics of each column."""
        column_analysis = {}
        
        for column in df.columns:
            col_data = df[column]
            
            analysis = {
                "data_type": str(col_data.dtype),
                "unique_count": int(col_data.nunique()),
                "null_count": int(col_data.isnull().sum()),
                "null_percentage": float(col_data.isnull().sum() / len(col_data) * 100)
            }
            
            if col_data.dtype in ['int64', 'float64']:
                analysis.update({
                    "min": self._convert_numpy_types(col_data.min()) if not col_data.empty else None,
                    "max": self._convert_numpy_types(col_data.max()) if not col_data.empty else None,
                    "mean": self._convert_numpy_types(col_data.mean()) if not col_data.empty else None,
                    "median": self._convert_numpy_types(col_data.median()) if not col_data.empty else None,
                    "std": self._convert_numpy_types(col_data.std()) if not col_data.empty else None,
                    "skewness": self._convert_numpy_types(col_data.skew()) if not col_data.empty else None,
                    "kurtosis": self._convert_numpy_types(col_data.kurtosis()) if not col_data.empty else None
                })
            elif col_data.dtype == 'object':
                analysis.update({
                    "most_frequent": str(col_data.mode().iloc[0]) if not col_data.mode().empty else None,
                    "avg_length": self._convert_numpy_types(col_data.astype(str).str.len().mean()) if not col_data.empty else None
                })
            
            column_analysis[column] = analysis
        
        return column_analysis
    
    def _calculate_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate correlations between numerical columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return {"message": "Need at least 2 numerical columns for correlation analysis"}
        
        correlation_matrix = numeric_df.corr()
        
        # Convert correlation matrix to native Python types
        converted_matrix = {}
        for col, col_corrs in correlation_matrix.to_dict().items():
            converted_matrix[col] = {row: self._convert_numpy_types(val) for row, val in col_corrs.items()}
        
        # Find strong correlations (>0.7 or <-0.7)
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        "column1": correlation_matrix.columns[i],
                        "column2": correlation_matrix.columns[j],
                        "correlation": self._convert_numpy_types(corr_value)
                    })
        
        return {
            "correlation_matrix": converted_matrix,
            "strong_correlations": strong_correlations
        }
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of numerical columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {"message": "No numerical columns found for distribution analysis"}
        
        distributions = {}
        
        for column in numeric_df.columns:
            col_data = numeric_df[column].dropna()
            
            if len(col_data) == 0:
                continue
            
            # Test for normality
            try:
                _, p_value = stats.normaltest(col_data)
                is_normal = bool(p_value > 0.05)
            except:
                is_normal = False
            
            distributions[column] = {
                "is_normal": is_normal,
                "skewness": self._convert_numpy_types(col_data.skew()),
                "kurtosis": self._convert_numpy_types(col_data.kurtosis()),
                "range": self._convert_numpy_types(col_data.max() - col_data.min()),
                "iqr": self._convert_numpy_types(col_data.quantile(0.75) - col_data.quantile(0.25))
            }
        
        return distributions
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in numerical columns using IQR method."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {"message": "No numerical columns found for outlier detection"}
        
        outliers = {}
        
        for column in numeric_df.columns:
            col_data = numeric_df[column].dropna()
            
            if len(col_data) == 0:
                continue
            
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
            outlier_count = outlier_mask.sum()
            
            outliers[column] = {
                "count": self._convert_numpy_types(outlier_count),
                "percentage": self._convert_numpy_types(outlier_count / len(col_data) * 100),
                "lower_bound": self._convert_numpy_types(lower_bound),
                "upper_bound": self._convert_numpy_types(upper_bound)
            }
        
        return outliers
    
    def generate_chart(self, df: pd.DataFrame, chart_type: str, **kwargs) -> Dict[str, Any]:
        """Generate various types of charts from DataFrame data."""
        try:
            logger.info(f"Generating {chart_type} chart")
            
            chart_generators = {
                "histogram": self._create_histogram,
                "bar": self._create_bar_chart,
                "line": self._create_line_chart,
                "scatter": self._create_scatter_plot,
                "box": self._create_box_plot,
                "heatmap": self._create_heatmap,
                "pie": self._create_pie_chart,
                "distribution": self._create_distribution_plot
            }
            
            if chart_type not in chart_generators:
                raise ValueError(f"Unsupported chart type: {chart_type}")
            
            return chart_generators[chart_type](df, **kwargs)
            
        except Exception as e:
            logger.error(f"Error generating {chart_type} chart: {str(e)}")
            raise
    
    def _create_histogram(self, df: pd.DataFrame, column: str = None, bins: int = 30) -> Dict[str, Any]:
        """Create histogram for numerical data."""
        if column is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numerical columns found for histogram")
            column = numeric_cols[0]
        
        # Create matplotlib version
        fig, ax = plt.subplots(figsize=(10, 6))
        df[column].hist(bins=bins, ax=ax, alpha=0.7, edgecolor='black')
        ax.set_title(f'Histogram of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        
        matplotlib_chart = self._fig_to_base64(fig)
        plt.close(fig)
        
        # Create Plotly version
        plotly_fig = px.histogram(df, x=column, nbins=bins, title=f'Histogram of {column}')
        plotly_chart = json.dumps(plotly_fig, cls=PlotlyJSONEncoder)
        
        return {
            "chart_type": "histogram",
            "matplotlib_chart": matplotlib_chart,
            "plotly_chart": plotly_chart,
            "description": f"Histogram showing the distribution of {column}"
        }
    
    def _create_bar_chart(self, df: pd.DataFrame, x_column: str = None, y_column: str = None) -> Dict[str, Any]:
        """Create bar chart."""
        if x_column is None or y_column is None:
            # Auto-detect suitable columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(categorical_cols) == 0 or len(numeric_cols) == 0:
                raise ValueError("Need both categorical and numerical columns for bar chart")
            
            x_column = categorical_cols[0]
            y_column = numeric_cols[0]
        
        # Aggregate data
        grouped_data = df.groupby(x_column)[y_column].mean().reset_index()
        
        # Create matplotlib version
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(grouped_data[x_column], grouped_data[y_column])
        ax.set_title(f'Bar Chart: {y_column} by {x_column}')
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        matplotlib_chart = self._fig_to_base64(fig)
        plt.close(fig)
        
        # Create Plotly version
        plotly_fig = px.bar(grouped_data, x=x_column, y=y_column, 
                           title=f'Bar Chart: {y_column} by {x_column}')
        plotly_chart = json.dumps(plotly_fig, cls=PlotlyJSONEncoder)
        
        return {
            "chart_type": "bar",
            "matplotlib_chart": matplotlib_chart,
            "plotly_chart": plotly_chart,
            "description": f"Bar chart showing {y_column} grouped by {x_column}"
        }
    
    def _create_line_chart(self, df: pd.DataFrame, x_column: str = None, y_column: str = None) -> Dict[str, Any]:
        """Create line chart."""
        if x_column is None or y_column is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                raise ValueError("Need at least 2 numerical columns for line chart")
            x_column, y_column = numeric_cols[0], numeric_cols[1]
        
        # Sort by x column for proper line chart
        sorted_df = df.sort_values(x_column)
        
        # Create matplotlib version
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(sorted_df[x_column], sorted_df[y_column], marker='o', markersize=4)
        ax.set_title(f'Line Chart: {y_column} vs {x_column}')
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        plt.tight_layout()
        
        matplotlib_chart = self._fig_to_base64(fig)
        plt.close(fig)
        
        # Create Plotly version
        plotly_fig = px.line(sorted_df, x=x_column, y=y_column, 
                            title=f'Line Chart: {y_column} vs {x_column}')
        plotly_chart = json.dumps(plotly_fig, cls=PlotlyJSONEncoder)
        
        return {
            "chart_type": "line",
            "matplotlib_chart": matplotlib_chart,
            "plotly_chart": plotly_chart,
            "description": f"Line chart showing {y_column} vs {x_column}"
        }
    
    def _create_scatter_plot(self, df: pd.DataFrame, x_column: str = None, y_column: str = None) -> Dict[str, Any]:
        """Create scatter plot."""
        if x_column is None or y_column is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                raise ValueError("Need at least 2 numerical columns for scatter plot")
            x_column, y_column = numeric_cols[0], numeric_cols[1]
        
        # Create matplotlib version
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(df[x_column], df[y_column], alpha=0.6)
        ax.set_title(f'Scatter Plot: {y_column} vs {x_column}')
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        
        matplotlib_chart = self._fig_to_base64(fig)
        plt.close(fig)
        
        # Create Plotly version
        plotly_fig = px.scatter(df, x=x_column, y=y_column, 
                               title=f'Scatter Plot: {y_column} vs {x_column}')
        plotly_chart = json.dumps(plotly_fig, cls=PlotlyJSONEncoder)
        
        return {
            "chart_type": "scatter",
            "matplotlib_chart": matplotlib_chart,
            "plotly_chart": plotly_chart,
            "description": f"Scatter plot showing relationship between {x_column} and {y_column}"
        }
    
    def _create_box_plot(self, df: pd.DataFrame, column: str = None) -> Dict[str, Any]:
        """Create box plot for numerical data."""
        if column is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numerical columns found for box plot")
            column = numeric_cols[0]
        
        # Create matplotlib version
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(df[column].dropna())
        ax.set_title(f'Box Plot of {column}')
        ax.set_ylabel(column)
        
        matplotlib_chart = self._fig_to_base64(fig)
        plt.close(fig)
        
        # Create Plotly version
        plotly_fig = px.box(df, y=column, title=f'Box Plot of {column}')
        plotly_chart = json.dumps(plotly_fig, cls=PlotlyJSONEncoder)
        
        return {
            "chart_type": "box",
            "matplotlib_chart": matplotlib_chart,
            "plotly_chart": plotly_chart,
            "description": f"Box plot showing distribution and outliers of {column}"
        }
    
    def _create_heatmap(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create correlation heatmap."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            raise ValueError("Need at least 2 numerical columns for heatmap")
        
        correlation_matrix = numeric_df.corr()
        
        # Create matplotlib version
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Heatmap')
        
        matplotlib_chart = self._fig_to_base64(fig)
        plt.close(fig)
        
        # Create Plotly version
        plotly_fig = px.imshow(correlation_matrix, 
                              text_auto=True, 
                              aspect="auto",
                              title='Correlation Heatmap')
        plotly_chart = json.dumps(plotly_fig, cls=PlotlyJSONEncoder)
        
        return {
            "chart_type": "heatmap",
            "matplotlib_chart": matplotlib_chart,
            "plotly_chart": plotly_chart,
            "description": "Correlation heatmap showing relationships between numerical columns"
        }
    
    def _create_pie_chart(self, df: pd.DataFrame, column: str = None) -> Dict[str, Any]:
        """Create pie chart for categorical data."""
        if column is None:
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) == 0:
                raise ValueError("No categorical columns found for pie chart")
            column = categorical_cols[0]
        
        value_counts = df[column].value_counts().head(10)  # Limit to top 10 categories
        
        # Create matplotlib version
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'Distribution of {column}')
        
        matplotlib_chart = self._fig_to_base64(fig)
        plt.close(fig)
        
        # Create Plotly version
        plotly_fig = px.pie(values=value_counts.values, names=value_counts.index, 
                           title=f'Distribution of {column}')
        plotly_chart = json.dumps(plotly_fig, cls=PlotlyJSONEncoder)
        
        return {
            "chart_type": "pie",
            "matplotlib_chart": matplotlib_chart,
            "plotly_chart": plotly_chart,
            "description": f"Pie chart showing distribution of {column}"
        }
    
    def _create_distribution_plot(self, df: pd.DataFrame, column: str = None) -> Dict[str, Any]:
        """Create distribution plot with histogram and KDE."""
        if column is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numerical columns found for distribution plot")
            column = numeric_cols[0]
        
        # Create matplotlib version with seaborn
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df, x=column, kde=True, ax=ax)
        ax.set_title(f'Distribution of {column}')
        
        matplotlib_chart = self._fig_to_base64(fig)
        plt.close(fig)
        
        # Create Plotly version
        plotly_fig = px.histogram(df, x=column, marginal="box", title=f'Distribution of {column}')
        plotly_chart = json.dumps(plotly_fig, cls=PlotlyJSONEncoder)
        
        return {
            "chart_type": "distribution",
            "matplotlib_chart": matplotlib_chart,
            "plotly_chart": plotly_chart,
            "description": f"Distribution plot showing histogram and density curve of {column}"
        }
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        return image_base64
    
    def suggest_visualizations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Suggest appropriate visualizations based on data characteristics."""
        suggestions = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Histogram for numerical columns
        for col in numeric_cols:
            suggestions.append({
                "chart_type": "histogram",
                "description": f"Histogram of {col} to show distribution",
                "parameters": {"column": col}
            })
        
        # Bar chart for categorical vs numerical
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            suggestions.append({
                "chart_type": "bar",
                "description": f"Bar chart showing {numeric_cols[0]} by {categorical_cols[0]}",
                "parameters": {"x_column": categorical_cols[0], "y_column": numeric_cols[0]}
            })
        
        # Scatter plot for numerical columns
        if len(numeric_cols) >= 2:
            suggestions.append({
                "chart_type": "scatter",
                "description": f"Scatter plot of {numeric_cols[0]} vs {numeric_cols[1]}",
                "parameters": {"x_column": numeric_cols[0], "y_column": numeric_cols[1]}
            })
        
        # Correlation heatmap
        if len(numeric_cols) >= 2:
            suggestions.append({
                "chart_type": "heatmap",
                "description": "Correlation heatmap of numerical columns",
                "parameters": {}
            })
        
        # Pie chart for categorical columns
        for col in categorical_cols:
            if df[col].nunique() <= 10:  # Only suggest for columns with few unique values
                suggestions.append({
                    "chart_type": "pie",
                    "description": f"Pie chart showing distribution of {col}",
                    "parameters": {"column": col}
                })
        
        return suggestions[:5]  # Limit to top 5 suggestions

# Global data analysis service instance
analysis_service = None

def get_analysis_service() -> DataAnalysisService:
    """Get or create data analysis service instance."""
    global analysis_service
    if analysis_service is None:
        analysis_service = DataAnalysisService()
    return analysis_service 