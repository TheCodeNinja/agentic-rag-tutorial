import os
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sqlalchemy import create_engine, text, inspect, MetaData, Table
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseService:
    """Service for handling PostgreSQL database operations with comprehensive logging and error handling."""
    
    def __init__(self):
        self.engine = None
        self.session_maker = None
        self._connection_params = self._get_connection_params()
        self._connect()
    
    def _get_connection_params(self) -> Dict[str, str]:
        """Get database connection parameters from environment variables."""
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'postgres'),
            'username': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', ''),
        }
    
    def _connect(self):
        """Establish database connection with proper error handling."""
        try:
            params = self._connection_params
            connection_string = f"postgresql://{params['username']}:{params['password']}@{params['host']}:{params['port']}/{params['database']}"
            
            self.engine = create_engine(
                connection_string,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.session_maker = sessionmaker(bind=self.engine)
            logger.info(f"Successfully connected to PostgreSQL database: {params['host']}:{params['port']}/{params['database']}")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise
    
    def test_connection(self) -> Dict[str, Any]:
        """Test database connection and return status."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()[0]
                
            return {
                "status": "connected",
                "database": self._connection_params['database'],
                "host": self._connection_params['host'],
                "port": self._connection_params['port'],
                "version": version
            }
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_database_schema(self) -> Dict[str, Any]:
        """Get comprehensive database schema information."""
        try:
            inspector = inspect(self.engine)
            schema_info = {
                "tables": {},
                "total_tables": 0
            }
            
            table_names = inspector.get_table_names()
            schema_info["total_tables"] = len(table_names)
            
            for table_name in table_names:
                columns = inspector.get_columns(table_name)
                foreign_keys = inspector.get_foreign_keys(table_name)
                indexes = inspector.get_indexes(table_name)
                
                # Get row count
                try:
                    with self.engine.connect() as conn:
                        count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                        row_count = count_result.fetchone()[0]
                except:
                    row_count = "N/A"
                
                schema_info["tables"][table_name] = {
                    "columns": [
                        {
                            "name": col["name"],
                            "type": str(col["type"]),
                            "nullable": col["nullable"],
                            "default": col.get("default")
                        }
                        for col in columns
                    ],
                    "foreign_keys": foreign_keys,
                    "indexes": indexes,
                    "row_count": row_count
                }
            
            logger.info(f"Retrieved schema for {len(table_names)} tables")
            return schema_info
            
        except Exception as e:
            logger.error(f"Failed to get database schema: {str(e)}")
            raise
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a SQL query and return results with comprehensive error handling."""
        try:
            logger.info(f"Executing query: {query[:100]}{'...' if len(query) > 100 else ''}")
            
            with self.engine.begin() as conn:  # Use begin() for auto-commit transaction
                result = conn.execute(text(query), params or {})
                
                if result.returns_rows:
                    # Fetch data and convert to DataFrame
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    
                    # Convert DataFrame to JSON-serializable format
                    import json
                    import numpy as np
                    
                    # Convert numpy types to native Python types for JSON serialization
                    def convert_numpy_types(obj):
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
                    
                    # Apply conversion to all data
                    data = []
                    for record in df.to_dict('records'):
                        converted_record = {key: convert_numpy_types(value) for key, value in record.items()}
                        data.append(converted_record)
                    
                    # Get basic statistics
                    stats = {
                        "row_count": int(len(df)),
                        "column_count": int(len(df.columns)),
                        "columns": list(df.columns),
                        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
                    }
                    
                    # Add numerical statistics for numeric columns
                    numeric_columns = df.select_dtypes(include=['number']).columns
                    if len(numeric_columns) > 0:
                        numeric_summary = df[numeric_columns].describe().to_dict()
                        # Convert numpy types in the summary as well
                        converted_summary = {}
                        for col, col_stats in numeric_summary.items():
                            converted_summary[col] = {stat: convert_numpy_types(value) for stat, value in col_stats.items()}
                        stats["numeric_summary"] = converted_summary
                    
                    logger.info(f"Query executed successfully. Returned {len(data)} rows, {len(df.columns)} columns")
                    
                    return {
                        "success": True,
                        "data": data,
                        "statistics": stats,
                        "query": query
                    }
                else:
                    # For non-SELECT queries
                    affected_rows = result.rowcount
                    logger.info(f"Query executed successfully. Affected {affected_rows} rows")
                    
                    return {
                        "success": True,
                        "affected_rows": affected_rows,
                        "message": f"Query executed successfully. {affected_rows} rows affected.",
                        "query": query
                    }
        
        except SQLAlchemyError as e:
            error_msg = f"SQL Error: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "error_type": "SQL_ERROR",
                "query": query
            }
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "error_type": "GENERAL_ERROR",
                "query": query
            }
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> Dict[str, Any]:
        """Get sample data from a specific table."""
        try:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
            return self.execute_query(query)
        except Exception as e:
            logger.error(f"Failed to get sample data from {table_name}: {str(e)}")
            raise
    
    def analyze_table(self, table_name: str) -> Dict[str, Any]:
        """Perform comprehensive analysis of a table."""
        try:
            analysis = {}
            
            # Basic table info
            with self.engine.connect() as conn:
                # Row count
                count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                row_count = count_result.fetchone()[0]
                
                # Sample data
                sample_result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 10"))
                sample_df = pd.DataFrame(sample_result.fetchall(), columns=sample_result.keys())
                
            analysis = {
                "table_name": table_name,
                "row_count": row_count,
                "column_count": len(sample_df.columns),
                "columns": list(sample_df.columns),
                "sample_data": sample_df.to_dict('records'),
                "column_analysis": {}
            }
            
            # Analyze each column
            for column in sample_df.columns:
                col_analysis = {
                    "data_type": str(sample_df[column].dtype),
                    "null_count": int(sample_df[column].isnull().sum()),
                    "unique_count": int(sample_df[column].nunique())
                }
                
                if sample_df[column].dtype in ['int64', 'float64']:
                    col_analysis.update({
                        "min": float(sample_df[column].min()) if not sample_df[column].empty else None,
                        "max": float(sample_df[column].max()) if not sample_df[column].empty else None,
                        "mean": float(sample_df[column].mean()) if not sample_df[column].empty else None,
                        "std": float(sample_df[column].std()) if not sample_df[column].empty else None
                    })
                
                analysis["column_analysis"][column] = col_analysis
            
            logger.info(f"Analysis completed for table: {table_name}")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze table {table_name}: {str(e)}")
            raise
    
    def generate_sql_from_natural_language(self, natural_query: str, schema_info: Dict[str, Any]) -> str:
        """Generate SQL query from natural language using the schema context."""
        # This is a placeholder for more sophisticated NL to SQL conversion
        # In a production system, you'd use a specialized model or service
        
        # Simple keyword-based approach for demonstration
        query_lower = natural_query.lower()
        
        # Extract table names mentioned in the query
        mentioned_tables = []
        for table_name in schema_info["tables"].keys():
            if table_name.lower() in query_lower:
                mentioned_tables.append(table_name)
        
        if not mentioned_tables:
            # If no specific table mentioned, suggest exploring available tables
            table_list = ", ".join(schema_info["tables"].keys())
            return f"-- No specific table found in query. Available tables: {table_list}\n-- Please specify which table you'd like to query"
        
        # Basic query generation based on keywords
        if any(word in query_lower for word in ['count', 'how many']):
            return f"SELECT COUNT(*) FROM {mentioned_tables[0]};"
        elif any(word in query_lower for word in ['all', 'everything', 'show me']):
            return f"SELECT * FROM {mentioned_tables[0]} LIMIT 100;"
        elif any(word in query_lower for word in ['average', 'avg', 'mean']):
            numeric_cols = []
            for col in schema_info["tables"][mentioned_tables[0]]["columns"]:
                if 'int' in col["type"].lower() or 'float' in col["type"].lower() or 'numeric' in col["type"].lower():
                    numeric_cols.append(col["name"])
            if numeric_cols:
                return f"SELECT AVG({numeric_cols[0]}) FROM {mentioned_tables[0]};"
        
        # Default to showing structure
        return f"SELECT * FROM {mentioned_tables[0]} LIMIT 10;"

    def close(self):
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")

# Global database service instance
db_service = None

def get_database_service() -> DatabaseService:
    """Get or create database service instance."""
    global db_service
    if db_service is None:
        db_service = DatabaseService()
    return db_service 