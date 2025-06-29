import os
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Generator
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
from .database_service import get_database_service
from .data_analysis_service import get_analysis_service

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class DatabaseChatService:
    """Service for handling database chat interactions with natural language processing."""
    
    def __init__(self):
        self.openai_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"), 
            base_url="https://api.deepseek.com"
        )
        self.db_service = get_database_service()
        self.analysis_service = get_analysis_service()
        self.conversation_history = []
    
    def _sanitize_for_json(self, obj):
        """Recursively sanitize an object to make it JSON serializable."""
        if isinstance(obj, dict):
            return {key: self._sanitize_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (float, np.float64, np.float32)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        elif pd.isna(obj):
            return None
        elif hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat()
        else:
            return obj
        
    def get_system_prompt(self, schema_info: Dict[str, Any]) -> str:
        """Generate system prompt with database schema context."""
        schema_description = self._format_schema_for_prompt(schema_info)
        
        return f"""You are an intelligent business data analyst who helps users understand their data through natural conversation. Your goal is to provide accurate, practical insights that help users make informed decisions.

Database Schema Information:
{schema_description}

Your Approach:
1. **Understand Intent**: First understand what the user really wants to know from a business perspective
2. **Generate Accurate SQL**: Create precise PostgreSQL queries that answer their actual question
3. **Be Practical**: Focus on actionable insights and real-world implications
4. **Handle Edge Cases**: Consider data quality, missing values, and business context
5. **Provide Context**: Explain what the data means in business terms

SQL Generation Rules:
- Use exact table and column names from the schema
- Include LIMIT 100 for large datasets unless user specifies otherwise
- For aggregations, always include proper GROUP BY clauses
- Use appropriate JOINs for multi-table queries
- Handle NULL values appropriately
- Consider date ranges and filters that make business sense
- For trending questions, use proper time-based ordering
- For comparisons, include relevant groupings and filters

Quality Checks:
- Verify query logic matches user intent
- Ensure results will be meaningful and actionable
- Consider if additional context or filters would be helpful
- Think about potential data quality issues

Response Format:
```sql
[Your SQL query here]
```

Explain what the query does and why it answers the user's question."""

    def _format_schema_for_prompt(self, schema_info: Dict[str, Any]) -> str:
        """Format database schema information for the system prompt."""
        if not schema_info.get("tables"):
            return "No tables found in the database."
        
        schema_text = f"Database contains {schema_info['total_tables']} tables:\n\n"
        
        for table_name, table_info in schema_info["tables"].items():
            schema_text += f"Table: {table_name} (Rows: {table_info.get('row_count', 'Unknown')})\n"
            schema_text += "Columns:\n"
            
            for column in table_info["columns"]:
                nullable = "NULL" if column["nullable"] else "NOT NULL"
                default = f" DEFAULT {column['default']}" if column.get("default") else ""
                schema_text += f"  - {column['name']}: {column['type']} {nullable}{default}\n"
            
            if table_info.get("foreign_keys"):
                schema_text += "Foreign Keys:\n"
                for fk in table_info["foreign_keys"]:
                    schema_text += f"  - {fk}\n"
            
            schema_text += "\n"
        
        return schema_text
    
    def process_natural_language_query(self, user_query: str) -> Dict[str, Any]:
        """Process a natural language query and return SQL, results, and analysis."""
        try:
            logger.info(f"Processing natural language query: {user_query[:100]}...")
            
            # Get fresh schema information
            schema_info = self.db_service.get_database_schema()
            
            # Generate SQL from natural language
            sql_response = self._generate_sql_from_natural_language(user_query, schema_info)
            
            if not sql_response.get("sql_query"):
                return self._sanitize_for_json({
                    "success": False,
                    "error": "Could not generate SQL query from natural language",
                    "response": sql_response.get("explanation", "")
                })
            
            sql_query = sql_response["sql_query"]
            explanation = sql_response.get("explanation", "")
            
            # Execute the SQL query
            execution_result = self.db_service.execute_query(sql_query)
            
            result = {
                "success": execution_result["success"],
                "user_query": user_query,
                "generated_sql": sql_query,
                "explanation": explanation,
                "execution_result": execution_result
            }
            
            # Generate human-friendly response
            try:
                if execution_result["success"] and execution_result.get("data"):
                    df = pd.DataFrame(execution_result["data"])
                    human_response = self._generate_human_response(df, user_query, execution_result)
                    result["human_response"] = human_response
                elif execution_result["success"]:
                    # Query succeeded but no data returned (e.g., INSERT, UPDATE, DELETE)
                    result["human_response"] = f"I successfully completed your request: {user_query}"
                else:
                    # Query failed
                    result["human_response"] = f"I encountered an error while processing your query: {execution_result.get('error', 'Unknown error')}"
            except Exception as e:
                logger.warning(f"Error generating human response: {str(e)}")
                result["human_response"] = f"I processed your query: {user_query}"
            
            # If query was successful and returned data, perform analysis
            if execution_result["success"] and execution_result.get("data"):
                try:
                    df = pd.DataFrame(execution_result["data"])
                    
                    if not df.empty:
                        # Perform data analysis
                        analysis = self.analysis_service.analyze_dataframe(df, "basic")
                        result["data_analysis"] = analysis
                        
                        # Suggest visualizations
                        viz_suggestions = self.analysis_service.suggest_visualizations(df)
                        result["visualization_suggestions"] = viz_suggestions
                        
                        # Generate insights using LLM
                        insights = self._generate_data_insights(df, user_query, sql_query)
                        result["insights"] = insights
                
                except Exception as e:
                    logger.warning(f"Error in data analysis: {str(e)}")
                    result["analysis_error"] = str(e)
            
            # Add to conversation history
            self.conversation_history.append({
                "user_query": user_query,
                "sql_query": sql_query,
                "success": execution_result["success"]
            })
            
            # Sanitize the entire result to ensure JSON serialization
            return self._sanitize_for_json(result)
            
        except Exception as e:
            logger.error(f"Error processing natural language query: {str(e)}")
            return self._sanitize_for_json({
                "success": False,
                "error": str(e),
                "user_query": user_query
            })
    
    def _generate_sql_from_natural_language(self, user_query: str, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """Use OpenAI to convert natural language to SQL with enhanced understanding."""
        try:
            system_prompt = self.get_system_prompt(schema_info)
            
            # Enhanced user prompt with context and intent analysis
            enhanced_user_prompt = f"""
User Question: "{user_query}"

Please analyze this question and provide a SQL query that accurately answers what they're asking. Consider:

1. **Business Intent**: What business question are they trying to answer?
2. **Data Relationships**: Which tables and columns are most relevant?
3. **Practical Scope**: What would be a reasonable and useful result set?
4. **Context Clues**: Are there implied filters, time ranges, or groupings?

Examples of enhanced understanding:
- "top customers" → likely means by revenue/sales, may need ranking
- "recent" → consider last 30-90 days unless specified
- "performance" → likely metrics like totals, averages, trends
- "compare" → may need grouping or side-by-side analysis

Provide the SQL query and a brief explanation of your reasoning.
"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": enhanced_user_prompt}
            ]
            
            # Add conversation history for better context
            if self.conversation_history:
                context_messages = []
                for hist in self.conversation_history[-2:]:  # Last 2 interactions for better context
                    if hist.get("success", False):
                        context_messages.extend([
                            {"role": "user", "content": f"Previous question: {hist['user_query']}"},
                            {"role": "assistant", "content": f"Previous SQL: {hist['sql_query']}"}
                        ])
                
                # Insert context before the current question
                if context_messages:
                    messages.insert(-1, {"role": "system", "content": "Recent conversation context for reference:"})
                    messages = messages[:-1] + context_messages + messages[-1:]
            
            response = self.openai_client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
            
            response_content = response.choices[0].message.content
            
            # Extract SQL query from response
            sql_query = self._extract_sql_from_response(response_content)
            
            # Validate the generated SQL makes sense
            if sql_query:
                validation_result = self._validate_sql_query(sql_query, user_query, schema_info)
                if not validation_result["valid"]:
                    logger.warning(f"SQL validation failed: {validation_result['reason']}")
                    # Try to fix common issues
                    sql_query = self._attempt_sql_fix(sql_query, validation_result["reason"])
            
            return {
                "sql_query": sql_query,
                "explanation": response_content,
                "full_response": response_content
            }
            
        except Exception as e:
            logger.error(f"Error generating SQL from natural language: {str(e)}")
            return {
                "error": str(e),
                "sql_query": None,
                "explanation": f"I encountered an error while processing your question. Please try rephrasing your query or being more specific about what you'd like to know."
            }
    
    def _extract_sql_from_response(self, response_content: str) -> Optional[str]:
        """Extract SQL query from OpenAI response."""
        # Look for SQL in code blocks
        import re
        
        # Try to find SQL in code blocks (```sql or ```)
        sql_pattern = r'```(?:sql)?\s*(.*?)\s*```'
        matches = re.findall(sql_pattern, response_content, re.DOTALL | re.IGNORECASE)
        
        if matches:
            return matches[0].strip()
        
        # If no code blocks, look for lines that start with SELECT, INSERT, UPDATE, DELETE
        lines = response_content.split('\n')
        sql_lines = []
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith(('SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WITH')):
                sql_lines.append(line)
            elif sql_lines and line and not line.startswith('#') and not line.startswith('--'):
                sql_lines.append(line)
            elif sql_lines and line.endswith(';'):
                sql_lines.append(line)
                break
        
        if sql_lines:
            return ' '.join(sql_lines)
        
        return None
    
    def _generate_data_insights(self, df: pd.DataFrame, user_query: str, sql_query: str) -> str:
        """Generate insights about the data using LLM."""
        try:
            # Create a summary of the data
            data_summary = {
                "shape": df.shape,
                "columns": list(df.columns),
                "sample_data": df.head(3).to_dict('records'),
                "basic_stats": {}
            }
            
            # Add basic statistics for numerical columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                data_summary["basic_stats"][col] = {
                    "mean": float(df[col].mean()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "count": int(df[col].count())
                }
            
            insight_prompt = f"""
Analyze the following database query results and provide meaningful insights:

Original Question: {user_query}
SQL Query: {sql_query}
Data Summary: {json.dumps(data_summary, indent=2)}

Please provide:
1. Key findings from the data
2. Notable patterns or trends
3. Potential areas for further analysis
4. Any data quality observations

Keep insights concise and business-focused.
"""
            
            response = self.openai_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a data analyst providing insights from database query results."},
                    {"role": "user", "content": insight_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"Error generating data insights: {str(e)}")
            return "Unable to generate insights at this time."
    
    def _attempt_sql_fix(self, sql_query: str, issue: str) -> str:
        """Attempt to fix common SQL issues."""
        try:
            # Common fixes
            if "limit" in issue.lower() and "LIMIT" not in sql_query.upper():
                if not sql_query.rstrip().endswith(';'):
                    sql_query = sql_query.rstrip() + " LIMIT 100;"
                else:
                    sql_query = sql_query.rstrip(';') + " LIMIT 100;"
            
            # Remove multiple semicolons
            sql_query = sql_query.replace(';;', ';')
            
            return sql_query
        except Exception:
            return sql_query

    def _validate_sql_query(self, sql_query: str, user_query: str, schema_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate if the SQL query makes sense for the user's question."""
        try:
            if not sql_query:
                return {"valid": False, "reason": "No SQL query generated"}
            
            # Check for basic SQL structure
            sql_upper = sql_query.upper().strip()
            
            # Must start with a valid SQL command
            valid_starts = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE']
            if not any(sql_upper.startswith(cmd) for cmd in valid_starts):
                return {"valid": False, "reason": "Invalid SQL command"}
            
            # Check for reasonable LIMIT for SELECT queries
            if sql_upper.startswith('SELECT') and 'LIMIT' not in sql_upper:
                # For data exploration queries, suggest adding LIMIT
                exploration_keywords = ['show', 'list', 'get', 'find', 'all']
                if any(keyword in user_query.lower() for keyword in exploration_keywords):
                    return {"valid": False, "reason": "Should include LIMIT for data exploration"}
            
            # Check if referenced tables exist in schema
            available_tables = set(schema_info.get("tables", {}).keys())
            
            # Simple table name extraction (this could be more sophisticated)
            import re
            table_pattern = r'\bFROM\s+(\w+)|\bJOIN\s+(\w+)'
            found_tables = re.findall(table_pattern, sql_query, re.IGNORECASE)
            
            for match in found_tables:
                table_name = match[0] or match[1]  # Either FROM or JOIN match
                if table_name and table_name not in available_tables:
                    return {"valid": False, "reason": f"Table '{table_name}' not found in schema"}
            
            return {"valid": True, "reason": "Query looks valid"}
            
        except Exception as e:
            logger.warning(f"Error validating SQL: {str(e)}")
            return {"valid": True, "reason": "Validation error, proceeding anyway"}

    def _generate_human_response(self, df: pd.DataFrame, user_query: str, execution_result: Dict[str, Any]) -> str:
        """Generate a practical, business-focused human response."""
        try:
            if not execution_result.get("success"):
                error_msg = execution_result.get('error', 'Unknown error')
                # Provide helpful error suggestions
                suggestions = []
                if "column" in error_msg.lower():
                    suggestions.append("• Check if the column name is spelled correctly")
                    suggestions.append("• Try using different column names from your data")
                if "table" in error_msg.lower():
                    suggestions.append("• Verify the table name exists in your database")
                    suggestions.append("• Try rephrasing your question to be more specific")
                if "syntax" in error_msg.lower():
                    suggestions.append("• Try simplifying your question")
                    suggestions.append("• Ask for specific data rather than complex operations")
                
                suggestion_text = "\n".join(suggestions) if suggestions else ""
                return f"""I encountered an issue while processing your query: **{error_msg}**

{suggestion_text}

Please try rephrasing your question or let me know if you'd like help understanding what data is available."""
            
            if df.empty:
                # Provide actionable suggestions for empty results
                suggestions = [
                    "• Try broadening your search criteria",
                    "• Check if the data exists for the specified time period", 
                    "• Verify the filters or conditions in your question",
                    "• Ask me what data is available in the database"
                ]
                return f"""## No Results Found

I didn't find any data matching your query: **"{user_query}"**

**Suggestions to try:**
{chr(10).join(suggestions)}

Would you like me to help you explore what data is available?"""
            
            # Analyze the data for practical insights
            row_count = len(df)
            columns = list(df.columns)
            
            # Build comprehensive data analysis
            analysis_parts = []
            
            # 1. Direct answer to their question
            analysis_parts.append(f"## Answer to Your Question\n")
            
            # Determine query type and provide contextual response
            query_lower = user_query.lower()
            
            if any(word in query_lower for word in ['top', 'best', 'highest', 'most']):
                # Ranking/Top queries
                if row_count > 0:
                    top_item = df.iloc[0]
                    first_col = columns[0] if columns else "item"
                    second_col = columns[1] if len(columns) > 1 else "value"
                    analysis_parts.append(f"The top result is **{top_item[first_col]}** with {second_col} of **{top_item[second_col]}**.")
                    if row_count > 1:
                        analysis_parts.append(f"I found {row_count} total results for your query.")
            
            elif any(word in query_lower for word in ['total', 'sum', 'count', 'how many']):
                # Aggregation queries
                if len(columns) == 1:
                    total_value = df.iloc[0, 0]
                    analysis_parts.append(f"The total is **{total_value:,}** based on your criteria.")
                else:
                    analysis_parts.append(f"I found **{row_count}** records matching your criteria.")
            
            elif any(word in query_lower for word in ['compare', 'comparison', 'vs', 'versus']):
                # Comparison queries
                analysis_parts.append(f"Here's the comparison you requested with **{row_count}** items:")
                if row_count <= 5:  # Show details for small comparisons
                    for idx, row in df.iterrows():
                        if len(columns) >= 2:
                            analysis_parts.append(f"• **{row[columns[0]]}**: {row[columns[1]]}")
            
            else:
                # General data queries
                analysis_parts.append(f"I found **{row_count}** records that match your query.")
            
            # 2. Key insights from the data
            if row_count > 0:
                analysis_parts.append(f"\n## Key Insights")
                
                # Statistical insights for numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                for col in numeric_cols[:3]:  # Top 3 numeric columns
                    if not df[col].isna().all():
                        avg_val = df[col].mean()
                        min_val = df[col].min()
                        max_val = df[col].max()
                        analysis_parts.append(f"• **{col}**: Average of {avg_val:,.2f}, ranging from {min_val:,.2f} to {max_val:,.2f}")
                
                # Categorical insights
                categorical_cols = df.select_dtypes(include=['object', 'string']).columns
                for col in categorical_cols[:2]:  # Top 2 categorical columns
                    if col in df.columns and not df[col].isna().all():
                        unique_count = df[col].nunique()
                        if unique_count < row_count:  # Not all unique
                            most_common = df[col].value_counts().iloc[0]
                            most_common_name = df[col].value_counts().index[0]
                            analysis_parts.append(f"• **{col}**: {unique_count} unique values, most common is '{most_common_name}' ({most_common} times)")
            
            # 3. Business recommendations
            analysis_parts.append(f"\n## What This Means")
            
            if any(word in query_lower for word in ['sales', 'revenue', 'income']):
                analysis_parts.append("💰 This sales data can help you identify top performers and revenue opportunities.")
            elif any(word in query_lower for word in ['customer', 'client', 'user']):
                analysis_parts.append("👥 Understanding your customer patterns can help improve service and retention.")
            elif any(word in query_lower for word in ['product', 'item', 'inventory']):
                analysis_parts.append("📦 Product insights can guide inventory and marketing decisions.")
            elif any(word in query_lower for word in ['time', 'date', 'month', 'year']):
                analysis_parts.append("📅 Time-based trends help you understand seasonal patterns and growth.")
            else:
                analysis_parts.append("📊 This data provides valuable insights for making informed business decisions.")
            
            # 4. Next steps suggestions
            if row_count > 10:
                analysis_parts.append(f"\n**💡 Suggestions:**")
                analysis_parts.append("• Try asking for specific time periods or filtered results")
                analysis_parts.append("• Ask me to break this down by categories or regions")
                analysis_parts.append("• Request trends or comparisons to identify patterns")
            
            return "\n".join(analysis_parts)
            
        except Exception as e:
            logger.warning(f"Error generating practical human response: {str(e)}")
            # Fallback to basic response
            if not execution_result.get("success"):
                return f"I encountered an error: {execution_result.get('error', 'Unknown error')}. Please try rephrasing your question."
            elif df.empty:
                return f"I didn't find any results for your query about {user_query}. Try adjusting your search criteria."
            else:
                row_count = len(df)
                return f"I found {row_count} result{'s' if row_count != 1 else ''} for your query: {user_query}. The data is shown in the table below."
    
    def stream_database_chat(self, user_query: str) -> Generator[Dict[str, Any], None, None]:
        """Stream database chat response for real-time interaction."""
        try:
            # First yield the query processing status
            yield {"type": "status", "message": "Analyzing your query..."}
            
            # Get schema information
            schema_info = self.db_service.get_database_schema()
            yield {"type": "status", "message": "Generating SQL query..."}
            
            # Generate SQL
            sql_response = self._generate_sql_from_natural_language(user_query, schema_info)
            
            if not sql_response.get("sql_query"):
                yield {
                    "type": "error",
                    "message": "Could not generate SQL query",
                    "details": sql_response.get("explanation", "")
                }
                return
            
            yield {
                "type": "sql_generated",
                "sql_query": sql_response["sql_query"],
                "explanation": sql_response.get("explanation", "")
            }
            
            # Execute query
            yield {"type": "status", "message": "Executing query..."}
            
            execution_result = self.db_service.execute_query(sql_response["sql_query"])
            
            yield {
                "type": "query_result",
                "result": execution_result
            }
            
            # If successful, perform analysis
            if execution_result["success"] and execution_result.get("data"):
                yield {"type": "status", "message": "Analyzing results..."}
                
                df = pd.DataFrame(execution_result["data"])
                
                if not df.empty:
                    # Analysis
                    analysis = self.analysis_service.analyze_dataframe(df, "basic")
                    yield {"type": "analysis", "analysis": analysis}
                    
                    # Visualization suggestions
                    viz_suggestions = self.analysis_service.suggest_visualizations(df)
                    yield {"type": "visualization_suggestions", "suggestions": viz_suggestions}
                    
                    # Insights
                    insights = self._generate_data_insights(df, user_query, sql_response["sql_query"])
                    yield {"type": "insights", "insights": insights}
            
            yield {"type": "complete", "message": "Analysis complete"}
            
        except Exception as e:
            logger.error(f"Error in streaming database chat: {str(e)}")
            yield {"type": "error", "message": str(e)}
    
    def generate_chart_from_query_result(self, data: List[Dict], chart_type: str, **kwargs) -> Dict[str, Any]:
        """Generate a chart from query result data."""
        try:
            df = pd.DataFrame(data)
            if df.empty:
                raise ValueError("No data available for chart generation")
            
            return self.analysis_service.generate_chart(df, chart_type, **kwargs)
            
        except Exception as e:
            logger.error(f"Error generating chart: {str(e)}")
            raise
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")

# Global database chat service instance
db_chat_service = None

def get_database_chat_service() -> DatabaseChatService:
    """Get or create database chat service instance."""
    global db_chat_service
    if db_chat_service is None:
        db_chat_service = DatabaseChatService()
    return db_chat_service 