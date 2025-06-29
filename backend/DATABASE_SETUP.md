# Database Chat Setup Guide

This guide will help you set up the database chat functionality for the Agentic RAG application.

## Prerequisites

1. PostgreSQL database (local or cloud)
2. DeepSeek API key
3. Python dependencies (automatically installed from requirements.txt)

## Environment Variables

Create a `.env` file in the `backend` directory with the following variables:

```env
# DeepSeek API Configuration (required for database chat)
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# OpenAI API Configuration (required for RAG functionality)
OPENAI_API_KEY=your_openai_api_key_here

# PostgreSQL Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=your_database_name
DB_USER=your_username
DB_PASSWORD=your_password
```

### Example Configurations

#### Local PostgreSQL
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=mydatabase
DB_USER=myuser
DB_PASSWORD=mypassword
```

#### Cloud PostgreSQL (AWS RDS, Google Cloud SQL, etc.)
```env
DB_HOST=your-database-host.region.rds.amazonaws.com
DB_PORT=5432
DB_NAME=production_db
DB_USER=dbuser
DB_PASSWORD=securepassword
```

#### Local Docker PostgreSQL
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=postgres
DB_USER=postgres
DB_PASSWORD=postgres
```

## Quick Setup with Docker

If you don't have PostgreSQL installed, you can quickly set up a local instance using Docker:

```bash
# Run PostgreSQL in Docker
docker run --name postgres-db \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=testdb \
  -p 5432:5432 \
  -d postgres:13

# Connect and create sample data (optional)
docker exec -it postgres-db psql -U postgres -d testdb
```

## Installation

1. Install the required Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Set up your `.env` file with the database credentials

3. Start the application:
```bash
cd backend
uvicorn app.main:app --reload
```

## Features

The database chat functionality provides:

### 1. Natural Language Queries
- Convert natural language to SQL
- Execute queries safely
- Get intelligent explanations

Example: "Show me the top 10 customers by revenue"

### 2. Data Analysis
- Automatic statistical analysis
- Data quality assessment
- Distribution analysis
- Outlier detection

### 3. Visualization
- Automatic chart suggestions
- Multiple chart types (bar, line, scatter, histogram, etc.)
- Interactive Plotly charts
- Static matplotlib charts

### 4. Database Exploration
- Schema inspection
- Table analysis
- Sample data viewing
- Relationship discovery

## API Endpoints

### Database Status
- `GET /database/status` - Check database connection
- `GET /database/schema` - Get database schema

### Querying
- `POST /database/query` - Natural language query
- `POST /database/query/stream` - Streaming query response
- `POST /database/sql` - Raw SQL execution

### Analysis
- `GET /database/tables/{table_name}/analyze` - Analyze specific table
- `POST /database/chart` - Generate charts from data

### Conversation
- `GET /database/conversation/history` - Get chat history
- `DELETE /database/conversation/history` - Clear chat history

## Security Considerations

1. **SQL Injection Prevention**: All queries are parameterized and validated
2. **Access Control**: Consider implementing user authentication
3. **Query Limits**: Automatic LIMIT clauses prevent large data dumps
4. **Audit Logging**: All queries are logged for security monitoring

## Troubleshooting

### Database Connection Issues
1. Check your `.env` file configuration
2. Ensure PostgreSQL is running
3. Verify network connectivity
4. Check firewall settings

### Permission Issues
1. Ensure the database user has appropriate permissions
2. Check schema access rights
3. Verify table permissions

### Performance Issues
1. Add appropriate indexes to your tables
2. Use LIMIT clauses for large datasets
3. Consider query optimization

## Sample Data

To test the functionality, you can create sample tables:

```sql
-- Create a sample customers table
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    signup_date DATE,
    total_orders INTEGER,
    total_spent DECIMAL(10,2)
);

-- Insert sample data
INSERT INTO customers (name, email, signup_date, total_orders, total_spent) VALUES
('John Doe', 'john@example.com', '2023-01-15', 5, 250.00),
('Jane Smith', 'jane@example.com', '2023-02-20', 12, 890.50),
('Bob Johnson', 'bob@example.com', '2023-03-10', 3, 150.25);

-- Create a sample orders table
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id),
    order_date DATE,
    amount DECIMAL(10,2),
    status VARCHAR(20)
);

-- Insert sample orders
INSERT INTO orders (customer_id, order_date, amount, status) VALUES
(1, '2023-01-20', 50.00, 'completed'),
(1, '2023-02-15', 75.00, 'completed'),
(2, '2023-02-25', 120.00, 'completed'),
(2, '2023-03-05', 95.50, 'pending');
```

## Support

If you encounter issues:

1. Check the application logs for error messages
2. Verify your database connection and credentials
3. Ensure all dependencies are installed
4. Review the API documentation at `http://localhost:8000/docs` 