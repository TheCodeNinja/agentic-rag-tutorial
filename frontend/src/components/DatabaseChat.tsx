import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
    Box, TextField, Button, Paper, Typography, 
    CircularProgress, Alert, Chip, Card, CardContent,
    Table, TableBody, TableCell, TableContainer, TableHead, TableRow,
    Accordion, AccordionSummary, AccordionDetails, IconButton,
    Divider, Grow, Fab, Collapse, Drawer
} from '@mui/material';
import {
    Send as SendIcon,
    ExpandMore as ExpandMoreIcon,
    Storage as DatabaseIcon,
    Assessment as AnalyticsIcon,
    BarChart as ChartIcon,
    TableChart as TableIcon,
    Refresh as RefreshIcon,
    Info as InfoIcon,
    Visibility as VisibilityIcon,
    VisibilityOff as VisibilityOffIcon,
    Code as CodeIcon,
    Clear as ClearIcon
} from '@mui/icons-material';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';

// Types
interface DatabaseStatus {
    status: string;
    database?: string;
    host?: string;
    port?: string;
    version?: string;
    error?: string;
}

interface TableInfo {
    columns: Array<{
        name: string;
        type: string;
        nullable: boolean;
        default?: string;
    }>;
    row_count: number;
}

interface QueryResult {
    success: boolean;
    user_query: string;
    generated_sql: string;
    explanation: string;
    human_response?: string;
    execution_result: {
        success: boolean;
        data?: Array<Record<string, any>>;
        statistics?: {
            row_count: number;
            column_count: number;
            columns: string[];
        };
        error?: string;
    };
    data_analysis?: any;
    visualization_suggestions?: Array<{
        chart_type: string;
        description: string;
        parameters: Record<string, any>;
    }>;
    insights?: string;
}

const DatabaseChat: React.FC = () => {
    const [query, setQuery] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [results, setResults] = useState<QueryResult[]>([]);
    const [dbStatus, setDbStatus] = useState<DatabaseStatus | null>(null);
    const [schema, setSchema] = useState<Record<string, TableInfo>>({});
    const [showSqlMap, setShowSqlMap] = useState<Record<number, boolean>>({});
    const [showAnalysisMap, setShowAnalysisMap] = useState<Record<number, boolean>>({});
    const [rightDrawerOpen, setRightDrawerOpen] = useState(false);
    const [activeRightTab, setActiveRightTab] = useState('chat');
    
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const API_BASE_URL = 'http://localhost:8000';

    // Initialize database status and schema
    useEffect(() => {
        checkDatabaseStatus();
        loadDatabaseSchema();
    }, []);

    useEffect(() => {
            scrollToBottom();
    }, [results]);

    const scrollToBottom = useCallback(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, []);

    const toggleSqlVisibility = useCallback((index: number) => {
        setShowSqlMap(prev => ({
            ...prev,
            [index]: !prev[index]
        }));
    }, []);

    const toggleAnalysisVisibility = useCallback((index: number) => {
        setShowAnalysisMap(prev => ({
            ...prev,
            [index]: !prev[index]
        }));
    }, []);

    const clearChat = useCallback(() => {
        setResults([]);
        setShowSqlMap({});
        setShowAnalysisMap({});
    }, []);

    const handleQueryChange = useCallback((value: string) => {
        setQuery(value);
    }, []);

    const checkDatabaseStatus = async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/database/status`);
            setDbStatus(response.data);
        } catch (error) {
            console.error('Error checking database status:', error);
            setDbStatus({ status: 'error', error: 'Failed to connect' });
        }
    };

    const loadDatabaseSchema = async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/database/schema`);
            setSchema(response.data.tables || {});
        } catch (error) {
            console.error('Error loading database schema:', error);
        }
    };

    const handleSubmitQuery = useCallback(async () => {
        if (!query.trim() || isLoading) return;

        setIsLoading(true);
        try {
            const response = await axios.post(`${API_BASE_URL}/database/query`, {
                query: query.trim()
            });

            setResults(prev => [...prev, response.data]);
            setQuery('');
        } catch (error: any) {
            console.error('Error executing query:', error);
            const errorResult: QueryResult = {
                success: false,
                user_query: query.trim(),
                generated_sql: '',
                explanation: '',
                human_response: `Error: ${error.response?.data?.detail || error.message}`,
                execution_result: {
                    success: false,
                    error: error.response?.data?.detail || error.message
                }
            };
            setResults(prev => [...prev, errorResult]);
            setQuery('');
        } finally {
            setIsLoading(false);
        }
    }, [query, isLoading]);

    const renderQueryResult = useCallback((result: QueryResult, index: number) => {
        const shouldShowSql = showSqlMap[index];
        const shouldShowAnalysis = showAnalysisMap[index];
        
        return (
            <Box key={index} sx={{ mb: 3, width: '100%', maxWidth: '100%', boxSizing: 'border-box' }}>
                {/* User Query Message */}
                <Grow in={true} timeout={500}>
                    <Paper elevation={0} sx={{
                        p: 2, mb: 2, maxWidth: '85%',
                        width: 'fit-content',
                        alignSelf: 'flex-end',
                        ml: 'auto',
                        bgcolor: 'rgba(25, 118, 210, 0.15)',
                        borderRadius: '20px 20px 5px 20px',
                        border: '1px solid rgba(25, 118, 210, 0.3)',
                        boxSizing: 'border-box',
                        overflowWrap: 'break-word',
                        wordBreak: 'break-word'
                    }}>
                        <Box sx={{ display: 'flex', alignItems: 'flex-start', maxWidth: '100%' }}>
                            <Box sx={{
                                width: 32, height: 32, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
                                bgcolor: 'primary.main', color: 'white', mr: 1.5, fontSize: '0.8rem', fontWeight: 'bold', 
                                boxShadow: '0 0 10px rgba(25, 118, 210, 0.4)', flexShrink: 0
                            }}>
                                U
                            </Box>
                            <Typography variant="body1" sx={{ 
                                flex: 1, 
                                fontWeight: 500,
                                overflowWrap: 'break-word',
                                wordBreak: 'break-word',
                                minWidth: 0
                            }}>
                                {result.user_query}
                            </Typography>
                        </Box>
                    </Paper>
                </Grow>

                {/* Bot Response Message */}
                <Grow in={true} timeout={700}>
                    <Paper elevation={0} sx={{
                        p: 2, mb: 2, maxWidth: '100%',
                        width: '100%',
                        alignSelf: 'flex-start',
                        bgcolor: 'rgba(255, 255, 255, 0.08)',
                        borderRadius: '20px 20px 20px 5px',
                        border: '1px solid rgba(255, 255, 255, 0.15)',
                        boxSizing: 'border-box',
                        overflow: 'hidden',
                        minWidth: 0,
                        position: 'relative'
                    }}>
                        <Box sx={{ display: 'flex', alignItems: 'flex-start', maxWidth: '100%' }}>
                            <Box sx={{
                                width: 32, height: 32, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
                                bgcolor: 'rgba(0, 255, 255, 0.2)', color: 'cyan', mr: 1.5, fontSize: '0.8rem', fontWeight: 'bold', 
                                boxShadow: '0 0 10px rgba(0, 255, 255, 0.3)', flexShrink: 0, border: '1px solid rgba(0, 255, 255, 0.3)'
                            }}>
                                <DatabaseIcon fontSize="small" />
                            </Box>
                            <Box sx={{ flex: 1, overflow: 'hidden', minWidth: 0, maxWidth: '100%' }}>
                                {/* Toggle buttons */}
                                <Box sx={{ 
                                    display: 'flex', 
                                    gap: 1, 
                                    mb: 2, 
                                    flexWrap: 'wrap',
                                    width: '100%',
                                    maxWidth: '100%',
                                    boxSizing: 'border-box'
                                }}>
                                    {result.generated_sql && (
                                        <Chip
                                            icon={shouldShowSql ? <VisibilityOffIcon /> : <CodeIcon />}
                                            label="SQL"
                                            onClick={() => toggleSqlVisibility(index)}
                                            variant={shouldShowSql ? "filled" : "outlined"}
                                            size="small"
                                            sx={{ 
                                                bgcolor: shouldShowSql ? 'primary.main' : 'transparent',
                                                maxWidth: '100px',
                                                overflow: 'hidden',
                                                textOverflow: 'ellipsis',
                                                '&:hover': { bgcolor: shouldShowSql ? 'primary.dark' : 'rgba(25, 118, 210, 0.1)' }
                                            }}
                                        />
                                    )}
                                    {(result.data_analysis || result.insights) && (
                                        <Chip
                                            icon={shouldShowAnalysis ? <VisibilityOffIcon /> : <AnalyticsIcon />}
                                            label="Analysis"
                                            onClick={() => toggleAnalysisVisibility(index)}
                                            variant={shouldShowAnalysis ? "filled" : "outlined"}
                                            size="small"
                                            sx={{ 
                                                bgcolor: shouldShowAnalysis ? 'secondary.main' : 'transparent',
                                                maxWidth: '120px',
                                                overflow: 'hidden',
                                                textOverflow: 'ellipsis',
                                                '&:hover': { bgcolor: shouldShowAnalysis ? 'secondary.dark' : 'rgba(156, 39, 176, 0.1)' }
                                            }}
                                        />
                                    )}
                                </Box>

                                {result.execution_result?.success ? (
                                    <Box sx={{ width: '100%', maxWidth: '100%', overflow: 'hidden', minWidth: 0 }}>
                                        {/* Human-friendly response */}
                                        {result.human_response && (
                                            <Box sx={{ mb: 3, width: '100%', maxWidth: '100%', overflow: 'hidden', minWidth: 0 }}>
                                                <Box sx={{ 
                                                    p: 3, 
                                                    bgcolor: 'rgba(0, 255, 255, 0.05)', 
                                                    border: '1px solid rgba(0, 255, 255, 0.2)',
                                                    borderRadius: 2,
                                                    width: '100%',
                                                    maxWidth: '100%',
                                                    boxSizing: 'border-box',
                                                    overflow: 'hidden',
                                                    overflowWrap: 'break-word',
                                                    wordBreak: 'break-word',
                                                    '& h1, & h2, & h3': { 
                                                        color: 'primary.main',
                                                        fontWeight: 'bold',
                                                        mb: 1,
                                                        overflowWrap: 'break-word',
                                                        wordBreak: 'break-word'
                                                    },
                                                    '& p': { 
                                                        mb: 1.5,
                                                        lineHeight: 1.6,
                                                        overflowWrap: 'break-word',
                                                        wordBreak: 'break-word'
                                                    },
                                                    '& ul, & ol': { 
                                                        pl: 3,
                                                        mb: 1.5
                                                    },
                                                    '& li': { 
                                                        mb: 0.5,
                                                        lineHeight: 1.5,
                                                        overflowWrap: 'break-word',
                                                        wordBreak: 'break-word'
                                                    },
                                                    '& strong': {
                                                        color: 'primary.light',
                                                        fontWeight: 'bold',
                                                        overflowWrap: 'break-word',
                                                        wordBreak: 'break-word'
                                                    },
                                                    '& code': {
                                                        bgcolor: 'rgba(255, 255, 255, 0.1)',
                                                        padding: '2px 6px',
                                                        borderRadius: 1,
                                                        fontSize: '0.875rem',
                                                        overflowWrap: 'break-word',
                                                        wordBreak: 'break-all',
                                                        whiteSpace: 'pre-wrap'
                                                    }
                                                }}>
                                                    <ReactMarkdown>{result.human_response}</ReactMarkdown>
                                                </Box>
                                            </Box>
                                        )}

                                        {/* Data table */}
                                        {result.execution_result.data && (
                                            <Collapse in={true}>
                                                <Box sx={{ mb: 2, width: '100%', maxWidth: '100%', overflow: 'hidden', minWidth: 0 }}>
                                                    <Typography variant="h6" gutterBottom sx={{ 
                                                        display: 'flex', 
                                                        alignItems: 'center',
                                                        color: 'primary.main',
                                                        fontWeight: 'bold',
                                                        overflowWrap: 'break-word',
                                                        wordBreak: 'break-word'
                                                    }}>
                                                        <TableIcon sx={{ mr: 1, flexShrink: 0 }} />
                                                        Data Results ({result.execution_result.statistics?.row_count || 0} rows)
                                                    </Typography>
                                                    <TableContainer 
                                                        component={Paper} 
                                                        variant="outlined" 
                                                        sx={{ 
                                                            maxHeight: 400,
                                                            width: '100%',
                                                            maxWidth: '100%',
                                                            overflowX: 'auto',
                                                            overflowY: 'auto',
                                                            border: '1px solid',
                                                            borderColor: 'rgba(255, 255, 255, 0.2)',
                                                            borderRadius: 2,
                                                            boxSizing: 'border-box',
                                                            bgcolor: 'rgba(255, 255, 255, 0.02)',
                                                            display: 'block',
                                                            position: 'relative',
                                                            '&::-webkit-scrollbar': {
                                                                width: '8px',
                                                                height: '8px'
                                                            },
                                                            '&::-webkit-scrollbar-track': {
                                                                bgcolor: 'rgba(255, 255, 255, 0.1)',
                                                                borderRadius: '4px'
                                                            },
                                                            '&::-webkit-scrollbar-thumb': {
                                                                bgcolor: 'rgba(255, 255, 255, 0.3)',
                                                                borderRadius: '4px',
                                                                '&:hover': {
                                                                    bgcolor: 'rgba(255, 255, 255, 0.5)'
                                                                }
                                                            }
                                                        }}
                                                    >
                                                        <Table size="small" stickyHeader sx={{ 
                                                            width: '100%',
                                                            minWidth: '600px',
                                                            tableLayout: 'fixed',
                                                            display: 'table'
                                                        }}>
                                                            <TableHead>
                                                                <TableRow>
                                                                    {result.execution_result.statistics?.columns.map((col, colIndex) => (
                                                                        <TableCell 
                                                                            key={col}
                                                                            sx={{ 
                                                                                whiteSpace: 'nowrap',
                                                                                width: `${Math.max(120, 600 / (result.execution_result.statistics?.columns.length || 1))}px`,
                                                                                fontWeight: 'bold',
                                                                                bgcolor: 'rgba(25, 118, 210, 0.08)',
                                                                                color: 'primary.dark',
                                                                                borderBottom: '2px solid',
                                                                                borderBottomColor: 'primary.light',
                                                                                overflow: 'hidden',
                                                                                textOverflow: 'ellipsis',
                                                                                padding: '8px'
                                                                            }}
                                                                            title={col}
                                                                        >
                                                                            {col}
                                                                        </TableCell>
                                                                    ))}
                                                                </TableRow>
                                                            </TableHead>
                                                            <TableBody>
                                                                {result.execution_result.data?.slice(0, 100).map((row, rowIndex) => (
                                                                    <TableRow key={rowIndex} sx={{
                                                                        '&:hover': { bgcolor: 'rgba(255, 255, 255, 0.05)' },
                                                                        '&:nth-of-type(even)': { bgcolor: 'rgba(255, 255, 255, 0.02)' }
                                                                    }}>
                                                                        {result.execution_result.statistics?.columns.map((col, colIndex) => (
                                                                            <TableCell 
                                                                                key={col}
                                                                                sx={{ 
                                                                                    whiteSpace: 'nowrap',
                                                                                    width: `${Math.max(120, 600 / (result.execution_result.statistics?.columns.length || 1))}px`,
                                                                                    overflow: 'hidden',
                                                                                    textOverflow: 'ellipsis',
                                                                                    padding: '8px'
                                                                                }}
                                                                                title={String(row[col] || '')}
                                                                            >
                                                                                {String(row[col] || '')}
                                                                            </TableCell>
                                                                        ))}
                                                                    </TableRow>
                                                                ))}
                                                            </TableBody>
                                                        </Table>
                                                    </TableContainer>
                                                </Box>
                                            </Collapse>
                                        )}

                                        {/* Visualization suggestions */}
                                        {result.visualization_suggestions && result.visualization_suggestions.length > 0 && (
                                            <Box sx={{ mb: 2, width: '100%', maxWidth: '100%', overflow: 'hidden' }}>
                                                <Typography variant="h6" gutterBottom sx={{ 
                                                    display: 'flex', 
                                                    alignItems: 'center',
                                                    color: 'secondary.main',
                                                    fontWeight: 'bold',
                                                    overflowWrap: 'break-word',
                                                    wordBreak: 'break-word'
                                                }}>
                                                    <ChartIcon sx={{ mr: 1, flexShrink: 0 }} />
                                                    Suggested Charts
                                                </Typography>
                                                <Box sx={{ 
                                                    display: 'flex', 
                                                    gap: 1, 
                                                    flexWrap: 'wrap', 
                                                    width: '100%', 
                                                    maxWidth: '100%',
                                                    boxSizing: 'border-box'
                                                }}>
                                                    {result.visualization_suggestions.map((suggestion, idx) => (
                                                        <Chip
                                                            key={idx}
                                                            icon={<ChartIcon />}
                                                            label={suggestion.chart_type}
                                                            onClick={() => alert('Chart generation coming soon!')}
                                                            variant="outlined"
                                                            sx={{
                                                                borderColor: 'secondary.main',
                                                                color: 'secondary.main',
                                                                maxWidth: '100%',
                                                                overflow: 'hidden',
                                                                textOverflow: 'ellipsis',
                                                                whiteSpace: 'nowrap',
                                                                '&:hover': {
                                                                    bgcolor: 'rgba(156, 39, 176, 0.1)',
                                                                    borderColor: 'secondary.light'
                                                                }
                                                            }}
                                                        />
                                                    ))}
                                                </Box>
                                            </Box>
                                        )}

                                        {/* SQL Information (collapsible) */}
                                        <Collapse in={shouldShowSql}>
                                            {result.generated_sql && (
                                                <Box sx={{ mb: 2, width: '100%', maxWidth: '100%', overflow: 'hidden' }}>
                                                    <Divider sx={{ mb: 2, borderColor: 'rgba(255, 255, 255, 0.1)' }} />
                                                    <Typography variant="h6" gutterBottom sx={{ 
                                                        color: 'primary.main',
                                                        fontWeight: 'bold',
                                                        display: 'flex',
                                                        alignItems: 'center',
                                                        overflowWrap: 'break-word',
                                                        wordBreak: 'break-word'
                                                    }}>
                                                        <CodeIcon sx={{ mr: 1, flexShrink: 0 }} />
                                                        Generated SQL
                                                    </Typography>
                                                    
                                                    <Paper variant="outlined" sx={{ 
                                                        p: 3, 
                                                        bgcolor: 'rgba(0, 0, 0, 0.4)', 
                                                        borderColor: 'rgba(25, 118, 210, 0.3)',
                                                        borderRadius: 2,
                                                        border: '1px solid rgba(25, 118, 210, 0.3)',
                                                        width: '100%',
                                                        maxWidth: '100%',
                                                        boxSizing: 'border-box',
                                                        overflow: 'auto',
                                                        '&::-webkit-scrollbar': {
                                                            width: '8px',
                                                            height: '8px'
                                                        },
                                                        '&::-webkit-scrollbar-track': {
                                                            bgcolor: 'rgba(255, 255, 255, 0.1)',
                                                            borderRadius: '4px'
                                                        },
                                                        '&::-webkit-scrollbar-thumb': {
                                                            bgcolor: 'rgba(255, 255, 255, 0.3)',
                                                            borderRadius: '4px',
                                                            '&:hover': {
                                                                bgcolor: 'rgba(255, 255, 255, 0.5)'
                                                            }
                                                        }
                                                    }}>
                                                        <code style={{ 
                                                            fontSize: '0.875rem',
                                                            color: '#e3f2fd',
                                                            fontFamily: 'Consolas, Monaco, "Courier New", monospace',
                                                            lineHeight: 1.6,
                                                            whiteSpace: 'pre-wrap',
                                                            overflowWrap: 'break-word',
                                                            wordBreak: 'break-all',
                                                            display: 'block',
                                                            width: '100%',
                                                            maxWidth: '100%',
                                                            boxSizing: 'border-box'
                                                        }}>{result.generated_sql}</code>
                                                    </Paper>
                                                </Box>
                                            )}
                                        </Collapse>

                                        {/* Data Analysis (collapsible) */}
                                        <Collapse in={shouldShowAnalysis}>
                                            <Box sx={{ mb: 2, width: '100%', maxWidth: '100%', overflow: 'hidden' }}>
                                                <Divider sx={{ mb: 2, borderColor: 'rgba(255, 255, 255, 0.1)' }} />
                                                <Typography variant="h6" gutterBottom sx={{ 
                                                    color: 'secondary.main',
                                                    fontWeight: 'bold',
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    overflowWrap: 'break-word',
                                                    wordBreak: 'break-word'
                                                }}>
                                                    <AnalyticsIcon sx={{ mr: 1, flexShrink: 0 }} />
                                                    Data Analysis
                                                </Typography>

                                                {result.insights && (
                                                    <Box sx={{ mb: 2, width: '100%', maxWidth: '100%', overflow: 'hidden' }}>
                                                        <Typography variant="subtitle1" gutterBottom sx={{ 
                                                            fontWeight: 'bold', 
                                                            color: 'info.main',
                                                            overflowWrap: 'break-word',
                                                            wordBreak: 'break-word'
                                                        }}>
                                                            AI Insights
                                                        </Typography>
                                                        <Paper variant="outlined" sx={{ 
                                                            p: 3, 
                                                            bgcolor: 'rgba(0, 150, 255, 0.05)',
                                                            borderColor: 'rgba(0, 150, 255, 0.2)',
                                                            borderRadius: 2,
                                                            width: '100%',
                                                            maxWidth: '100%',
                                                            boxSizing: 'border-box',
                                                            overflow: 'hidden',
                                                            overflowWrap: 'break-word',
                                                            wordBreak: 'break-word',
                                                            '& p': { 
                                                                overflowWrap: 'break-word',
                                                                wordBreak: 'break-word'
                                                            },
                                                            '& code': {
                                                                overflowWrap: 'break-word',
                                                                wordBreak: 'break-all',
                                                                whiteSpace: 'pre-wrap'
                                                            }
                                                        }}>
                                                            <ReactMarkdown>{result.insights}</ReactMarkdown>
                                                        </Paper>
                                                    </Box>
                                                )}

                                                {result.data_analysis && (
                                                    <Box sx={{ mb: 2, width: '100%', maxWidth: '100%', overflow: 'hidden' }}>
                                                        <Typography variant="subtitle1" gutterBottom sx={{ 
                                                            fontWeight: 'bold', 
                                                            color: 'warning.main',
                                                            overflowWrap: 'break-word',
                                                            wordBreak: 'break-word'
                                                        }}>
                                                            Statistical Analysis
                                                        </Typography>
                                                        <Paper variant="outlined" sx={{ 
                                                            p: 3, 
                                                            bgcolor: 'rgba(0, 0, 0, 0.4)', 
                                                            borderColor: 'rgba(255, 152, 0, 0.3)',
                                                            borderRadius: 2,
                                                            border: '1px solid rgba(255, 152, 0, 0.3)',
                                                            width: '100%',
                                                            maxWidth: '100%',
                                                            boxSizing: 'border-box',
                                                            overflow: 'auto',
                                                            '&::-webkit-scrollbar': {
                                                                width: '8px',
                                                                height: '8px'
                                                            },
                                                            '&::-webkit-scrollbar-track': {
                                                                bgcolor: 'rgba(255, 255, 255, 0.1)',
                                                                borderRadius: '4px'
                                                            },
                                                            '&::-webkit-scrollbar-thumb': {
                                                                bgcolor: 'rgba(255, 255, 255, 0.3)',
                                                                borderRadius: '4px',
                                                                '&:hover': {
                                                                    bgcolor: 'rgba(255, 255, 255, 0.5)'
                                                                }
                                                            }
                                                        }}>
                                                            <pre style={{ 
                                                                whiteSpace: 'pre-wrap', 
                                                                fontSize: '0.875rem', 
                                                                margin: 0,
                                                                color: '#fff3e0',
                                                                fontFamily: 'Consolas, Monaco, "Courier New", monospace',
                                                                lineHeight: 1.6,
                                                                overflowWrap: 'break-word',
                                                                wordBreak: 'break-all',
                                                                width: '100%',
                                                                maxWidth: '100%',
                                                                boxSizing: 'border-box'
                                                            }}>
                                                                {JSON.stringify(result.data_analysis, null, 2)}
                                                            </pre>
                                                        </Paper>
                                                    </Box>
                                                )}
                                            </Box>
                                        </Collapse>
                                    </Box>
                                ) : (
                                    <Alert severity="error" sx={{
                                        bgcolor: 'rgba(244, 67, 54, 0.1)',
                                        borderColor: 'rgba(244, 67, 54, 0.3)',
                                        color: 'error.main'
                                    }}>
                                        {result.execution_result?.error || 'Query execution failed'}
                                    </Alert>
                                )}
                            </Box>
                        </Box>
                    </Paper>
                </Grow>
            </Box>
        );
    }, [showSqlMap, showAnalysisMap, toggleSqlVisibility, toggleAnalysisVisibility]);

    const renderDatabaseStatus = () => (
        <Card sx={{ mb: 2 }}>
            <CardContent>
                <Typography variant="h6" gutterBottom>
                    <DatabaseIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                    Database Status
                </Typography>
                {dbStatus ? (
                    <Box>
                        <Chip 
                            label={dbStatus.status} 
                            color={dbStatus.status === 'connected' ? 'success' : 'error'}
                            sx={{ mb: 1 }}
                        />
                        {dbStatus.status === 'connected' && (
                            <Box>
                                <Typography variant="body2">Database: {dbStatus.database}</Typography>
                                <Typography variant="body2">Host: {dbStatus.host}:{dbStatus.port}</Typography>
                                <Typography variant="body2">Version: {dbStatus.version}</Typography>
                            </Box>
                        )}
                        {dbStatus.error && (
                            <Alert severity="error" sx={{ mt: 1 }}>
                                {dbStatus.error}
                            </Alert>
                        )}
                    </Box>
                ) : (
                    <CircularProgress size={20} />
                )}
                <Button 
                    startIcon={<RefreshIcon />} 
                    onClick={checkDatabaseStatus}
                    size="small"
                    sx={{ mt: 1 }}
                >
                    Refresh
                </Button>
            </CardContent>
        </Card>
    );

    const renderDatabaseSchema = () => (
        <Card sx={{ mb: 2 }}>
            <CardContent>
                <Typography variant="h6" gutterBottom>
                    <TableIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
                    Database Schema
                </Typography>
                {Object.keys(schema).length > 0 ? (
                    <Box>
                        {Object.entries(schema).map(([tableName, tableInfo]) => (
                            <Accordion key={tableName}>
                                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                    <Typography variant="subtitle1">
                                        {tableName} ({tableInfo.row_count} rows)
                                    </Typography>
                                </AccordionSummary>
                                <AccordionDetails>
                                    <TableContainer component={Paper} variant="outlined">
                                        <Table size="small">
                                            <TableHead>
                                                <TableRow>
                                                    <TableCell>Column</TableCell>
                                                    <TableCell>Type</TableCell>
                                                    <TableCell>Nullable</TableCell>
                                                    <TableCell>Default</TableCell>
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                {tableInfo.columns.map((column) => (
                                                    <TableRow key={column.name}>
                                                        <TableCell>{column.name}</TableCell>
                                                        <TableCell>{column.type}</TableCell>
                                                        <TableCell>{column.nullable ? 'Yes' : 'No'}</TableCell>
                                                        <TableCell>{column.default || '-'}</TableCell>
                                                    </TableRow>
                                                ))}
                                            </TableBody>
                                        </Table>
                                    </TableContainer>
                                </AccordionDetails>
                            </Accordion>
                        ))}
                    </Box>
                ) : (
                    <Typography variant="body2" color="text.secondary">
                        No tables found or schema not loaded
                    </Typography>
                )}
            </CardContent>
        </Card>
    );

    const rightDrawerContent = () => {
        switch (activeRightTab) {
            case 'status':
                return renderDatabaseStatus();
            case 'schema':
                return renderDatabaseSchema();
            default:
                return (
                    <Box sx={{ p: 2, textAlign: 'center' }}>
                        <Typography variant="h6" color="primary.main" gutterBottom>
                            Database Chat
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                            Ask questions about your database in natural language.
                        </Typography>
                    </Box>
                );
        }
    };

    return (
        <Box sx={{ 
            height: '100vh', 
            width: '100%',
            display: 'flex', 
            flexDirection: 'column',
            position: 'relative'
        }}>
            {/* Main Chat Area */}
            <Box sx={{ 
                flexGrow: 1, 
                overflowY: 'auto', 
                p: { xs: 1, sm: 2, md: 3 }, 
                display: 'flex', 
                flexDirection: 'column',
                pr: rightDrawerOpen ? '320px' : { xs: 1, sm: 2, md: 3 },
                transition: 'padding-right 0.3s ease'
            }}>
                <Box sx={{ flexGrow: 1 }} />
                {results.map((result, index) => renderQueryResult(result, index))}
                {isLoading && (
                    <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
                        <CircularProgress />
                    </Box>
                )}
                <div ref={messagesEndRef} />
            </Box>

            {/* Input Area */}
            <Box sx={{ 
                p: 2, 
                borderTop: '1px solid', 
                borderColor: 'rgba(255, 255, 255, 0.1)', 
                bgcolor: 'rgba(5, 8, 18, 0.7)', 
                backdropFilter: 'blur(10px)',
                pr: rightDrawerOpen ? '320px' : 2,
                transition: 'padding-right 0.3s ease'
            }}>
                <Paper component="form" onSubmit={(e) => { e.preventDefault(); handleSubmitQuery(); }} sx={{ 
                    p: '8px 12px', 
                    display: 'flex', 
                    alignItems: 'center', 
                    borderRadius: '16px', 
                    bgcolor: 'rgba(255, 255, 255, 0.1)',
                    border: '1px solid rgba(255, 255, 255, 0.2)',
                    boxShadow: '0px 2px 10px rgba(0,0,0,0.5)'
                }}>
                    <DatabaseIcon sx={{ color: 'cyan', mr: 1 }} />
                    <TextField
                        sx={{ 
                            ml: 1, 
                            flex: 1, 
                            '& .MuiInputBase-input::placeholder': { color: 'grey.400' },
                            '& .MuiInputBase-input': { color: 'white' }
                        }}
                        placeholder="Ask me about your database... (e.g., 'Show me the top 10 customers by revenue')"
                        fullWidth 
                        multiline 
                        maxRows={5}
                        value={query}
                        onChange={(e) => handleQueryChange(e.target.value)}
                        onKeyDown={(e) => { 
                            if (e.key === 'Enter' && !e.shiftKey) { 
                                e.preventDefault(); 
                                handleSubmitQuery(); 
                            } 
                        }}
                        disabled={isLoading}
                        InputProps={{ disableUnderline: true }}
                    />
                    <IconButton 
                        type="submit" 
                        color="primary" 
                        disabled={isLoading || !query.trim()}
                        sx={{ ml: 1 }}
                    >
                        <SendIcon />
                    </IconButton>
                    {results.length > 0 && (
                        <IconButton
                            onClick={clearChat}
                            size="small"
                            sx={{ ml: 1, color: 'grey.400' }}
                            title="Clear Chat"
                        >
                            <ClearIcon />
                        </IconButton>
                    )}
                </Paper>
            </Box>

            {/* Floating Right Panel Toggle Buttons */}
            <Box sx={{
                position: 'fixed',
                right: 16,
                top: '50%',
                transform: 'translateY(-50%)',
                display: 'flex',
                flexDirection: 'column',
                gap: 1,
                zIndex: 1000
            }}>
                <Fab
                    size="small"
                    onClick={() => {
                        setActiveRightTab('chat');
                        setRightDrawerOpen(!rightDrawerOpen);
                    }}
                    sx={{
                        bgcolor: activeRightTab === 'chat' && rightDrawerOpen ? 'primary.main' : 'rgba(255, 255, 255, 0.1)',
                        '&:hover': { bgcolor: 'primary.main' }
                    }}
                >
                    <DatabaseIcon />
                </Fab>
                <Fab
                    size="small"
                    onClick={() => {
                        setActiveRightTab('schema');
                        setRightDrawerOpen(true);
                    }}
                    sx={{
                        bgcolor: activeRightTab === 'schema' && rightDrawerOpen ? 'secondary.main' : 'rgba(255, 255, 255, 0.1)',
                        '&:hover': { bgcolor: 'secondary.main' }
                    }}
                >
                    <TableIcon />
                </Fab>
                <Fab
                    size="small"
                    onClick={() => {
                        setActiveRightTab('status');
                        setRightDrawerOpen(true);
                    }}
                    sx={{
                        bgcolor: activeRightTab === 'status' && rightDrawerOpen ? 'info.main' : 'rgba(255, 255, 255, 0.1)',
                        '&:hover': { bgcolor: 'info.main' }
                    }}
                >
                    <InfoIcon />
                </Fab>
            </Box>

            {/* Right Drawer */}
            <Drawer
                anchor="right"
                open={rightDrawerOpen}
                onClose={() => setRightDrawerOpen(false)}
                variant="persistent"
                sx={{
                    '& .MuiDrawer-paper': {
                        width: 300,
                        bgcolor: 'rgba(5, 8, 18, 0.95)',
                        borderLeft: '1px solid rgba(255, 255, 255, 0.1)',
                        backdropFilter: 'blur(15px)',
                        color: 'text.primary'
                    }
                }}
            >
                <Box sx={{ p: 2, borderBottom: '1px solid rgba(255, 255, 255, 0.1)' }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Typography variant="h6" sx={{ fontWeight: 'bold' }}>
                            {activeRightTab === 'chat' && 'Chat Info'}
                            {activeRightTab === 'schema' && 'Schema'}
                            {activeRightTab === 'status' && 'Status'}
                        </Typography>
                        <IconButton 
                            onClick={() => setRightDrawerOpen(false)}
                            size="small"
                        >
                            <ClearIcon />
                        </IconButton>
                    </Box>
                </Box>
                <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
                    {rightDrawerContent()}
                </Box>
            </Drawer>
        </Box>
    );
};

export default DatabaseChat; 