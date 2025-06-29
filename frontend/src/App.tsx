import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
    ThemeProvider, Box, TextField, Button, Paper, Typography, AppBar, Toolbar, CircularProgress,
    Drawer, List, ListItem, ListItemButton, ListItemText, IconButton, Divider, ListItemIcon, Chip, Grow, CssBaseline,
    Modal, Switch, FormControlLabel
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import HistoryIcon from '@mui/icons-material/History';
import AddCommentIcon from '@mui/icons-material/AddComment';
import MenuIcon from '@mui/icons-material/Menu';
import FolderIcon from '@mui/icons-material/Folder';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import DeleteIcon from '@mui/icons-material/Delete';
import DeleteSweepIcon from '@mui/icons-material/DeleteSweep';
import CloseIcon from '@mui/icons-material/Close';
import ArrowBackIosNewIcon from '@mui/icons-material/ArrowBackIosNew';
import ArrowForwardIosIcon from '@mui/icons-material/ArrowForwardIos';
import PsychologyIcon from '@mui/icons-material/Psychology';
import StorageIcon from '@mui/icons-material/Storage';
import { v4 as uuidv4 } from 'uuid';
import { formatDistanceToNow } from 'date-fns';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { AnimatePresence, motion } from 'framer-motion';

import theme from './theme';
import DatabaseChat from './components/DatabaseChat';

// --- Type Definitions ---
interface SourceInfo {
    question: string;
    source_document: string;
    page_num: number;
}

interface Message {
    id: string;
    type: 'user' | 'bot';
    text: string;
    images?: string[];
    sources?: SourceInfo[];
    isLoading?: boolean;
    answer_text?: string;
    isStreaming?: boolean;
}

interface Conversation {
    id: string;
    title: string;
    messages: Message[];
    lastModified: number;
}

interface RetrievedContext extends SourceInfo {
    images: string[];
    answer_text: string;
}

interface ApiAskResponse {
    llm_answer: string;
    retrieved_context: RetrievedContext[];
    sources: { question: string, source_document: string, page_num: number, images?: string[] }[];
    reasoning_process?: string;
    unanswered_aspects?: string[];
}

// --- Main App Component ---
function App() {
    const [conversations, setConversations] = useState<Record<string, Conversation>>({});
    const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
    const [input, setInput] = useState('');
    const [isSidebarVisible, setSidebarVisible] = useState(true);
    const [isUploading, setIsUploading] = useState(false);
    const [isSending, setIsSending] = useState(false);
    const [uploadError, setUploadError] = useState<string | null>(null);
    const [documents, setDocuments] = useState<string[]>([]);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const abortControllerRef = useRef<AbortController | null>(null);
    
    // State for image modal
    const [modalOpen, setModalOpen] = useState(false);
    const [modalImages, setModalImages] = useState<string[]>([]);
    const [selectedImage, setSelectedImage] = useState({ index: 0, direction: 0 });

    const drawerWidth = 280;

    const API_BASE_URL = 'http://localhost:8000';

    const activeMessages = useMemo(() => 
        activeConversationId ? conversations[activeConversationId]?.messages || [] : [],
        [activeConversationId, conversations]
    );

    const [useCot, setUseCot] = useState(false);
    const [currentMode, setCurrentMode] = useState<'rag' | 'database'>('rag');

    // --- Data Fetching and State Synchronization ---
    const fetchDocuments = async () => {
        try {
            const response = await axios.get<string[]>(`${API_BASE_URL}/documents`);
            setDocuments(response.data);
        } catch (error) {
            console.error('Error fetching documents:', error);
        }
    };

    useEffect(() => {
        const storedConversationsRaw = localStorage.getItem('conversations');
        const storedActiveId = localStorage.getItem('activeConversationId');
        let loadedConversations: Record<string, Conversation> = {};
    
        if (storedConversationsRaw) {
            try {
                loadedConversations = JSON.parse(storedConversationsRaw);
            } catch (e) {
                console.error("Failed to parse conversations from localStorage", e);
                // If parsing fails, start fresh
                localStorage.removeItem('conversations');
                localStorage.removeItem('activeConversationId');
            }
        }
        setConversations(loadedConversations);
    
        const conversationExists = storedActiveId && loadedConversations[storedActiveId];
        if (conversationExists) {
            setActiveConversationId(storedActiveId);
        } else if (Object.keys(loadedConversations).length > 0) {
            const mostRecentId = Object.values(loadedConversations).sort((a,b) => b.lastModified - a.lastModified)[0].id;
            setActiveConversationId(mostRecentId);
        } else {
            handleNewConversation();
        }
    
        fetchDocuments();
         // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
        if (Object.keys(conversations).length > 0) {
            try {
                // Remove images from conversations before saving to localStorage to save space
                const conversationsWithoutImages: { [key: string]: Conversation } = Object.fromEntries(
                    Object.entries(conversations).map(([id, conversation]: [string, Conversation]) => [
                        id,
                        {
                            ...conversation,
                            messages: conversation.messages.map((message: Message) => ({
                                ...message,
                                images: undefined // Don't save images to localStorage
                            }))
                        }
                    ])
                );
                
                localStorage.setItem('conversations', JSON.stringify(conversationsWithoutImages));
            } catch (error) {
                console.error('Failed to save conversations to localStorage:', error);
                
                // If quota exceeded, try to clean up old conversations
                if (error instanceof DOMException && error.name === 'QuotaExceededError') {
                    console.log('localStorage quota exceeded, cleaning up old conversations...');
                    try {
                        const conversationEntries = Object.entries(conversations);
                        // Keep only the 5 most recent conversations
                                                 const recentConversations: { [key: string]: Conversation } = conversationEntries
                             .sort(([,a], [,b]) => b.lastModified - a.lastModified)
                             .slice(0, 5)
                             .reduce((acc: { [key: string]: Conversation }, [id, conv]: [string, Conversation]) => ({ ...acc, [id]: conv }), {});
                        
                                                 // Remove images and try saving again
                         const cleanedConversations: { [key: string]: Conversation } = Object.fromEntries(
                             Object.entries(recentConversations).map(([id, conversation]: [string, Conversation]) => [
                                 id,
                                 {
                                     ...conversation,
                                     messages: conversation.messages.map((message: Message) => ({
                                         ...message,
                                         images: undefined
                                     }))
                                 }
                             ])
                         );
                        
                        localStorage.setItem('conversations', JSON.stringify(cleanedConversations));
                        
                        // Update state to reflect the cleaned conversations
                        setConversations(recentConversations);
                        
                        console.log('Successfully cleaned up localStorage - kept 5 most recent conversations');
                    } catch (cleanupError) {
                        console.error('Failed to clean up localStorage:', cleanupError);
                        // As last resort, clear all localStorage
                        localStorage.removeItem('conversations');
                        localStorage.removeItem('activeConversationId');
                        console.log('Cleared all localStorage as last resort');
                    }
                }
            }
        }
        if (activeConversationId) {
            try {
                localStorage.setItem('activeConversationId', activeConversationId);
            } catch (error) {
                console.error('Failed to save activeConversationId:', error);
            }
        }
        scrollToBottom();
    }, [conversations, activeConversationId]);

    useEffect(() => {
        scrollToBottom();
    }, [activeMessages]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'auto' });
    };

    // --- Conversation Management ---
    const handleNewConversation = () => {
        const newId = uuidv4();
        const newConversation: Conversation = {
            id: newId,
            title: 'New Conversation',
            messages: [],
            lastModified: Date.now(),
        };
        setConversations(prev => ({ ...prev, [newId]: newConversation }));
        setActiveConversationId(newId);
    };

    const switchConversation = (id: string) => {
        setActiveConversationId(id);
    };

    const handleDeleteConversation = (e: React.MouseEvent, idToDelete: string) => {
        e.stopPropagation(); // Prevent the ListItemButton's onClick from firing

        if (window.confirm('Are you sure you want to permanently delete this conversation?')) {
            const newConversations = { ...conversations };
            delete newConversations[idToDelete];
            setConversations(newConversations);

            if (activeConversationId === idToDelete) {
                const remainingIds = Object.keys(newConversations);
                if (remainingIds.length > 0) {
                    const mostRecentId = Object.values(newConversations).sort((a,b) => b.lastModified - a.lastModified)[0].id;
                    setActiveConversationId(mostRecentId);
                } else {
                    handleNewConversation();
                }
            }
        }
    };

    const handleClearMessages = () => {
        if (!activeConversationId) return;

        if (window.confirm('Are you sure you want to clear all messages in this conversation?')) {
            setConversations(prev => {
                const convoToUpdate = prev[activeConversationId];
                if (!convoToUpdate) return prev;
                
                return {
                    ...prev,
                    [activeConversationId]: {
                        ...convoToUpdate,
                        messages: [],
                    },
                };
            });
        }
    };

    // --- Core Actions: Send Message, Upload, Delete ---
    const handleSendMessage = async () => {
        if (!input.trim() || !activeConversationId || isUploading || isSending) return;

        setIsSending(true);
        const userInput: Message = { id: uuidv4(), type: 'user', text: input.trim() };
        const loadingBotMessage: Message = { 
            id: uuidv4(), 
            type: 'bot', 
            text: useCot ? 'I\'m planning my approach to answer your question...' : 'I\'m searching the knowledge base for relevant information...', 
            isLoading: false, 
            isStreaming: true 
        };

        setConversations(prev => {
            const prevConvo = prev[activeConversationId];
            if (!prevConvo) return prev;

            const updatedConvo: Conversation = {
                ...prevConvo,
                messages: [...prevConvo.messages, userInput, loadingBotMessage],
                lastModified: Date.now(),
                title: prevConvo.title === 'New Conversation'
                    ? input.trim().substring(0, 35) + (input.trim().length > 35 ? '...' : '')
                    : prevConvo.title,
            };

            return {
                ...prev,
                [activeConversationId]: updatedConvo,
            };
        });
        
        setInput('');

        // Show initial feedback quickly even before API responds
        setTimeout(() => {
            setConversations(prev => {
                const prevConvo = prev[activeConversationId!];
                if (!prevConvo) return prev;
                
                // Only update if still in loading state
                if (prevConvo.messages.find(m => m.id === loadingBotMessage.id)?.isStreaming) {
                    return {
                        ...prev,
                        [activeConversationId!]: {
                            ...prevConvo,
                            messages: prevConvo.messages.map(m => 
                                m.id === loadingBotMessage.id 
                                    ? { ...m, text: 'I found some relevant information and am analyzing it now...' } 
                                    : m
                            ),
                        }
                    };
                }
                return prev;
            });
        }, 1500);

        // Cancel any existing request
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
        }
        
        // Create new abort controller for this request
        abortControllerRef.current = new AbortController();
        const signal = abortControllerRef.current.signal;
        
        // Start loading full response with sources and images in parallel
        // This will significantly reduce the waiting time after streaming completes
        const endpoint = useCot ? `${API_BASE_URL}/ask/cot` : `${API_BASE_URL}/ask`;
        const fullResponsePromise = axios.post<ApiAskResponse>(endpoint, { query: userInput.text })
            .catch(error => {
                console.error('Error fetching full response:', error);
                return null;
            });

        try {
            // Check if backend is ready, retry with exponential backoff if necessary
            let streamResponse = null;
            let retryCount = 0;
            const maxRetries = 3;
            const initialDelay = 500;
            
            while (!streamResponse && retryCount < maxRetries) {
                try {
                    streamResponse = await fetch(`${API_BASE_URL}/ask/stream`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ query: userInput.text }),
                        signal
                    });
                    
                    if (!streamResponse.ok) {
                        // If server is still warming up, retry
                        if (streamResponse.status === 503 || streamResponse.status === 429) {
                            const delay = initialDelay * Math.pow(2, retryCount);
                            console.log(`Server busy, retrying in ${delay}ms...`);
                            await new Promise(resolve => setTimeout(resolve, delay));
                            retryCount++;
                            streamResponse = null;
                        } else {
                            throw new Error(`Stream API responded with ${streamResponse.status}`);
                        }
                    }
                } catch (error) {
                    if ((error as Error).name === 'AbortError') {
                        throw error;  // Propagate abort errors
                    }
                    
                    const delay = initialDelay * Math.pow(2, retryCount);
                    console.log(`Error connecting to streaming API, retrying in ${delay}ms...`, error);
                    await new Promise(resolve => setTimeout(resolve, delay));
                    retryCount++;
                    
                    if (retryCount >= maxRetries) {
                        throw error;  // Stop retrying after max attempts
                    }
                }
            }
            
            if (!streamResponse) {
                throw new Error('Failed to connect to streaming API after multiple attempts');
            }

            if (!streamResponse.body) {
                throw new Error('ReadableStream not supported');
            }

            let fullText = '';
            const reader = streamResponse.body.getReader();
            const decoder = new TextDecoder();
            
            const processStream = async (): Promise<void> => {
                const { done, value } = await reader.read();
                
                if (done) {
                    // After streaming is complete, get the full response that was loading in parallel
                    try {
                        const fullResponse = await fullResponsePromise;
                        
                        if (fullResponse) {
                            const { sources, retrieved_context, reasoning_process, unanswered_aspects } = fullResponse.data;
                            
                            // Log the data for debugging
                            console.log('=== FRONTEND PROCESSING ===');
                            console.log('LLM Sources:', sources);
                            console.log('Retrieved Context:', retrieved_context);
                            if (reasoning_process) console.log('Reasoning Process:', reasoning_process);
                            if (unanswered_aspects) console.log('Unanswered Aspects:', unanswered_aspects);
                            
                            // Trust the LLM's source information
                            let sources_to_display: SourceInfo[] = [];
                            let images_to_display: string[] = [];
                            
                            if (sources && sources.length > 0) {
                                // Use the LLM's source information directly
                                sources_to_display = sources;
                                console.log('Using LLM sources directly:', sources_to_display);
                            }
                            
                            // Get images from retrieved_context (this contains the actual QA block images)
                            if (retrieved_context && retrieved_context.length > 0) {
                                images_to_display = retrieved_context.flatMap(context => context.images || []);
                                console.log('Images from retrieved context:', images_to_display);
                            }
                            
                            console.log('Final sources to display (from LLM):', sources_to_display);
                            console.log('Final images to display:', images_to_display);
                            console.log('=== END FRONTEND PROCESSING ===');
                            
                            // Add sources and images to the completed message
                            setConversations(prev => {
                                const prevConvo = prev[activeConversationId!];
                                if (!prevConvo) return prev;
                                
                                // Prepare additional CoT information if available
                                let additionalText = '';
                                if (useCot && reasoning_process) {
                                    additionalText = '\n\n**Reasoning Process:**\n' + reasoning_process;
                                }
                                if (useCot && unanswered_aspects && unanswered_aspects.length > 0) {
                                    additionalText += '\n\n**Aspects I couldn\'t answer from the provided context:**\n- ' + 
                                        unanswered_aspects.join('\n- ');
                                }
                                
                                return {
                                    ...prev,
                                    [activeConversationId!]: {
                                        ...prevConvo,
                                        messages: prevConvo.messages.map(m => 
                                            m.id === loadingBotMessage.id 
                                                ? {
                                                    ...m,
                                                    isStreaming: false,
                                                    text: fullText + additionalText,
                                                    sources: sources_to_display,
                                                    images: images_to_display,
                                                }
                                                : m
                                        ),
                                    }
                                };
                            });
                        } else {
                            // Just mark streaming as complete if we couldn't get sources/images
                            setConversations(prev => {
                                const prevConvo = prev[activeConversationId!];
                                if (!prevConvo) return prev;
                                
                                return {
                                    ...prev,
                                    [activeConversationId!]: {
                                        ...prevConvo,
                                        messages: prevConvo.messages.map(m => 
                                            m.id === loadingBotMessage.id ? { ...m, isStreaming: false } : m
                                        ),
                                    }
                                };
                            });
                        }
                    } catch (error) {
                        console.error('Error processing full response:', error);
                        // Just mark streaming as complete if there was an error
                        setConversations(prev => {
                            const prevConvo = prev[activeConversationId!];
                            if (!prevConvo) return prev;
                            
                            return {
                                ...prev,
                                [activeConversationId!]: {
                                    ...prevConvo,
                                    messages: prevConvo.messages.map(m => 
                                        m.id === loadingBotMessage.id ? { ...m, isStreaming: false } : m
                                    ),
                                }
                            };
                        });
                    }
                    
                    setIsSending(false);
                    return;
                }
                
                fullText += decoder.decode(value, {stream: true});
                
                // Update the streaming message text
                setConversations(prev => {
                    const prevConvo = prev[activeConversationId!];
                    if (!prevConvo) return prev;
                    
                    return {
                        ...prev,
                        [activeConversationId!]: {
                            ...prevConvo,
                            messages: prevConvo.messages.map(m => 
                                m.id === loadingBotMessage.id 
                                    ? { ...m, text: fullText } 
                                    : m
                            ),
                        }
                    };
                });
                
                // Continue reading the stream
                return processStream();
            };
            
            // Start processing the stream
            await processStream();
        } catch (error) {
            if ((error as Error).name === 'AbortError') {
                console.log('Request was aborted');
            } else {
                console.error('Error with streaming:', error);
                const errorBotResponse: Message = {
                    id: loadingBotMessage.id,
                    type: 'bot',
                    text: 'Sorry, there was an error communicating with the backend. Please try again later.',
                    isLoading: false,
                    isStreaming: false,
                };
                
                setConversations(prev => {
                    const prevConvo = prev[activeConversationId!];
                    if (!prevConvo) return prev;
                    
                    return {
                        ...prev,
                        [activeConversationId!]: {
                            ...prevConvo,
                            messages: prevConvo.messages.map(m => 
                                m.id === loadingBotMessage.id ? errorBotResponse : m
                            ),
                        }
                    };
                });
            }
        } finally {
            setIsSending(false);
            abortControllerRef.current = null;
        }
    };

    const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setIsUploading(true);
        setUploadError(null);
        const formData = new FormData();
        formData.append('file', file);

        try {
            await axios.post(`${API_BASE_URL}/upload`, formData);
            fetchDocuments();
        } catch (error) {
            if (axios.isAxiosError(error) && error.response) {
                setUploadError(error.response.data.detail || 'Failed to upload file.');
            } else {
                setUploadError('An unknown error occurred during upload.');
            }
        } finally {
            setIsUploading(false);
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
        }
    };

    const handleDeleteDocument = async (filename: string) => {
        if (window.confirm(`Are you sure you want to delete ${filename}? This cannot be undone.`)) {
            try {
                await axios.delete(`${API_BASE_URL}/documents/${filename}`);
                fetchDocuments();
            } catch (error) {
                console.error('Error deleting document:', error);
                alert('Failed to delete the document.');
            }
        }
    };

    const handleImageClick = (images: string[], index: number) => {
        setModalImages(images);
        setSelectedImage({ index, direction: 0 });
        setModalOpen(true);
    };

    const handleCloseModal = () => {
        setModalOpen(false);
    };

    const handlePrevImage = () => {
        setSelectedImage(prev => ({
            index: (prev.index - 1 + modalImages.length) % modalImages.length,
            direction: -1,
        }));
    };
    
    const handleNextImage = () => {
        setSelectedImage(prev => ({
            index: (prev.index + 1) % modalImages.length,
            direction: 1,
        }));
    };

    const imageVariants = {
        enter: (direction: number) => ({
            x: direction > 0 ? '100%' : '-100%',
            opacity: 0,
        }),
        center: {
            x: 0,
            opacity: 1,
        },
        exit: (direction: number) => ({
            x: direction < 0 ? '100%' : '-100%',
            opacity: 0,
        }),
    };

  return (
        <ThemeProvider theme={theme}>
            <Box sx={{ display: 'flex', height: '100vh', bgcolor: 'background.default', color: 'text.primary' }}>
                <CssBaseline />
                <AppBar 
                    position="fixed" 
                    color="transparent" 
                    elevation={0} 
                    sx={{ 
                        borderBottom: '1px solid',
                        borderColor: 'grey.800',
                        backdropFilter: 'blur(10px)', 
                        bgcolor: 'rgba(5, 8, 18, 0.7)',
                        transition: theme.transitions.create(['margin', 'width'], {
                            easing: theme.transitions.easing.sharp,
                            duration: theme.transitions.duration.leavingScreen,
                        }),
                        ...(isSidebarVisible && {
                            width: `calc(100% - ${drawerWidth}px)`,
                            marginLeft: `${drawerWidth}px`,
                            transition: theme.transitions.create(['margin', 'width'], {
                                easing: theme.transitions.easing.easeOut,
                                duration: theme.transitions.duration.enteringScreen,
                            }),
                        }),
                    }}
                >
                    <Toolbar>
                        <IconButton 
                            color="inherit" 
                            aria-label="open drawer" 
                            onClick={() => setSidebarVisible(!isSidebarVisible)} 
                            edge="start" 
                            sx={{ mr: 2, ...(isSidebarVisible && { display: 'none' }) }}
                        >
                            <MenuIcon />
                        </IconButton>
                        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                            {activeConversationId ? conversations[activeConversationId]?.title || 'Agentic RAG Assistant' : 'Agentic RAG Assistant'}
                        </Typography>
                        
                        <FormControlLabel
                            control={
                                <Switch 
                                    checked={useCot}
                                    onChange={(e) => setUseCot(e.target.checked)}
                                    color="primary"
                                />
                            }
                            label={
                                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                    <PsychologyIcon sx={{ mr: 0.5 }} />
                                    <Typography variant="body2">Chain-of-Thought</Typography>
                                </Box>
                            }
                            sx={{ mr: 2 }}
                        />
                        
                        <Button
                            variant="outlined"
                            startIcon={<AddCommentIcon />}
                            onClick={handleNewConversation}
                            sx={{ ml: 1 }}
                        >
                            New Chat
                        </Button>
                        
                        {activeConversationId && conversations[activeConversationId]?.messages.length > 0 && (
                            <IconButton onClick={handleClearMessages} title="Clear all messages in this chat" sx={{ ml: 1 }}>
                                <DeleteSweepIcon />
                            </IconButton>
                        )}
                    </Toolbar>
                </AppBar>
                <Drawer
                    variant="persistent"
                    open={isSidebarVisible}
                    sx={{
                        width: drawerWidth, flexShrink: 0, '& .MuiDrawer-paper': {
                            width: drawerWidth, boxSizing: 'border-box', bgcolor: 'rgba(5, 8, 18, 0.95)',
                            borderRight: '1px solid', borderColor: 'rgba(0, 255, 255, 0.2)', backdropFilter: 'blur(15px)',
                            color: 'text.primary', display: 'flex', flexDirection: 'column'
                        },
                    }}
                >
                    <Toolbar sx={{ justifyContent: 'space-between' }}>
                        <Typography variant="h5" sx={{ fontWeight: 'bold' }}>Agentic<span style={{ color: '#0ff' }}>RAG</span></Typography>
                        <IconButton onClick={() => setSidebarVisible(false)}>
                            <MenuIcon />
                        </IconButton>
                    </Toolbar>
                    <Divider sx={{ borderColor: 'grey.700' }} />

                    <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'grey.800', flexShrink: 0 }}>
                        <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}><FolderIcon sx={{ mr: 1 }} /> Knowledge Base</Typography>
                        <Button component="label" variant="outlined" color="primary" fullWidth startIcon={<CloudUploadIcon />} disabled={isUploading}>
                            {isUploading ? 'Uploading...' : 'Upload PDF'}
                            <input type="file" hidden accept="application/pdf" onChange={handleFileUpload} ref={fileInputRef} />
                        </Button>
                        {uploadError && <Typography color="error" sx={{ mt: 1, fontSize: '0.8rem' }}>{uploadError}</Typography>}
                        <List sx={{ mt: 2, overflowY: 'auto' }}>
                            {documents.map((doc) => (
                                <ListItem key={doc} disablePadding sx={{ bgcolor: 'rgba(255, 255, 255, 0.05)', mb: 1, borderRadius: 1, '&:hover .delete-icon': { opacity: 1 } }}>
                                    <ListItemButton sx={{ py: 0.5 }}>
                                        <ListItemIcon sx={{ minWidth: 32 }}><InsertDriveFileIcon fontSize="small" /></ListItemIcon>
                                        <ListItemText primary={doc} primaryTypographyProps={{ sx: { fontSize: '0.9rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' } }} />
                                        <IconButton onClick={() => handleDeleteDocument(doc)} size="small" className="delete-icon" sx={{ opacity: 0, transition: 'opacity 0.2s' }}><DeleteIcon fontSize="small" /></IconButton>
                                    </ListItemButton>
                                </ListItem>
                            ))}
                        </List>
                    </Box>

                    {/* Mode Selection */}
                    <Box sx={{ p: 2, borderBottom: '1px solid', borderColor: 'grey.800', flexShrink: 0 }}>
                        <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
                            Chat Mode
                        </Typography>
                        <List>
                            <ListItem disablePadding sx={{ mb: 1 }}>
                                <ListItemButton 
                                    selected={currentMode === 'rag'}
                                    onClick={() => setCurrentMode('rag')}
                                    sx={{ 
                                        borderRadius: 1,
                                        '&.Mui-selected': { 
                                            bgcolor: 'rgba(0, 255, 255, 0.1)', 
                                            borderLeft: '3px solid', 
                                            borderColor: 'primary.main' 
                                        }
                                    }}
                                >
                                    <ListItemIcon sx={{ minWidth: 32 }}>
                                        <FolderIcon />
                                    </ListItemIcon>
                                    <ListItemText 
                                        primary="Documents (RAG)" 
                                        secondary="Chat with your PDF documents"
                                        primaryTypographyProps={{ sx: { fontWeight: 'medium' } }}
                                        secondaryTypographyProps={{ sx: { fontSize: '0.75rem' } }}
                                    />
                                </ListItemButton>
                            </ListItem>
                            <ListItem disablePadding>
                                <ListItemButton 
                                    selected={currentMode === 'database'}
                                    onClick={() => setCurrentMode('database')}
                                    sx={{ 
                                        borderRadius: 1,
                                        '&.Mui-selected': { 
                                            bgcolor: 'rgba(0, 255, 255, 0.1)', 
                                            borderLeft: '3px solid', 
                                            borderColor: 'primary.main' 
                                        }
                                    }}
                                >
                                    <ListItemIcon sx={{ minWidth: 32 }}>
                                        <StorageIcon />
                                    </ListItemIcon>
                                    <ListItemText 
                                        primary="Database Chat" 
                                        secondary="Query and analyze your database"
                                        primaryTypographyProps={{ sx: { fontWeight: 'medium' } }}
                                        secondaryTypographyProps={{ sx: { fontSize: '0.75rem' } }}
                                    />
                                </ListItemButton>
                            </ListItem>
                        </List>
                    </Box>

                    <Box sx={{ p: 2, flexGrow: 1, overflowY: 'auto' }}>
                        <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}><HistoryIcon sx={{ mr: 1 }} /> Chat History</Typography>
                        <List>
                            {Object.values(conversations).sort((a,b) => b.lastModified - a.lastModified).map((convo) => (
                                <ListItem 
                                    key={convo.id}
                                    disablePadding
                                    sx={{ 
                                        mb: 0.5,
                                        '&:hover .delete-convo-icon': { opacity: 1 },
                                        '& .MuiListItemButton-root': { borderRadius: 1, transition: 'background-color 0.3s' },
                                        '& .Mui-selected': { bgcolor: 'rgba(0, 255, 255, 0.1) !important', borderLeft: '3px solid', borderColor: 'primary.main', pl: '13px' },
                                        '& .MuiListItemButton-root:hover': { bgcolor: 'rgba(0, 255, 255, 0.05)' }
                                    }}
                                    secondaryAction={
                                        <IconButton edge="end" aria-label="delete" onClick={(e) => handleDeleteConversation(e, convo.id)} className="delete-convo-icon" sx={{ opacity: 0, transition: 'opacity 0.2s' }} title="Delete conversation">
                                            <DeleteIcon fontSize="small" />
                                        </IconButton>
                                    }
                                >
                                    <ListItemButton onClick={() => switchConversation(convo.id)} selected={activeConversationId === convo.id}>
                                        <ListItemText 
                                            primary={convo.title}
                                            secondary={formatDistanceToNow(new Date(convo.lastModified), { addSuffix: true })}
                                            primaryTypographyProps={{ sx: { color: 'text.primary', fontWeight: 'medium' } }}
                                            secondaryTypographyProps={{ sx: { color: 'text.secondary' } }}
                                        />
                                    </ListItemButton>
                                </ListItem>
                            ))}
                        </List>
                    </Box>
                </Drawer>

                <Box 
                    component="main" 
                    sx={{ 
                        flexGrow: 1, 
                        display: 'flex', 
                        flexDirection: 'column', 
                        height: '100vh',
                        transition: theme.transitions.create('margin', {
                            easing: theme.transitions.easing.sharp,
                            duration: theme.transitions.duration.leavingScreen,
                        }),
                        marginLeft: `-${drawerWidth}px`,
                        ...(isSidebarVisible && {
                            transition: theme.transitions.create('margin', {
                                easing: theme.transitions.easing.easeOut,
                                duration: theme.transitions.duration.enteringScreen,
                            }),
                            marginLeft: 0,
                        }),
                    }}
                >
                    <Toolbar /> {/* Spacer for the fixed AppBar */}
                    {currentMode === 'rag' ? (
                        <>
                            <Box sx={{ flexGrow: 1, overflowY: 'auto', p: { xs: 1, sm: 2, md: 3 }, display: 'flex', flexDirection: 'column' }}>
                                <Box sx={{ flexGrow: 1 }} /> 
                                {activeMessages.map((msg) => (
                                    <Grow in={true} key={msg.id} timeout={500}>
                                        <Paper elevation={0} sx={{
                                            p: 2, mt: 1, mb: 1, maxWidth: '85%',
                                            alignSelf: msg.type === 'user' ? 'flex-end' : 'flex-start',
                                            bgcolor: msg.type === 'user' ? 'rgba(0, 100, 255, 0.2)' : 'rgba(255, 255, 255, 0.08)',
                                            borderRadius: msg.type === 'user' ? '20px 20px 5px 20px' : '20px 20px 20px 5px',
                                            border: '1px solid', borderColor: msg.type === 'user' ? 'rgba(0, 150, 255, 0.7)' : 'rgba(255, 255, 255, 0.25)',
                                        }}>
                                            <Box sx={{ display: 'flex', alignItems: 'flex-start' }}>
                                                <Box sx={{
                                                    width: 32, height: 32, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                    bgcolor: 'grey.800', color: 'text.primary', mr: 1.5, fontSize: '0.8rem', fontWeight: 'bold', boxShadow: '0 0 5px rgba(0, 255, 255, 0.5)', flexShrink: 0
                                                }}>{msg.type === 'user' ? 'U' : 'A'}</Box>
                                                <Box sx={{ flex: 1, overflow: 'hidden', '& a': { color: 'primary.main' }, '& p': {margin: 0}, '& ul, & ol': {pl: 2.5, my: 1}, '& li': {mb: 0.5} }}>
                                                    {msg.isLoading ? <CircularProgress size={20} /> : <ReactMarkdown children={msg.text} remarkPlugins={[remarkGfm]} />}
                                                    {msg.isStreaming && !msg.isLoading && (
                                                        <Box component="span" className="streaming-cursor" />
                                                    )}
                                                    {msg.images && msg.images.length > 0 && (
                                                        <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                                                            {msg.images.map((img, i) => 
                                                                <img 
                                                                    key={i} 
                                                                    src={img} 
                                                                    alt={`detail ${i + 1}`} 
                                                                    style={{ maxWidth: '150px', maxHeight: '150px', borderRadius: '4px', border: '1px solid #0ff', cursor: 'pointer' }}
                                                                    onClick={() => handleImageClick(msg.images || [], i)}
                                                                />
                                                            )}
                                                        </Box>
                                                    )}
                                                    {msg.sources && msg.sources.length > 0 && (
                                                        <Box sx={{ mt: 1.5, display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                                                            <Typography variant="caption" sx={{ color: 'grey.400', fontSize: '0.7rem' }}>Sources:</Typography>
                                                            {msg.sources.map((source, i) => {
                                                                // Defensive check for old data format (string) vs new format (SourceInfo object)
                                                                const isNewFormat = typeof source === 'object' && source !== null && 'question' in source;
                                                                let label = '';
                                                                
                                                                if (isNewFormat) {
                                                                    // Extract question number/identifier (e.g., "Q5" from "Q5. What information can I see in HA Go?")
                                                                    const questionMatch = source.question.match(/^(Q\d+)/i);
                                                                    const questionId = questionMatch ? questionMatch[1] : `Q${i + 1}`;
                                                                    label = `${questionId} from ${source.source_document} (Page ${source.page_num})`;
                                                                } else {
                                                                    label = String(source);
                                                                }

                                                                return (
                                                                    <Chip 
                                                                        key={i} 
                                                                        icon={<InsertDriveFileIcon />} 
                                                                        label={label}
                                                                        size="small" 
                                                                        sx={{ 
                                                                            bgcolor: 'rgba(255, 255, 255, 0.05)', 
                                                                            color: 'grey.400', 
                                                                            fontSize: '0.75rem',
                                                                            border: '1px solid rgba(255, 255, 255, 0.1)',
                                                                            height: '24px',
                                                                            '& .MuiChip-icon': { color: 'grey.500', fontSize: '1rem', ml: '6px' },
                                                                            '&:hover': { bgcolor: 'rgba(255, 255, 255, 0.15)' }
                                                                        }} 
                                                                    />
                                                                );
                                                            })}
                                                        </Box>
                                                    )}
                                                </Box>
                                            </Box>
                                        </Paper>
                                    </Grow>
                                ))}
                                <div ref={messagesEndRef} />
                            </Box>
                            
                            <Box sx={{ p: 2, borderTop: '1px solid', borderColor: 'grey.800', bgcolor: 'rgba(5, 8, 18, 0.7)', backdropFilter: 'blur(10px)' }}>
                                <Paper component="form" onSubmit={(e) => { e.preventDefault(); handleSendMessage(); }} sx={{ 
                                    p: '8px 12px', 
                                    display: 'flex', 
                                    alignItems: 'center', 
                                    borderRadius: '16px', 
                                    bgcolor: 'rgba(255, 255, 255, 0.1)',
                                    border: '1px solid rgba(255, 255, 255, 0.2)',
                                    boxShadow: '0px 2px 10px rgba(0,0,0,0.5)'
                                }}>
                                    <TextField
                                        sx={{ ml: 1, flex: 1, '& .MuiInputBase-input::placeholder': { color: 'grey.400' } }}
                                        placeholder="Ask the agent about your documents..."
                                        fullWidth multiline maxRows={5}
                                        value={input}
                                        onChange={(e) => setInput(e.target.value)}
                                        onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSendMessage(); } }}
                                        disabled={isUploading || isSending || !activeConversationId}
                                        InputProps={{ disableUnderline: true }}
                                    />
                                    <IconButton type="submit" color="primary" disabled={isUploading || isSending || !input.trim() || !activeConversationId}><SendIcon /></IconButton>
                                </Paper>
                            </Box>
                        </>
                    ) : (
                        <DatabaseChat />
                    )}
                </Box>
            </Box>
            <AnimatePresence>
                {modalOpen && (
                    <Modal open={modalOpen} onClose={handleCloseModal} sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }} closeAfterTransition>
                        <motion.div
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.9 }}
                            transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                        >
                            <Paper sx={{ 
                                position: 'relative', 
                                bgcolor: 'rgba(10, 25, 41, 0.85)', 
                                backdropFilter: 'blur(12px)', 
                                p: 1, 
                                boxShadow: '0px 8px 40px rgba(0, 255, 255, 0.2)', 
                                outline: 'none', 
                                width: '90vw', 
                                height: '90vh', 
                                display: 'flex', 
                                flexDirection: 'column', 
                                border: '1px solid rgba(0, 255, 255, 0.3)',
                                borderRadius: 4,
                                overflow: 'hidden'
                            }}>
                                <IconButton onClick={handleCloseModal} sx={{ position: 'absolute', top: 12, right: 12, zIndex: 10, color: 'white', bgcolor: 'rgba(0,0,0,0.4)', '&:hover': {bgcolor: 'rgba(0,0,0,0.7)'} }}>
                                    <CloseIcon />
                                </IconButton>
                                <Box sx={{ flexGrow: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden', position: 'relative' }}>
                                    <AnimatePresence initial={false} custom={selectedImage.direction}>
                                        <motion.img
                                            key={selectedImage.index}
                                            src={modalImages[selectedImage.index]}
                                            custom={selectedImage.direction}
                                            variants={imageVariants}
                                            initial="enter"
                                            animate="center"
                                            exit="exit"
                                            transition={{
                                                x: { type: "spring", stiffness: 300, damping: 30 },
                                                opacity: { duration: 0.2 }
                                            }}
                                            style={{
                                                position: 'absolute',
                                                width: 'auto',
                                                height: 'auto',
                                                maxWidth: '100%',
                                                maxHeight: '100%',
                                                objectFit: 'contain',
                                                borderRadius: '8px',
                                            }}
                                        />
                                    </AnimatePresence>
                                </Box>
                                {modalImages.length > 1 && (
                                    <Box sx={{ position: 'absolute', top: '50%', left: 16, right: 16, display: 'flex', justifyContent: 'space-between', transform: 'translateY(-50%)', pointerEvents: 'none' }}>
                                        <IconButton onClick={handlePrevImage} sx={{bgcolor: 'rgba(0,0,0,0.3)', color: 'white', pointerEvents: 'auto', '&:hover': { bgcolor: 'rgba(0,0,0,0.6)'}}}>
                                            <ArrowBackIosNewIcon />
                                        </IconButton>
                                        <IconButton onClick={handleNextImage} sx={{bgcolor: 'rgba(0,0,0,0.3)', color: 'white', pointerEvents: 'auto', '&:hover': { bgcolor: 'rgba(0,0,0,0.6)'}}}>
                                            <ArrowForwardIosIcon />
                                        </IconButton>
                                    </Box>
                                )}
                                <Typography sx={{color: 'white', fontWeight: 'bold', textAlign: 'center', p:1, flexShrink: 0, textShadow: '0 1px 3px rgba(0,0,0,0.5)' }}>
                                    {selectedImage.index + 1} / {modalImages.length}
                                </Typography>
                            </Paper>
                        </motion.div>
                    </Modal>
                )}
            </AnimatePresence>
        </ThemeProvider>
  );
}

export default App;
