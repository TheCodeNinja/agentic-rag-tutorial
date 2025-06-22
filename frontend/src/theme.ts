import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00e5ff', // A vibrant cyan/aqua
    },
    secondary: {
      main: '#ff00ff', // Magenta for contrast
    },
    background: {
      default: '#050812', // A very dark blue, almost black
      paper: 'rgba(23, 28, 48, 0.7)', // A semi-transparent dark blue for surfaces
    },
    text: {
      primary: '#e0e0e0',
      secondary: '#7F8C8D', // Muted gray for secondary text
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
      letterSpacing: '0.1em',
      textShadow: '0 0 8px #00e5ff, 0 0 16px #00e5ff, 0 0 24px #00e5ff',
    },
    h2: {
        fontSize: '1.75rem',
        fontWeight: 700,
        letterSpacing: '0.05em',
        color: '#00e5ff'
    },
    body1: {
      fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
      fontSize: '1rem',
    }
  },
  components: {
    MuiToolbar: {
        styleOverrides: {
          root: {
            minHeight: '48px',
            '@media (min-width: 600px)': {
              minHeight: '48px',
            },
          },
        },
      },
    MuiAppBar: {
        styleOverrides: {
            root: {
                backgroundColor: 'transparent',
                boxShadow: 'none',
                borderBottom: '1px solid rgba(0, 229, 255, 0.2)',
            }
        }
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          border: '1px solid rgba(0, 229, 255, 0.1)',
          backdropFilter: 'blur(10px)',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderWidth: '1px',
          borderStyle: 'solid',
          borderColor: 'rgba(0, 229, 255, 0.5)',
          color: '#00e5ff',
          '&:hover': {
            backgroundColor: 'rgba(0, 229, 255, 0.1)',
            borderColor: '#00e5ff',
            boxShadow: '0 0 10px #00e5ff80',
          },
        },
      },
    },
    MuiOutlinedInput: {
        styleOverrides: {
            root: {
                backgroundColor: 'rgba(10, 15, 30, 0.7)',
                '& fieldset': {
                    borderColor: 'rgba(0, 229, 255, 0.3)',
                },
                '&:hover fieldset': {
                    borderColor: 'rgba(0, 229, 255, 0.7)',
                },
                '&.Mui-focused fieldset': {
                    borderColor: '#00e5ff',
                    boxShadow: '0 0 10px #00e5ff80',
                },
            },
        },
    },
    MuiSvgIcon: {
        styleOverrides: {
            root: {
                color: 'rgba(0, 229, 255, 0.7)',
            }
        }
    }
  },
});

export default theme; 