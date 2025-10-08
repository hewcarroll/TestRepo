/**
 * Authentication JavaScript for PDT Trading Bot Admin UI
 * 
 * Handles login, logout, token management, and user session operations.
 */

// Authentication utilities
class AuthManager {
    constructor() {
        this.token = localStorage.getItem('access_token');
        this.refreshToken = localStorage.getItem('refresh_token');
        this.apiBaseUrl = window.APP_CONFIG ? window.APP_CONFIG.apiBaseUrl : '/api';
    }

    // Login user with username and password
    async login(username, password) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/auth/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    username: username,
                    password: password
                })
            });

            const data = await response.json();

            if (response.ok) {
                // Store tokens
                this.token = data.access_token;
                this.refreshToken = data.refresh_token;
                
                localStorage.setItem('access_token', data.access_token);
                localStorage.setItem('refresh_token', data.refresh_token);
                
                // Update global config
                if (window.APP_CONFIG) {
                    window.APP_CONFIG.token = data.access_token;
                }
                
                return { success: true, data: data };
            } else {
                return { success: false, error: data.detail || 'Login failed' };
            }
        } catch (error) {
            return { success: false, error: 'Network error. Please try again.' };
        }
    }

    // Logout user and clear tokens
    async logout() {
        try {
            if (this.token) {
                await fetch(`${this.apiBaseUrl}/auth/logout`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${this.token}`
                    }
                });
            }
        } catch (error) {
            console.error('Logout error:', error);
        } finally {
            // Clear tokens regardless of API call result
            this.token = null;
            this.refreshToken = null;
            
            localStorage.removeItem('access_token');
            localStorage.removeItem('refresh_token');
            
            if (window.APP_CONFIG) {
                window.APP_CONFIG.token = null;
            }
        }
    }

    // Refresh access token
    async refreshAccessToken() {
        try {
            if (!this.refreshToken) {
                throw new Error('No refresh token available');
            }

            const response = await fetch(`${this.apiBaseUrl}/auth/refresh`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    refresh_token: this.refreshToken
                })
            });

            const data = await response.json();

            if (response.ok) {
                this.token = data.access_token;
                localStorage.setItem('access_token', data.access_token);
                
                if (window.APP_CONFIG) {
                    window.APP_CONFIG.token = data.access_token;
                }
                
                return { success: true, token: data.access_token };
            } else {
                throw new Error(data.detail || 'Token refresh failed');
            }
        } catch (error) {
            // Clear invalid tokens
            this.token = null;
            this.refreshToken = null;
            localStorage.removeItem('access_token');
            localStorage.removeItem('refresh_token');
            
            if (window.APP_CONFIG) {
                window.APP_CONFIG.token = null;
            }
            
            return { success: false, error: error.message };
        }
    }

    // Get current user information
    async getCurrentUser() {
        try {
            if (!this.token) {
                throw new Error('No access token available');
            }

            const response = await fetch(`${this.apiBaseUrl}/auth/me`, {
                headers: {
                    'Authorization': `Bearer ${this.token}`
                }
            });

            if (response.ok) {
                return await response.json();
            } else if (response.status === 401) {
                // Token might be expired, try to refresh
                const refreshResult = await this.refreshAccessToken();
                if (refreshResult.success) {
                    // Retry the original request
                    const retryResponse = await fetch(`${this.apiBaseUrl}/auth/me`, {
                        headers: {
                            'Authorization': `Bearer ${this.token}`
                        }
                    });
                    
                    if (retryResponse.ok) {
                        return await retryResponse.json();
                    }
                }
                
                throw new Error('Authentication required');
            } else {
                throw new Error('Failed to get user information');
            }
        } catch (error) {
            return null;
        }
    }

    // Check if user is authenticated
    isAuthenticated() {
        return !!(this.token && this.refreshToken);
    }

    // Get authorization header for API requests
    getAuthHeader() {
        return this.token ? `Bearer ${this.token}` : null;
    }

    // API request wrapper with automatic token refresh
    async apiRequest(url, options = {}) {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        };

        if (this.token) {
            defaultOptions.headers['Authorization'] = `Bearer ${this.token}`;
        }

        const mergedOptions = { ...defaultOptions, ...options };

        try {
            let response = await fetch(url, mergedOptions);

            // If unauthorized, try to refresh token and retry
            if (response.status === 401 && this.refreshToken) {
                const refreshResult = await this.refreshAccessToken();
                if (refreshResult.success) {
                    mergedOptions.headers['Authorization'] = `Bearer ${this.token}`;
                    response = await fetch(url, mergedOptions);
                }
            }

            return response;
        } catch (error) {
            throw error;
        }
    }
}

// Global authentication manager instance
const authManager = new AuthManager();

// Global functions for templates
async function handleLogin(event) {
    event.preventDefault();
    
    const username = document.getElementById('username').value.trim();
    const password = document.getElementById('password').value;
    
    if (!username || !password) {
        showAlert('Please enter both username and password', 'warning');
        return;
    }
    
    showLoading(true);
    
    const result = await authManager.login(username, password);
    
    if (result.success) {
        // Redirect to dashboard
        window.location.href = '/';
    } else {
        showAlert(result.error, 'danger');
        showLoading(false);
    }
}

async function logout() {
    await authManager.logout();
    window.location.href = '/login';
}

async function loadUserInfo() {
    const userInfo = await authManager.getCurrentUser();
    if (userInfo) {
        const usernameElement = document.getElementById('username');
        if (usernameElement) {
            usernameElement.textContent = userInfo.username;
        }
        
        // Update page title if needed
        const pageTitle = document.getElementById('page-title');
        if (pageTitle && !pageTitle.textContent) {
            pageTitle.textContent = `Welcome, ${userInfo.username}`;
        }
    }
}

// Token refresh utility for long-running sessions
setInterval(async () => {
    if (authManager.isAuthenticated()) {
        try {
            await authManager.refreshAccessToken();
        } catch (error) {
            console.warn('Token refresh failed:', error);
        }
    }
}, 15 * 60 * 1000); // Refresh every 15 minutes

// Export for use in other modules
window.authManager = authManager;
