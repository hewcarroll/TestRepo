/**
 * Dashboard JavaScript for PDT Trading Bot Admin UI
 * 
 * Handles WebSocket connections, real-time updates, data visualization,
 * and dashboard interactions.
 */

// Dashboard utilities
class DashboardManager {
    constructor() {
        this.wsManager = null;
        this.charts = {};
        this.autoRefreshInterval = null;
        this.apiBaseUrl = window.APP_CONFIG ? window.APP_CONFIG.apiBaseUrl : '/api';
        this.wsUrl = window.APP_CONFIG ? window.APP_CONFIG.wsUrl : null;
        this.isConnected = false;
    }

    // Initialize WebSocket connection
    async initializeWebSocket() {
        if (!this.wsUrl) {
            console.warn('WebSocket URL not configured');
            return;
        }

        try {
            // Create WebSocket manager if not exists
            if (!this.wsManager) {
                this.wsManager = new WebSocketManager(this.wsUrl);
            }

            // Connect to WebSocket
            await this.wsManager.connect();
            this.isConnected = true;

            // Set up event listeners
            this.setupWebSocketListeners();

            // Subscribe to real-time updates
            await this.subscribeToUpdates();

            console.log('WebSocket connected successfully');
        } catch (error) {
            console.error('WebSocket connection failed:', error);
            this.updateConnectionStatus(false);
        }
    }

    // Set up WebSocket event listeners
    setupWebSocketListeners() {
        if (!this.wsManager) return;

        this.wsManager.on('trading_data', (data) => {
            this.updateTradingData(data);
        });

        this.wsManager.on('pnl_update', (data) => {
            this.updatePnLData(data);
        });

        this.wsManager.on('position_update', (data) => {
            this.updatePositionsData(data);
        });

        this.wsManager.on('bot_status', (data) => {
            this.updateBotStatus(data);
        });

        this.wsManager.on('alert', (data) => {
            this.handleAlert(data);
        });

        this.wsManager.on('disconnect', () => {
            this.isConnected = false;
            this.updateConnectionStatus(false);
        });

        this.wsManager.on('reconnect', () => {
            this.isConnected = true;
            this.updateConnectionStatus(true);
        });
    }

    // Subscribe to real-time updates
    async subscribeToUpdates() {
        if (!this.wsManager) return;

        try {
            await this.wsManager.subscribe('trading_data');
            await this.wsManager.subscribe('pnl_update');
            await this.wsManager.subscribe('position_update');
            await this.wsManager.subscribe('bot_status');
            await this.wsManager.subscribe('alert');
        } catch (error) {
            console.error('Failed to subscribe to updates:', error);
        }
    }

    // Update connection status indicator
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('ws-status');
        if (statusElement) {
            if (connected) {
                statusElement.className = 'badge bg-success';
                statusElement.innerHTML = '<i class="fas fa-circle me-1"></i>Connected';
            } else {
                statusElement.className = 'badge bg-danger';
                statusElement.innerHTML = '<i class="fas fa-circle me-1"></i>Disconnected';
            }
        }
    }

    // Load dashboard data
    async loadDashboardData() {
        try {
            const response = await authManager.apiRequest(`${this.apiBaseUrl}/data/dashboard-summary`);
            const data = await response.json();

            if (response.ok) {
                this.updateDashboardData(data);
            } else {
                throw new Error(data.detail || 'Failed to load dashboard data');
            }
        } catch (error) {
            console.error('Error loading dashboard data:', error);
            showAlert('Failed to load dashboard data: ' + error.message, 'danger');
        }
    }

    // Update all dashboard data
    updateDashboardData(data) {
        this.updateOverviewCards(data);
        this.updatePositionsTable(data.positions || []);
        this.updateTradesTable(data.recent_trades || []);
        this.updateMarketDataTable(data.market_data || {});
        this.updatePDTStatus(data.pdt_status || {});
        this.updateBotStatus(data.bot_status || {});
        this.updateRiskMetrics(data.risk_metrics || {});
    }

    // Update overview cards
    updateOverviewCards(data) {
        const pnlData = data.pnl_data || {};
        
        // Daily P&L
        const dailyPnlElement = document.getElementById('daily-pnl');
        if (dailyPnlElement) {
            dailyPnlElement.textContent = formatCurrency(pnlData.daily_pnl || 0);
            dailyPnlElement.className = (pnlData.daily_pnl || 0) >= 0 ? 'text-success' : 'text-danger';
        }

        // Total P&L
        const totalPnlElement = document.getElementById('total-pnl');
        if (totalPnlElement) {
            totalPnlElement.textContent = formatCurrency(pnlData.total_pnl || 0);
            totalPnlElement.className = (pnlData.total_pnl || 0) >= 0 ? 'text-success' : 'text-danger';
        }

        // Daily return percentage
        const dailyReturnElement = document.getElementById('daily-return');
        if (dailyReturnElement) {
            const returnPercent = ((pnlData.daily_pnl || 0) / 10000 * 100).toFixed(2);
            dailyReturnElement.textContent = formatPercentage(returnPercent / 100);
        }

        // Total return percentage
        const totalReturnElement = document.getElementById('total-return');
        if (totalReturnElement) {
            const returnPercent = pnlData.total_return || 0;
            totalReturnElement.textContent = formatPercentage(returnPercent);
        }

        // Active positions
        const positions = data.positions || [];
        const activePositionsElement = document.getElementById('active-positions');
        if (activePositionsElement) {
            activePositionsElement.textContent = positions.length;
        }

        // Positions value
        const positionsValueElement = document.getElementById('positions-value');
        if (positionsValueElement) {
            const totalValue = positions.reduce((sum, pos) => sum + (pos.market_value || 0), 0);
            positionsValueElement.textContent = formatCurrency(totalValue);
        }

        // Win rate
        const winRateElement = document.getElementById('win-rate');
        if (winRateElement) {
            const winRate = pnlData.win_rate || 0;
            winRateElement.textContent = formatPercentage(winRate);
        }

        // Sharpe ratio
        const sharpeElement = document.getElementById('sharpe-ratio');
        if (sharpeElement) {
            sharpeElement.textContent = `Sharpe: ${pnlData.sharpe_ratio || 0}`;
        }
    }

    // Update positions table
    updatePositionsTable(positions) {
        const tableBody = document.getElementById('positions-table');
        if (!tableBody) return;

        if (positions.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">No positions</td></tr>';
            return;
        }

        tableBody.innerHTML = positions.map(position => `
            <tr>
                <td>${position.symbol}</td>
                <td>${formatNumber(position.quantity)}</td>
                <td>${formatCurrency(position.current_price)}</td>
                <td class="${(position.unrealized_pnl || 0) >= 0 ? 'text-success' : 'text-danger'}">
                    ${formatCurrency(position.unrealized_pnl || 0)}
                </td>
            </tr>
        `).join('');
    }

    // Update trades table
    updateTradesTable(trades) {
        const tableBody = document.getElementById('trades-table');
        if (!tableBody) return;

        if (trades.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">No recent trades</td></tr>';
            return;
        }

        tableBody.innerHTML = trades.map(trade => `
            <tr>
                <td>${trade.symbol}</td>
                <td><span class="badge bg-${trade.side === 'buy' ? 'success' : 'danger'}">${trade.side.toUpperCase()}</span></td>
                <td>${formatNumber(trade.quantity)}</td>
                <td class="${(trade.pnl || 0) >= 0 ? 'text-success' : 'text-danger'}">
                    ${formatCurrency(trade.pnl || 0)}
                </td>
            </tr>
        `).join('');
    }

    // Update market data table
    updateMarketDataTable(marketData) {
        const tableBody = document.getElementById('market-data-table');
        if (!tableBody) return;

        const symbols = Object.keys(marketData);
        if (symbols.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No market data</td></tr>';
            return;
        }

        tableBody.innerHTML = symbols.map(symbol => {
            const data = marketData[symbol];
            return `
                <tr>
                    <td><strong>${symbol}</strong></td>
                    <td>${formatCurrency(data.price)}</td>
                    <td class="text-muted">+0.00%</td>
                    <td>${formatNumber(data.volume)}</td>
                    <td>${formatCurrency(data.bid)}</td>
                    <td>${formatCurrency(data.ask)}</td>
                </tr>
            `;
        }).join('');

        // Update last update time
        const lastUpdateElement = document.getElementById('market-last-update');
        if (lastUpdateElement) {
            lastUpdateElement.textContent = new Date().toLocaleTimeString();
        }
    }

    // Update PDT status
    updatePDTStatus(pdtData) {
        // PDT progress bar
        const progressBar = document.getElementById('pdt-progress-bar');
        if (progressBar) {
            const progress = Math.min(100, ((pdtData.volume_towards_threshold || 0) / 25000) * 100);
            progressBar.style.width = progress + '%';
            progressBar.setAttribute('aria-valuenow', progress);
        }

        // Day trades used
        const dayTradesElement = document.getElementById('day-trades-used');
        if (dayTradesElement) {
            dayTradesElement.textContent = pdtData.day_trades_used || 0;
        }

        // Volume progress
        const volumeElement = document.getElementById('volume-progress');
        if (volumeElement) {
            volumeElement.textContent = formatCurrency(pdtData.volume_towards_threshold || 0);
        }

        // PDT status badge
        const statusElement = document.getElementById('pdt-status');
        if (statusElement) {
            const status = pdtData.status || 'unknown';
            statusElement.className = `badge bg-${status === 'compliant' ? 'success' : status === 'warning' ? 'warning' : 'danger'}`;
            statusElement.textContent = status.charAt(0).toUpperCase() + status.slice(1);
        }
    }

    // Update bot status
    updateBotStatus(botData) {
        const statusElement = document.getElementById('bot-status');
        if (statusElement) {
            const status = botData.status || 'unknown';
            statusElement.className = `badge bg-${status === 'active' ? 'success' : status === 'paused' ? 'warning' : 'danger'}`;
            statusElement.innerHTML = `<i class="fas fa-robot me-1"></i>${status.charAt(0).toUpperCase() + status.slice(1)}`;
        }

        // Update control buttons
        this.updateBotControlButtons(botData);
    }

    // Update bot control buttons
    updateBotControlButtons(botData) {
        const startBtn = document.getElementById('start-btn');
        const pauseBtn = document.getElementById('pause-btn');
        const stopBtn = document.getElementById('stop-btn');

        if (!startBtn || !pauseBtn || !stopBtn) return;

        const isRunning = botData.is_running || false;
        const status = botData.status || 'stopped';

        // Enable/disable buttons based on current state
        if (status === 'stopped') {
            startBtn.disabled = false;
            pauseBtn.disabled = true;
            stopBtn.disabled = true;
        } else if (status === 'active') {
            startBtn.disabled = true;
            pauseBtn.disabled = false;
            stopBtn.disabled = false;
        } else if (status === 'paused') {
            startBtn.disabled = true;
            pauseBtn.disabled = true;
            stopBtn.disabled = false;
        }
    }

    // Update risk metrics
    updateRiskMetrics(riskData) {
        // VaR 95%
        const var95Element = document.getElementById('var-95');
        if (var95Element) {
            var95Element.textContent = formatCurrency(riskData.var_95 || 0);
        }

        // Volatility
        const volatilityElement = document.getElementById('volatility');
        if (volatilityElement) {
            volatilityElement.textContent = formatPercentage(riskData.volatility || 0);
        }

        // Max drawdown
        const maxDrawdownElement = document.getElementById('max-drawdown');
        if (maxDrawdownElement) {
            maxDrawdownElement.textContent = formatPercentage(Math.abs(riskData.max_drawdown || 0));
        }

        // Leverage
        const leverageElement = document.getElementById('leverage');
        if (leverageElement) {
            leverageElement.textContent = `${riskData.leverage || 0}x`;
        }
    }

    // Handle real-time trading data updates
    updateTradingData(data) {
        // Update positions and P&L in real-time
        if (data.positions) {
            this.updatePositionsTable(data.positions);
        }
        
        if (data.pnl) {
            this.updatePnLData(data.pnl);
        }
    }

    // Update P&L data
    updatePnLData(pnlData) {
        const dailyPnlElement = document.getElementById('daily-pnl');
        if (dailyPnlElement) {
            dailyPnlElement.textContent = formatCurrency(pnlData.daily_pnl || 0);
        }

        const totalPnlElement = document.getElementById('total-pnl');
        if (totalPnlElement) {
            totalPnlElement.textContent = formatCurrency(pnlData.total_pnl || 0);
        }
    }

    // Update positions data
    updatePositionsData(positionData) {
        if (positionData.positions) {
            this.updatePositionsTable(positionData.positions);
        }
    }

    // Handle alerts
    handleAlert(alertData) {
        const { alert_type, data } = alertData;
        showAlert(`${alert_type}: ${data.message || 'Alert received'}`, 
                 data.severity || 'info');
    }

    // Control bot operations
    async controlBot(action) {
        try {
            showLoading(true);
            
            const response = await authManager.apiRequest(
                `${this.apiBaseUrl}/data/bot/control?action=${action}`,
                { method: 'POST' }
            );

            const result = await response.json();

            if (response.ok) {
                showAlert(`Bot ${action} command executed successfully`, 'success');
                // Refresh data to show updated status
                setTimeout(() => this.loadDashboardData(), 1000);
            } else {
                throw new Error(result.detail || `Failed to ${action} bot`);
            }
        } catch (error) {
            showAlert(`Failed to ${action} bot: ${error.message}`, 'danger');
        } finally {
            hideLoading();
        }
    }

    // Initialize charts
    initializeCharts() {
        this.createPnLChart();
    }

    // Create P&L performance chart
    createPnLChart() {
        const ctx = document.getElementById('pnlChart');
        if (!ctx) return;

        if (this.charts.pnl) {
            this.charts.pnl.destroy();
        }

        this.charts.pnl = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Cumulative P&L',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return formatCurrency(value);
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `P&L: ${formatCurrency(context.parsed.y)}`;
                            }
                        }
                    }
                }
            }
        });
    }

    // Update P&L chart with new data
    async updatePnLChart() {
        try {
            const response = await authManager.apiRequest(
                `${this.apiBaseUrl}/data/performance-history?days=30`
            );
            const data = await response.json();

            if (response.ok && this.charts.pnl) {
                const chartData = data.performance_data || [];
                
                this.charts.pnl.data.labels = chartData.map(d => 
                    new Date(d.date).toLocaleDateString()
                );
                this.charts.pnl.data.datasets[0].data = chartData.map(d => d.cumulative_pnl);
                
                this.charts.pnl.update();
            }
        } catch (error) {
            console.error('Error updating P&L chart:', error);
        }
    }

    // Start auto-refresh
    startAutoRefresh() {
        if (this.autoRefreshInterval) {
            clearInterval(this.autoRefreshInterval);
        }

        const interval = (window.APP_CONFIG && window.APP_CONFIG.refreshInterval) || 30000;
        
        this.autoRefreshInterval = setInterval(() => {
            this.loadDashboardData();
        }, interval);
    }

    // Stop auto-refresh
    stopAutoRefresh() {
        if (this.autoRefreshInterval) {
            clearInterval(this.autoRefreshInterval);
            this.autoRefreshInterval = null;
        }
    }
}

// WebSocket Manager class
class WebSocketManager {
    constructor(wsUrl) {
        this.wsUrl = wsUrl;
        this.socket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
        this.eventListeners = {};
        this.isConnecting = false;
    }

    async connect() {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            return;
        }

        if (this.isConnecting) {
            return;
        }

        this.isConnecting = true;

        try {
            // Add authentication token to WebSocket URL
            const token = authManager ? authManager.token : null;
            const wsUrl = token ? `${this.wsUrl}?token=${token}` : this.wsUrl;

            this.socket = new WebSocket(wsUrl);
            
            return new Promise((resolve, reject) => {
                this.socket.onopen = () => {
                    console.log('WebSocket connected');
                    this.reconnectAttempts = 0;
                    this.isConnecting = false;
                    this.emit('connect');
                    resolve();
                };

                this.socket.onmessage = (event) => {
                    try {
                        const message = JSON.parse(event.data);
                        this.emit(message.type, message.data);
                    } catch (error) {
                        console.error('Error parsing WebSocket message:', error);
                    }
                };

                this.socket.onclose = () => {
                    console.log('WebSocket disconnected');
                    this.isConnecting = false;
                    this.emit('disconnect');
                    
                    // Attempt to reconnect
                    if (this.reconnectAttempts < this.maxReconnectAttempts) {
                        setTimeout(() => {
                            this.reconnectAttempts++;
                            this.connect();
                        }, this.reconnectDelay * this.reconnectAttempts);
                    }
                };

                this.socket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.isConnecting = false;
                    this.emit('error', error);
                    reject(error);
                };

                // Timeout after 10 seconds
                setTimeout(() => {
                    if (this.socket.readyState !== WebSocket.OPEN) {
                        reject(new Error('WebSocket connection timeout'));
                    }
                }, 10000);
            });
        } catch (error) {
            this.isConnecting = false;
            throw error;
        }
    }

    disconnect() {
        if (this.socket) {
            this.socket.close();
            this.socket = null;
        }
    }

    async subscribe(subscriptionType) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            const message = {
                type: 'subscribe',
                data: { subscription_type: subscriptionType }
            };
            this.socket.send(JSON.stringify(message));
        }
    }

    async unsubscribe(subscriptionType) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            const message = {
                type: 'unsubscribe',
                data: { subscription_type: subscriptionType }
            };
            this.socket.send(JSON.stringify(message));
        }
    }

    on(event, callback) {
        if (!this.eventListeners[event]) {
            this.eventListeners[event] = [];
        }
        this.eventListeners[event].push(callback);
    }

    emit(event, data) {
        if (this.eventListeners[event]) {
            this.eventListeners[event].forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in event listener for ${event}:`, error);
                }
            });
        }
    }
}

// Global dashboard manager instance
const dashboardManager = new DashboardManager();

// Global functions for templates
async function initializeWebSocket() {
    await dashboardManager.initializeWebSocket();
}

async function loadDashboardData() {
    await dashboardManager.loadDashboardData();
}

async function controlBot(action) {
    await dashboardManager.controlBot(action);
}

async function pauseBot() {
    await dashboardManager.controlBot('pause');
}

async function stopBot() {
    await dashboardManager.controlBot('stop');
}

async function activateKillSwitch() {
    // Show confirmation dialog
    if (confirm('Are you sure you want to activate the emergency kill switch? This will immediately stop all trading.')) {
        try {
            showLoading(true);
            
            const response = await authManager.apiRequest(
                `${dashboardManager.apiBaseUrl}/dashboard/emergency/kill-switch`,
                { method: 'POST' }
            );

            const result = await response.json();

            if (response.ok) {
                showAlert('Emergency kill switch activated successfully', 'success');
                // Close modal and refresh data
                const modal = bootstrap.Modal.getInstance(document.getElementById('emergencyModal'));
                if (modal) modal.hide();
                setTimeout(() => dashboardManager.loadDashboardData(), 1000);
            } else {
                throw new Error(result.detail || 'Failed to activate kill switch');
            }
        } catch (error) {
            showAlert(`Failed to activate kill switch: ${error.message}`, 'danger');
        } finally {
            hideLoading();
        }
    }
}

async function toggleCircuitBreaker(breakerType, enabled) {
    try {
        const response = await authManager.apiRequest(
            `${dashboardManager.apiBaseUrl}/dashboard/emergency/circuit-breaker?breaker_type=${breakerType}&enabled=${enabled}`,
            { method: 'POST' }
        );

        const result = await response.json();

        if (response.ok) {
            showAlert(`Circuit breaker ${enabled ? 'enabled' : 'disabled'} successfully`, 'success');
        } else {
            throw new Error(result.detail || 'Failed to toggle circuit breaker');
        }
    } catch (error) {
        showAlert(`Failed to toggle circuit breaker: ${error.message}`, 'danger');
    }
}

async function refreshStrategies() {
    try {
        showLoading(true);
        
        const response = await authManager.apiRequest(
            `${dashboardManager.apiBaseUrl}/dashboard/strategies`
        );
        const strategies = await response.json();

        if (response.ok) {
            // Update strategies list in UI
            const strategiesList = document.getElementById('strategies-list');
            if (strategiesList) {
                strategiesList.innerHTML = strategies.map(strategy => 
                    `<div class="strategy-item">
                        <strong>${strategy.strategy_name}</strong>
                        <span class="badge bg-${strategy.is_active ? 'success' : 'secondary'} ms-2">
                            ${strategy.is_active ? 'Active' : 'Inactive'}
                        </span>
                    </div>`
                ).join('');
            }
            
            showAlert('Strategies refreshed successfully', 'success');
        } else {
            throw new Error('Failed to refresh strategies');
        }
    } catch (error) {
        showAlert(`Failed to refresh strategies: ${error.message}`, 'danger');
    } finally {
        hideLoading();
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', async function() {
    // Initialize charts
    dashboardManager.initializeCharts();
    
    // Load initial data
    await loadDashboardData();
    
    // Update P&L chart
    await dashboardManager.updatePnLChart();
    
    // Start auto-refresh
    dashboardManager.startAutoRefresh();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        dashboardManager.stopAutoRefresh();
    } else {
        dashboardManager.startAutoRefresh();
        loadDashboardData(); // Refresh data when page becomes visible
    }
});

// Export for use in other modules
window.dashboardManager = dashboardManager;
