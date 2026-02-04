/**
 * Anti-Scam Sentinel Dashboard - Client Application
 * Handles API interactions, chat simulation, and analytics display
 */

class AntiScamClient {
    constructor() {
        this.baseUrl = localStorage.getItem('api_url') || 'http://localhost:8000';
        this.apiKey = localStorage.getItem('api_key') || '';
        this.sessionId = this.generateSessionId();
        this.autoScroll = true;
    }

    generateSessionId() {
        return 'session-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    }

    getHeaders() {
        const headers = {
            'Content-Type': 'application/json'
        };
        if (this.apiKey) {
            headers['X-API-Key'] = this.apiKey;
        }
        return headers;
    }

    async sendMessage(message) {
        const response = await fetch(`${this.baseUrl}/message`, {
            method: 'POST',
            headers: this.getHeaders(),
            body: JSON.stringify({
                session_id: this.sessionId,
                message: message,
                timestamp: new Date().toISOString()
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        return await response.json();
    }

    async healthCheck() {
        const response = await fetch(`${this.baseUrl}/health`, {
            headers: this.getHeaders()
        });
        return await response.json();
    }

    async getSession(sessionId) {
        const response = await fetch(`${this.baseUrl}/session/${sessionId}`, {
            headers: this.getHeaders()
        });
        return await response.json();
    }

    async getAnalytics(sessionId) {
        const response = await fetch(`${this.baseUrl}/analytics/${sessionId}`, {
            headers: this.getHeaders()
        });
        return await response.json();
    }

    async getMetrics() {
        const response = await fetch(`${this.baseUrl}/metrics`, {
            headers: this.getHeaders()
        });
        return await response.json();
    }

    async testEndpoint(url, apiKey) {
        const testMessage = "Your SBI account will be blocked within 24 hours due to KYC update pending.";
        const headers = {
            'Content-Type': 'application/json'
        };
        if (apiKey) {
            headers['X-API-Key'] = apiKey;
        }

        const startTime = performance.now();
        
        const response = await fetch(`${url}/message`, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify({
                session_id: 'test-' + Date.now(),
                message: testMessage,
                timestamp: new Date().toISOString()
            })
        });

        const endTime = performance.now();
        const latency = Math.round(endTime - startTime);

        const data = await response.json();
        
        return {
            success: response.ok,
            status: response.status,
            latency: latency,
            data: data
        };
    }
}

// Dashboard Application
class Dashboard {
    constructor() {
        this.client = new AntiScamClient();
        this.isProcessing = false;
        this.init();
    }

    init() {
        this.bindElements();
        this.bindEvents();
        this.checkConnection();
        this.updateSessionDisplay();
    }

    bindElements() {
        // Chat elements
        this.chatContainer = document.getElementById('chat-container');
        this.messageInput = document.getElementById('message-input');
        this.sendBtn = document.getElementById('send-btn');
        this.newSessionBtn = document.getElementById('new-session-btn');
        this.sessionDisplay = document.getElementById('session-display');

        // Status elements
        this.statusDot = document.getElementById('status-dot');
        this.statusText = document.getElementById('status-text');

        // Detection elements
        this.detectionStatus = document.getElementById('detection-status');
        this.detectionDetails = document.getElementById('detection-details');
        this.scamType = document.getElementById('scam-type');
        this.confidenceBar = document.getElementById('confidence-bar');
        this.confidenceValue = document.getElementById('confidence-value');
        this.threatLevel = document.getElementById('threat-level');
        this.personaUsed = document.getElementById('persona-used');

        // Intelligence elements
        this.intelScore = document.getElementById('intel-score');
        this.upiCount = document.getElementById('upi-count');
        this.bankCount = document.getElementById('bank-count');
        this.phoneCount = document.getElementById('phone-count');
        this.urlCount = document.getElementById('url-count');
        this.intelList = document.getElementById('intel-list');

        // Metrics elements
        this.metricTurns = document.getElementById('metric-turns');
        this.metricLatency = document.getElementById('metric-latency');
        this.metricPhase = document.getElementById('metric-phase');

        // Tester elements
        this.apiUrl = document.getElementById('api-url');
        this.apiKeyInput = document.getElementById('api-key');
        this.testEndpointBtn = document.getElementById('test-endpoint-btn');
        this.testResult = document.getElementById('test-result');
        this.resultStatus = document.getElementById('result-status');
        this.resultTime = document.getElementById('result-time');
        this.resultBody = document.getElementById('result-body');

        // Settings elements
        this.settingsBtn = document.getElementById('settings-btn');
        this.settingsModal = document.getElementById('settings-modal');
        this.closeSettings = document.getElementById('close-settings');
        this.settingsApiUrl = document.getElementById('settings-api-url');
        this.settingsApiKey = document.getElementById('settings-api-key');
        this.saveSettings = document.getElementById('save-settings');

        // Quick buttons
        this.quickBtns = document.querySelectorAll('.quick-btn');

        // Toast container
        this.toastContainer = document.getElementById('toast-container');
    }

    bindEvents() {
        // Send message
        this.sendBtn.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // New session
        this.newSessionBtn.addEventListener('click', () => this.newSession());

        // Quick buttons
        this.quickBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                this.messageInput.value = btn.dataset.message;
                this.sendMessage();
            });
        });

        // Test endpoint
        this.testEndpointBtn.addEventListener('click', () => this.testEndpoint());

        // Settings
        this.settingsBtn.addEventListener('click', () => this.openSettings());
        this.closeSettings.addEventListener('click', () => this.closeSettingsModal());
        this.saveSettings.addEventListener('click', () => this.saveSettingsHandler());
        this.settingsModal.addEventListener('click', (e) => {
            if (e.target === this.settingsModal) this.closeSettingsModal();
        });

        // Load saved settings
        this.loadSettings();
    }

    async checkConnection() {
        try {
            const health = await this.client.healthCheck();
            this.statusDot.classList.add('connected');
            this.statusText.textContent = `Connected (${health.version})`;
            this.showToast('Connected to API', 'success');
        } catch (error) {
            this.statusDot.classList.add('error');
            this.statusText.textContent = 'Disconnected';
            this.showToast('Could not connect to API', 'error');
        }
    }

    updateSessionDisplay() {
        this.sessionDisplay.textContent = `Session: ${this.client.sessionId.substr(0, 20)}...`;
    }

    newSession() {
        this.client.sessionId = this.client.generateSessionId();
        this.updateSessionDisplay();
        this.clearChat();
        this.resetAnalytics();
        this.showToast('New session started', 'info');
    }

    clearChat() {
        this.chatContainer.innerHTML = `
            <div class="chat-welcome">
                <div class="welcome-icon">🎭</div>
                <h3>Start a Scam Simulation</h3>
                <p>Type a scam message to see how the honeypot agent responds.<br>
                The agent will detect scams and extract intelligence.</p>
            </div>
        `;
    }

    resetAnalytics() {
        this.detectionStatus.innerHTML = '<span class="status-icon">⏳</span><span>Awaiting message...</span>';
        this.detectionStatus.className = 'detection-status';
        this.detectionDetails.style.display = 'none';
        this.intelScore.textContent = '0/100';
        this.upiCount.textContent = '0';
        this.bankCount.textContent = '0';
        this.phoneCount.textContent = '0';
        this.urlCount.textContent = '0';
        this.intelList.innerHTML = '<p class="empty-state">No intelligence extracted yet</p>';
        this.metricTurns.textContent = '0';
        this.metricLatency.textContent = '--';
        this.metricPhase.textContent = '--';
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isProcessing) return;

        this.isProcessing = true;
        this.sendBtn.disabled = true;
        this.messageInput.value = '';

        // Remove welcome message if present
        const welcome = this.chatContainer.querySelector('.chat-welcome');
        if (welcome) welcome.remove();

        // Add user message
        this.addMessage('scammer', message);

        // Show typing indicator
        this.showTypingIndicator();

        try {
            const response = await this.client.sendMessage(message);
            
            // Remove typing indicator
            this.hideTypingIndicator();

            // Add agent response
            this.addMessage('agent', response.agent_response);

            // Update analytics
            this.updateDetection(response);
            this.updateIntelligence(response.extracted_entities);
            this.updateMetrics(response.metadata);

        } catch (error) {
            this.hideTypingIndicator();
            this.showToast(`Error: ${error.message}`, 'error');
            this.addMessage('agent', '⚠️ Error processing message. Check API connection.');
        }

        this.isProcessing = false;
        this.sendBtn.disabled = false;
    }

    addMessage(role, content) {
        const messageEl = document.createElement('div');
        messageEl.className = `message ${role}`;
        
        const roleLabel = role === 'scammer' ? '🎭 Scammer' : '🛡️ Agent';
        const time = new Date().toLocaleTimeString();

        messageEl.innerHTML = `
            <div class="message-header">
                <span>${roleLabel}</span>
                <span>${time}</span>
            </div>
            <div class="message-content">${this.escapeHtml(content)}</div>
        `;

        this.chatContainer.appendChild(messageEl);
        
        if (this.client.autoScroll) {
            messageEl.scrollIntoView({ behavior: 'smooth' });
        }
    }

    showTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.id = 'typing-indicator';
        indicator.innerHTML = '<span></span><span></span><span></span>';
        this.chatContainer.appendChild(indicator);
        indicator.scrollIntoView({ behavior: 'smooth' });
    }

    hideTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) indicator.remove();
    }

    updateDetection(response) {
        const isScam = response.is_scam;
        const forensics = response.forensics;
        const confidence = Math.round(response.confidence_score * 100);

        // Update status
        if (isScam) {
            this.detectionStatus.innerHTML = `
                <span class="status-icon">🚨</span>
                <span><strong>SCAM DETECTED</strong></span>
            `;
            this.detectionStatus.className = 'detection-status scam';
        } else {
            this.detectionStatus.innerHTML = `
                <span class="status-icon">✅</span>
                <span>No scam detected</span>
            `;
            this.detectionStatus.className = 'detection-status safe';
        }

        // Show details
        this.detectionDetails.style.display = 'flex';
        this.scamType.textContent = forensics.scam_type || 'unknown';
        this.confidenceBar.style.width = `${confidence}%`;
        this.confidenceValue.textContent = `${confidence}%`;
        
        this.threatLevel.textContent = forensics.threat_level || 'low';
        this.threatLevel.className = `threat-badge ${forensics.threat_level || 'low'}`;
        
        this.personaUsed.textContent = forensics.persona_used || 'default';
    }

    updateIntelligence(entities) {
        if (!entities) return;

        const upiIds = entities.upi_ids || [];
        const bankAccounts = entities.bank_accounts || [];
        const phones = entities.phone_numbers || [];
        const urls = entities.urls || [];
        const score = Math.round(entities.intel_completeness_score || 0);

        this.intelScore.textContent = `${score}/100`;
        this.upiCount.textContent = upiIds.length;
        this.bankCount.textContent = bankAccounts.length;
        this.phoneCount.textContent = phones.length;
        this.urlCount.textContent = urls.length;

        // Build intel list
        let listHtml = '';

        upiIds.forEach(upi => {
            const upiId = typeof upi === 'string' ? upi : upi.upi_id;
            listHtml += `<div class="intel-entry">💳 UPI: ${this.escapeHtml(upiId)}</div>`;
        });

        bankAccounts.forEach(acc => {
            const accNum = typeof acc === 'string' ? acc : acc.account_number;
            listHtml += `<div class="intel-entry">🏦 Bank: ${this.escapeHtml(accNum)}</div>`;
        });

        phones.forEach(phone => {
            listHtml += `<div class="intel-entry">📱 Phone: ${this.escapeHtml(phone)}</div>`;
        });

        urls.forEach(url => {
            listHtml += `<div class="intel-entry">🔗 URL: ${this.escapeHtml(url)}</div>`;
        });

        this.intelList.innerHTML = listHtml || '<p class="empty-state">No intelligence extracted yet</p>';
    }

    updateMetrics(metadata) {
        if (!metadata) return;

        this.metricTurns.textContent = metadata.turn_count || 0;
        this.metricLatency.textContent = metadata.latency_ms ? `${metadata.latency_ms}ms` : '--';
        this.metricPhase.textContent = metadata.phase || '--';
    }

    async testEndpoint() {
        const url = this.apiUrl.value.trim();
        const apiKey = this.apiKeyInput.value.trim();

        if (!url) {
            this.showToast('Please enter an endpoint URL', 'warning');
            return;
        }

        this.testEndpointBtn.disabled = true;
        this.testEndpointBtn.textContent = '⏳ Testing...';

        try {
            const result = await this.client.testEndpoint(url, apiKey);
            
            this.testResult.style.display = 'block';
            this.resultStatus.textContent = result.success ? '✅ Success' : '❌ Failed';
            this.resultStatus.className = `result-status ${result.success ? 'success' : 'error'}`;
            this.resultTime.textContent = `${result.latency}ms`;
            this.resultBody.textContent = JSON.stringify(result.data, null, 2);

            this.showToast(
                result.success ? 'Endpoint test successful!' : `Endpoint test failed: ${result.status}`,
                result.success ? 'success' : 'error'
            );

        } catch (error) {
            this.testResult.style.display = 'block';
            this.resultStatus.textContent = '❌ Error';
            this.resultStatus.className = 'result-status error';
            this.resultTime.textContent = '--';
            this.resultBody.textContent = error.message;

            this.showToast(`Connection error: ${error.message}`, 'error');
        }

        this.testEndpointBtn.disabled = false;
        this.testEndpointBtn.textContent = '🔌 Test Honeypot Endpoint';
    }

    openSettings() {
        this.settingsApiUrl.value = this.client.baseUrl;
        this.settingsApiKey.value = this.client.apiKey;
        this.settingsModal.classList.add('active');
    }

    closeSettingsModal() {
        this.settingsModal.classList.remove('active');
    }

    saveSettingsHandler() {
        const url = this.settingsApiUrl.value.trim();
        const apiKey = this.settingsApiKey.value.trim();

        if (url) {
            this.client.baseUrl = url;
            localStorage.setItem('api_url', url);
            this.apiUrl.value = url;
        }

        if (apiKey !== undefined) {
            this.client.apiKey = apiKey;
            localStorage.setItem('api_key', apiKey);
            this.apiKeyInput.value = apiKey;
        }

        this.closeSettingsModal();
        this.checkConnection();
        this.showToast('Settings saved', 'success');
    }

    loadSettings() {
        const savedUrl = localStorage.getItem('api_url');
        const savedKey = localStorage.getItem('api_key');

        if (savedUrl) {
            this.client.baseUrl = savedUrl;
            this.apiUrl.value = savedUrl;
        }

        if (savedKey) {
            this.client.apiKey = savedKey;
            this.apiKeyInput.value = savedKey;
        }
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };

        toast.innerHTML = `
            <span>${icons[type]}</span>
            <span>${message}</span>
        `;

        this.toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('hiding');
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new Dashboard();
});
