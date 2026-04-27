/**
 * BIS Standards Recommendation Engine — Frontend Controller
 * Professional UI with comprehensive error handling
 */

// ── State ──────────────────────────────────────────────────────────────────
const state = {
    currentTab: 'single',
    isLoading: false,
    lastResults: null,
    lastBatchResults: null,
    standards: [],
    categories: []
};

// ── DOM Elements ──────────────────────────────────────────────────────────
const els = {
    // Tabs
    tabBtns: document.querySelectorAll('.tab-btn'),
    tabPanels: document.querySelectorAll('.tab-panel'),
    
    // Single Search
    queryInput: document.getElementById('queryInput'),
    charCount: document.getElementById('charCount'),
    topKInput: document.getElementById('topKInput'),
    topKValue: document.getElementById('topKValue'),
    searchBtn: document.getElementById('searchBtn'),
    loadingState: document.getElementById('loadingState'),
    errorState: document.getElementById('errorState'),
    errorTitle: document.getElementById('errorTitle'),
    errorMessage: document.getElementById('errorMessage'),
    errorSuggestion: document.getElementById('errorSuggestion'),
    resultsSection: document.getElementById('resultsSection'),
    resultQuery: document.getElementById('resultQuery'),
    resultLatency: document.getElementById('resultLatency'),
    resultCount: document.getElementById('resultCount'),
    resultsGrid: document.getElementById('resultsGrid'),
    copyResultsBtn: document.getElementById('copyResultsBtn'),
    exportResultsBtn: document.getElementById('exportResultsBtn'),
    
    // Batch
    uploadZone: document.getElementById('uploadZone'),
    fileInput: document.getElementById('fileInput'),
    batchLoadingState: document.getElementById('batchLoadingState'),
    batchProgress: document.getElementById('batchProgress'),
    batchErrorState: document.getElementById('batchErrorState'),
    batchErrorTitle: document.getElementById('batchErrorTitle'),
    batchErrorMessage: document.getElementById('batchErrorMessage'),
    batchResultsSection: document.getElementById('batchResultsSection'),
    batchTotal: document.getElementById('batchTotal'),
    batchLatency: document.getElementById('batchLatency'),
    batchTableBody: document.getElementById('batchTableBody'),
    exportBatchBtn: document.getElementById('exportBatchBtn'),
    
    // Browse
    browseSearch: document.getElementById('browseSearch'),
    categoryFilter: document.getElementById('categoryFilter'),
    standardsList: document.getElementById('standardsList'),
    totalStandards: document.getElementById('totalStandards'),
    
    // Modal
    modalOverlay: document.getElementById('modalOverlay'),
    modalClose: document.getElementById('modalClose'),
    modalTitle: document.getElementById('modalTitle'),
    modalBody: document.getElementById('modalBody'),
    
    // Toast
    toastContainer: document.getElementById('toastContainer'),
    
    // Theme
    themeToggle: document.getElementById('themeToggle')
};

// ── Helper Functions ──────────────────────────────────────────────────────

function showToast(message, type = 'info', duration = 4000) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icons = { success: 'fa-check-circle', error: 'fa-times-circle', info: 'fa-info-circle' };
    toast.innerHTML = `
        <i class="fas ${icons[type]} toast-icon"></i>
        <span class="toast-message">${escapeHtml(message)}</span>
    `;
    els.toastContainer.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function setLoading(isLoading) {
    state.isLoading = isLoading;
    els.searchBtn.disabled = isLoading;
    els.searchBtn.innerHTML = isLoading 
        ? '<i class="fas fa-spinner fa-spin"></i> Searching...'
        : '<i class="fas fa-search"></i> Search Standards';
}

function showError(title, message, suggestion = '') {
    els.errorState.classList.add('active');
    els.resultsSection.classList.remove('active');
    els.loadingState.classList.remove('active');
    els.errorTitle.textContent = title;
    els.errorMessage.textContent = message;
    els.errorSuggestion.textContent = suggestion;
}

function hideError() {
    els.errorState.classList.remove('active');
}

function showResults() {
    els.resultsSection.classList.add('active');
    els.loadingState.classList.remove('active');
    els.errorState.classList.remove('active');
}

function getScoreClass(score) {
    if (score >= 85) return 'score-high';
    if (score >= 70) return 'score-medium';
    return 'score-low';
}

function getCategoryIcon(category) {
    const icons = {
        'Cement': 'fa-cube',
        'Steel': 'fa-industry',
        'Concrete': 'fa-hammer',
        'Aggregates': 'fa-mountain',
        'Bricks and Masonry': 'fa-border-all',
        'Tiles and Flooring': 'fa-th-large',
        'Pipes and Precast': 'fa-circle',
        'Testing': 'fa-vial'
    };
    return icons[category] || 'fa-file-alt';
}

async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        showToast('Copied to clipboard!', 'success');
    } catch (err) {
        showToast('Failed to copy', 'error');
    }
}

function downloadJson(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast(`Downloaded ${filename}`, 'success');
}

// ── API Functions ─────────────────────────────────────────────────────────

async function apiSearch(query, topK = 5) {
    const response = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: topK })
    });
    
    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail?.detail || error.detail || `HTTP ${response.status}`);
    }
    
    return response.json();
}

async function apiBatch(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('/api/batch', {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(error.detail?.detail || error.detail || `HTTP ${response.status}`);
    }
    
    return response.json();
}

async function fetchStandards() {
    const response = await fetch('/api/standards');
    if (!response.ok) throw new Error('Failed to fetch standards');
    return response.json();
}

// ── UI Render Functions ───────────────────────────────────────────────────

function renderResults(data) {
    els.resultQuery.textContent = data.query;
    els.resultLatency.textContent = data.latency_seconds;
    els.resultCount.textContent = data.results.length;
    
    els.resultsGrid.innerHTML = data.results.map(r => `
        <div class="result-card" onclick="showStandardDetail('${r.code}')">
            <div class="result-card-header">
                <div class="result-rank">${r.rank}</div>
                <div class="result-title-group">
                    <div class="result-code">${escapeHtml(r.code)}</div>
                    <div class="result-title">${escapeHtml(r.title)}</div>
                </div>
                <div class="result-score">
                    <span class="score-badge ${getScoreClass(r.relevance_score)}">${r.relevance_score}%</span>
                </div>
            </div>
            <div class="result-category">
                <i class="fas ${getCategoryIcon(r.category)}"></i>
                ${escapeHtml(r.category)}
            </div>
            <div class="result-description">${escapeHtml(r.description)}</div>
            <div class="result-tags">
                ${r.keywords.map(k => `<span class="tag">${escapeHtml(k)}</span>`).join('')}
            </div>
        </div>
    `).join('');
    
    showResults();
}

function renderBatchResults(data) {
    els.batchTotal.textContent = data.total_queries;
    els.batchLatency.textContent = data.avg_latency_seconds;
    
    els.batchTableBody.innerHTML = data.results.map(r => `
        <tr>
            <td><strong>${escapeHtml(r.id)}</strong></td>
            <td>
                <div class="standards-cell">
                    ${r.retrieved_standards.map(s => `<span class="std-badge">${escapeHtml(s)}</span>`).join('')}
                </div>
            </td>
            <td>${r.latency_seconds}s</td>
            <td>
                <button class="btn-icon" onclick='showBatchDetail(${JSON.stringify(r).replace(/'/g, "&#39;")})' title="View details">
                    <i class="fas fa-eye"></i>
                </button>
            </td>
        </tr>
    `).join('');
    
    els.batchResultsSection.classList.add('active');
    els.batchLoadingState.classList.remove('active');
    els.batchErrorState.classList.remove('active');
}

function renderStandardsList(standards, filter = '', category = '') {
    const filtered = standards.filter(s => {
        const matchesSearch = !filter || 
            s.code.toLowerCase().includes(filter) ||
            s.title.toLowerCase().includes(filter) ||
            s.keywords.some(k => k.toLowerCase().includes(filter));
        const matchesCategory = !category || s.category === category;
        return matchesSearch && matchesCategory;
    });
    
    els.totalStandards.textContent = filtered.length;
    
    els.standardsList.innerHTML = filtered.map(s => `
        <div class="standard-item" onclick="showStandardDetail('${s.code}')">
            <div class="standard-item-code">${escapeHtml(s.code)}</div>
            <div class="standard-item-info">
                <div class="standard-item-title">${escapeHtml(s.title)}</div>
                <div class="standard-item-category">
                    <i class="fas ${getCategoryIcon(s.category)}"></i>
                    ${escapeHtml(s.category)}
                </div>
            </div>
        </div>
    `).join('') || '<p style="text-align:center;color:var(--text-muted);padding:2rem;">No standards found.</p>';
}

// ── Modal Functions ───────────────────────────────────────────────────────

window.showStandardDetail = async function(code) {
    try {
        const std = state.standards.find(s => s.code === code);
        if (!std) {
            showToast('Standard details not found', 'error');
            return;
        }
        
        els.modalTitle.textContent = std.code;
        els.modalBody.innerHTML = `
            <div style="margin-bottom:1rem;">
                <span class="score-badge score-high" style="display:inline-block;margin-bottom:0.5rem;">
                    <i class="fas ${getCategoryIcon(std.category)}"></i> ${escapeHtml(std.category)}
                </span>
            </div>
            <h4 style="margin-bottom:0.75rem;color:var(--text);">${escapeHtml(std.title)}</h4>
            <p style="color:var(--text-secondary);line-height:1.6;margin-bottom:1rem;">${escapeHtml(std.description || 'No description available.')}</p>
            <div style="margin-bottom:1rem;">
                <strong style="font-size:0.8125rem;color:var(--text);">Keywords:</strong>
                <div class="result-tags" style="margin-top:0.5rem;">
                    ${(std.keywords || []).map(k => `<span class="tag">${escapeHtml(k)}</span>`).join('')}
                </div>
            </div>
            <div style="margin-bottom:1rem;">
                <strong style="font-size:0.8125rem;color:var(--text);">Applications:</strong>
                <ul style="margin-top:0.5rem;padding-left:1.25rem;color:var(--text-secondary);">
                    ${(std.applications || []).map(a => `<li>${escapeHtml(a)}</li>`).join('')}
                </ul>
            </div>
        `;
        els.modalOverlay.classList.add('active');
    } catch (err) {
        showToast('Error loading details', 'error');
    }
};

window.showBatchDetail = function(result) {
    els.modalTitle.textContent = `Query: ${result.id}`;
    els.modalBody.innerHTML = `
        <div style="margin-bottom:1rem;">
            <strong>Retrieved Standards:</strong>
            <div class="result-tags" style="margin-top:0.5rem;">
                ${result.retrieved_standards.map(s => `<span class="tag">${escapeHtml(s)}</span>`).join('')}
            </div>
        </div>
        <div style="margin-bottom:1rem;">
            <strong>Latency:</strong> ${result.latency_seconds}s
        </div>
        ${result.results_detail ? `
        <div>
            <strong>Detailed Results:</strong>
            <div style="margin-top:0.75rem;display:grid;gap:0.75rem;">
                ${result.results_detail.map(r => `
                    <div class="standard-item" style="cursor:default;">
                        <div class="standard-item-code">${r.code}</div>
                        <div class="standard-item-info">
                            <div class="standard-item-title">${escapeHtml(r.title)}</div>
                            <div style="font-size:0.75rem;color:var(--text-muted);">Relevance: ${r.relevance_score}%</div>
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>
        ` : ''}
    `;
    els.modalOverlay.classList.add('active');
};

// ── Event Handlers ────────────────────────────────────────────────────────

// Tab switching
els.tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const tab = btn.dataset.tab;
        state.currentTab = tab;
        
        els.tabBtns.forEach(b => b.classList.remove('active'));
        els.tabPanels.forEach(p => p.classList.remove('active'));
        
        btn.classList.add('active');
        document.getElementById(`${tab}-panel`).classList.add('active');
        
        if (tab === 'browse' && state.standards.length === 0) {
            loadBrowseData();
        }
    });
});

// Character count
els.queryInput.addEventListener('input', () => {
    els.charCount.textContent = els.queryInput.value.length;
});

// TopK slider
els.topKInput.addEventListener('input', () => {
    els.topKValue.textContent = els.topKInput.value;
});

// Search
els.searchBtn.addEventListener('click', async () => {
    const query = els.queryInput.value.trim();
    if (!query) {
        showToast('Please enter a product description', 'error');
        els.queryInput.focus();
        return;
    }
    
    const topK = parseInt(els.topKInput.value);
    
    hideError();
    els.resultsSection.classList.remove('active');
    els.loadingState.classList.add('active');
    setLoading(true);
    
    try {
        const data = await apiSearch(query, topK);
        state.lastResults = data;
        renderResults(data);
        showToast(`Found ${data.results.length} standards in ${data.latency_seconds}s`, 'success');
    } catch (err) {
        showError('Search Failed', err.message, 'Please check your query and try again.');
        showToast(err.message, 'error');
    } finally {
        setLoading(false);
    }
});

// Enter key to search
els.queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        els.searchBtn.click();
    }
});

// Copy results
els.copyResultsBtn.addEventListener('click', () => {
    if (!state.lastResults) return;
    const codes = state.lastResults.results.map(r => r.code).join('\n');
    copyToClipboard(codes);
});

// Export results
els.exportResultsBtn.addEventListener('click', () => {
    if (!state.lastResults) return;
    const exportData = {
        query: state.lastResults.query,
        latency_seconds: state.lastResults.latency_seconds,
        retrieved_standards: state.lastResults.results.map(r => r.code)
    };
    downloadJson(exportData, 'bis-search-results.json');
});

// ── Batch Upload ──────────────────────────────────────────────────────────

els.uploadZone.addEventListener('click', () => els.fileInput.click());

els.uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    els.uploadZone.classList.add('dragover');
});

els.uploadZone.addEventListener('dragleave', () => {
    els.uploadZone.classList.remove('dragover');
});

els.uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    els.uploadZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleBatchUpload(files[0]);
});

els.fileInput.addEventListener('change', () => {
    if (els.fileInput.files.length > 0) handleBatchUpload(els.fileInput.files[0]);
});

async function handleBatchUpload(file) {
    if (!file.name.endsWith('.json')) {
        showToast('Please upload a JSON file', 'error');
        return;
    }
    
    els.batchResultsSection.classList.remove('active');
    els.batchErrorState.classList.remove('active');
    els.batchLoadingState.classList.add('active');
    els.batchProgress.style.width = '30%';
    
    try {
        els.batchProgress.style.width = '60%';
        const data = await apiBatch(file);
        state.lastBatchResults = data;
        els.batchProgress.style.width = '100%';
        
        setTimeout(() => {
            renderBatchResults(data);
            showToast(`Processed ${data.total_queries} queries successfully`, 'success');
        }, 300);
    } catch (err) {
        els.batchLoadingState.classList.remove('active');
        els.batchErrorState.classList.add('active');
        els.batchErrorTitle.textContent = 'Batch Processing Failed';
        els.batchErrorMessage.textContent = err.message;
        showToast(err.message, 'error');
    }
}

els.exportBatchBtn.addEventListener('click', () => {
    if (!state.lastBatchResults) return;
    const exportData = state.lastBatchResults.results.map(r => ({
        id: r.id,
        retrieved_standards: r.retrieved_standards,
        latency_seconds: r.latency_seconds
    }));
    downloadJson(exportData, 'bis-batch-results.json');
});

// ── Browse Standards ──────────────────────────────────────────────────────

async function loadBrowseData() {
    try {
        const data = await fetchStandards();
        state.standards = data.standards;
        state.categories = data.categories;
        
        // Populate category filter
        els.categoryFilter.innerHTML = '<option value="">All Categories</option>' +
            data.categories.map(c => `<option value="${c}">${c}</option>`).join('');
        
        renderStandardsList(data.standards);
    } catch (err) {
        showToast('Failed to load standards', 'error');
    }
}

els.browseSearch.addEventListener('input', () => {
    const filter = els.browseSearch.value.toLowerCase();
    const category = els.categoryFilter.value;
    renderStandardsList(state.standards, filter, category);
});

els.categoryFilter.addEventListener('change', () => {
    const filter = els.browseSearch.value.toLowerCase();
    const category = els.categoryFilter.value;
    renderStandardsList(state.standards, filter, category);
});

// ── Modal ─────────────────────────────────────────────────────────────────

els.modalClose.addEventListener('click', () => {
    els.modalOverlay.classList.remove('active');
});

els.modalOverlay.addEventListener('click', (e) => {
    if (e.target === els.modalOverlay) {
        els.modalOverlay.classList.remove('active');
    }
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        els.modalOverlay.classList.remove('active');
    }
});

// ── Theme Toggle ──────────────────────────────────────────────────────────

function initTheme() {
    const saved = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const isDark = saved === 'dark' || (!saved && prefersDark);
    
    if (isDark) {
        document.documentElement.setAttribute('data-theme', 'dark');
        els.themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
    }
}

els.themeToggle.addEventListener('click', () => {
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    
    if (isDark) {
        document.documentElement.removeAttribute('data-theme');
        localStorage.setItem('theme', 'light');
        els.themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
    } else {
        document.documentElement.setAttribute('data-theme', 'dark');
        localStorage.setItem('theme', 'dark');
        els.themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
    }
});

// ── Initialize ────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    els.queryInput.focus();
});

