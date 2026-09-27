/* ============================================================================
   RAG PIPELINE EXPLORER - MAIN APPLICATION
   ============================================================================ */

// System definitions
const SYSTEMS = {
  'S-1': ['No Retrieval', 'Provider answers without corpus evidence'],
  'S0': ['Naive RAG', 'Dense retrieval plus fixed generation'],
  'S1': ['Pre-retrieval', 'Rewrite, HyDE, and rank fusion'],
  'S2': ['Post-retrieval', 'Hybrid search, rerank, filter, compress'],
  'S3': ['Decomposition', 'Sub-questions plus post-retrieval refinement'],
  'S4': ['LLM-centric', 'Gate, notes, critique, and revision'],
  'S5': ['Single-agent', 'Search loop with agent state'],
  'S6a': ['Acquisition Agents', 'Decision, query, and sufficiency roles'],
  'S6b': ['Utilization Agents', 'Evaluate, note, generate, reconcile'],
  'S6c': ['Hybrid Agents', 'Plan, route sources, refine, decide'],
  'A1': ['No Routing', 'S6c without source routing (ablation)'],
  'A2a': ['Rerank-only', 'S2 without filter and compression (ablation)'],
  'A2b': ['No Compression', 'S2 filtering without compression (ablation)'],
};

const SYSTEM_ORDER = [
  'S-1', 'S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6a', 'S6b', 'S6c', 'A1', 'A2a', 'A2b'
];

// Application state
const state = {
  selectedSystem: 'S0',
  isRunning: false,
  currentRun: null,
};

// DOM Elements
const systemSelector = document.getElementById('system-selector');
const questionInput = document.getElementById('question-input');
const runButton = document.getElementById('run-button');
const messagesContainer = document.getElementById('messages-container');
const selectedSystemBadge = document.getElementById('selected-system-badge');
const systemTitle = document.getElementById('system-title');
const systemDesc = document.getElementById('system-desc');
const runtimeMode = document.getElementById('runtime-mode');
const runtimeDetail = document.getElementById('runtime-detail');
const statLlm = document.getElementById('stat-llm');
const statTime = document.getElementById('stat-time');
const statStatus = document.getElementById('stat-status');
const runText = document.getElementById('run-text');
const runSpinner = document.getElementById('run-spinner');
const clearChatBtn = document.getElementById('clear-chat');
const exportTraceBtn = document.getElementById('export-trace');
const modal = document.getElementById('step-detail-modal');
const modalTitle = document.getElementById('modal-title');
const modalBody = document.getElementById('modal-body');
const modalClose = document.getElementById('modal-close');

/* ============================================================================
   INITIALIZATION
   ============================================================================ */

function init() {
  renderSystemSelector();
  setupEventListeners();
  loadRuntimeStatus();
}

function setupEventListeners() {
  runButton.addEventListener('click', handleRun);
  clearChatBtn.addEventListener('click', clearChat);
  exportTraceBtn.addEventListener('click', exportTrace);
  modalClose.addEventListener('click', closeModal);
  modal.addEventListener('click', (e) => {
    if (e.target === modal) closeModal();
  });
  questionInput.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') handleRun();
  });
}

function renderSystemSelector() {
  systemSelector.innerHTML = '';
  SYSTEM_ORDER.forEach((systemId) => {
    const [title, desc] = SYSTEMS[systemId];
    const btn = document.createElement('button');
    btn.className = `system-button ${systemId === state.selectedSystem ? 'active' : ''}`;
    btn.textContent = systemId;
    btn.title = `${systemId}: ${title} - ${desc}`;
    btn.addEventListener('click', () => selectSystem(systemId));
    systemSelector.appendChild(btn);
  });
}

function selectSystem(systemId) {
  state.selectedSystem = systemId;
  const [title, desc] = SYSTEMS[systemId];
  selectedSystemBadge.textContent = systemId;
  systemTitle.textContent = title;
  systemDesc.textContent = desc;
  renderSystemSelector();
}

function loadRuntimeStatus() {
  fetch('/api/status')
    .then(r => r.json())
    .then(status => {
      const mode = status.llm_mode === 'real' ? 'REAL' : 'FAKE';
      const provider = status.provider?.toUpperCase() || 'UNKNOWN';
      runtimeMode.textContent = mode;
      runtimeDetail.textContent = `${provider} ${status.model}`;
    })
    .catch(err => {
      console.error('Status load failed:', err);
      runtimeMode.textContent = 'ERROR';
      runtimeDetail.textContent = 'Failed to load runtime';
    });
}

/* ============================================================================
   MAIN RUN HANDLER
   ============================================================================ */

async function handleRun() {
  const question = questionInput.value.trim();
  if (!question) {
    alert('Please enter a question');
    return;
  }

  if (state.isRunning) return;

  state.isRunning = true;
  runButton.disabled = true;
  runText.style.display = 'none';
  runSpinner.style.display = 'inline-block';
  statStatus.textContent = 'Running';
  statStatus.className = 'stat-value status-running';

  // Add query message
  addMessage('query', question);

  try {
    const startTime = performance.now();
    const response = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        system: state.selectedSystem,
        question: question,
      }),
    });

    if (!response.ok) throw new Error(`API error: ${response.status}`);
    const result = await response.json();
    const elapsedTime = ((performance.now() - startTime) / 1000).toFixed(2);

    state.currentRun = result;

    // Update stats
    statLlm.textContent = result.llm_calls || '0';
    statTime.textContent = `${elapsedTime}s`;

    // Process and display results
    displayPipeline(result);

    // Add answer
    addAnswer(result);

    statStatus.textContent = 'Complete';
    statStatus.className = 'stat-value status-complete';
  } catch (err) {
    console.error('Run failed:', err);
    addMessage('error', `Error: ${err.message}`);
    statStatus.textContent = 'Error';
    statStatus.className = 'stat-value status-error';
  } finally {
    state.isRunning = false;
    runButton.disabled = false;
    runText.style.display = 'inline';
    runSpinner.style.display = 'none';
  }
}

/* ============================================================================
   PIPELINE VISUALIZATION
   ============================================================================ */

function displayPipeline(result) {
  const trace = result.trace || [];
  
  // Group steps by phase
  const phases = {
    'query_rewrite': { title: '🔄 Query Processing', icon: '🔄', steps: [] },
    'retrieval': { title: '🔍 Retrieval', icon: '🔍', steps: [] },
    'reranking': { title: '⭐ Reranking', icon: '⭐', steps: [] },
    'filtering': { title: '🎯 Filtering', icon: '🎯', steps: [] },
    'compression': { title: '📦 Compression', icon: '📦', steps: [] },
    'generation': { title: '✍️ Generation', icon: '✍️', steps: [] },
    'agent': { title: '🤖 Agent Steps', icon: '🤖', steps: [] },
  };

  trace.forEach(step => {
    const tag = (step.tag || '').toLowerCase();
    let phase = 'agent';
    
    if (tag.includes('rewrite') || tag.includes('hyde')) phase = 'query_rewrite';
    else if (tag.includes('dense') || tag.includes('sparse') || tag.includes('retriev')) phase = 'retrieval';
    else if (tag.includes('rerank')) phase = 'reranking';
    else if (tag.includes('filter')) phase = 'filtering';
    else if (tag.includes('compress')) phase = 'compression';
    else if (tag.includes('generat') || tag.includes('answer')) phase = 'generation';

    phases[phase].steps.push(step);
  });

  // Render each phase
  Object.values(phases).forEach(phase => {
    if (phase.steps.length > 0) {
      renderPhase(phase);
    }
  });

  // Show retrieved chunks
  if (result.candidates && result.candidates.length > 0) {
    addCandidatesMessage(result.candidates);
  }
}

function renderPhase(phase) {
  const phaseBlock = document.createElement('div');
  phaseBlock.className = 'pipeline-message';

  const header = document.createElement('div');
  header.className = 'message-block';
  header.innerHTML = `
    <div class="message-header">
      <div class="message-title">
        <span>${phase.title}</span>
      </div>
      <div class="message-status status-processing">Running</div>
      <button class="message-toggle" data-phase="${phase.title}">▼</button>
    </div>
    <div class="message-content">
      ${phase.steps.map((step, i) => `
        <div class="step-detail ${getStepStatus(step)}">
          <span class="step-label">${step.tag}</span>
          <div class="step-output">${formatStepOutput(step.result)}</div>
          ${i < phase.steps.length - 1 ? '<div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid rgba(255,255,255,0.1);"></div>' : ''}
        </div>
      `).join('')}
    </div>
  `;

  messagesContainer.appendChild(header);

  // Setup toggle
  const toggle = header.querySelector('.message-toggle');
  const content = header.querySelector('.message-content');
  toggle.addEventListener('click', (e) => {
    e.stopPropagation();
    content.classList.toggle('expanded');
    toggle.textContent = content.classList.contains('expanded') ? '▲' : '▼';
  });

  // Auto-expand first time
  content.classList.add('expanded');
  toggle.textContent = '▲';
}

function addCandidatesMessage(candidates) {
  const msg = document.createElement('div');
  msg.className = 'pipeline-message';
  
  const block = document.createElement('div');
  block.className = 'message-block';
  
  const header = document.createElement('div');
  header.className = 'message-header';
  header.innerHTML = `
    <div class="message-title">📋 Retrieved Chunks</div>
    <button class="message-toggle">▼</button>
  `;
  
  const content = document.createElement('div');
  content.className = 'message-content';
  content.innerHTML = `
    <div style="display: flex; flex-wrap: wrap; gap: 6px;">
      ${candidates.map(c => `<div class="chunk-chip">${c}</div>`).join('')}
    </div>
  `;
  
  block.appendChild(header);
  block.appendChild(content);
  msg.appendChild(block);
  messagesContainer.appendChild(msg);

  const toggle = header.querySelector('.message-toggle');
  toggle.addEventListener('click', (e) => {
    e.stopPropagation();
    content.classList.toggle('expanded');
    toggle.textContent = content.classList.contains('expanded') ? '▲' : '▼';
  });
}

function getStepStatus(step) {
  if (step.error) return 'error';
  if (step.result === 'INSUFFICIENT_EVIDENCE') return 'warning';
  return 'success';
}

function formatStepOutput(result) {
  if (!result) return '(no output)';
  if (typeof result === 'string') {
    if (result.length > 500) return result.substring(0, 500) + '...';
    return result;
  }
  try {
    const json = JSON.stringify(result, null, 2);
    if (json.length > 500) return json.substring(0, 500) + '...';
    return json;
  } catch {
    return String(result);
  }
}

function addAnswer(result) {
  const msg = document.createElement('div');
  msg.className = 'pipeline-message';
  
  const answer = result.answer || 'INSUFFICIENT_EVIDENCE';
  const isInsufficient = answer.includes('INSUFFICIENT_EVIDENCE');
  const statusClass = isInsufficient ? 'insufficient' : 'good';
  const statusText = isInsufficient ? 'NO EVIDENCE' : 'GENERATED';
  
  msg.innerHTML = `
    <div class="answer-message">
      <span class="answer-status ${statusClass}">${statusText}</span>
      <div class="answer-text">${escapeHtml(answer)}</div>
      ${result.evidence && result.evidence.length > 0 ? `
        <div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid rgba(255,255,255,0.1);">
          <div style="font-size: 11px; text-transform: uppercase; color: #00d4ff; margin-bottom: 8px; letter-spacing: 0.5px;">📎 Citations</div>
          ${result.evidence.map(e => `
            <div class="evidence-block">
              <div class="evidence-text">${escapeHtml(e.text)}</div>
              <div style="margin-top: 6px; font-size: 11px; color: #808080;">
                Source: <span class="chunk-chip secondary">${e.chunk_id}</span>
              </div>
            </div>
          `).join('')}
        </div>
      ` : ''}
    </div>
  `;
  
  messagesContainer.appendChild(msg);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function addMessage(type, text) {
  const msg = document.createElement('div');
  msg.className = 'pipeline-message';
  
  const typeIcons = {
    query: '❓',
    error: '❌',
    info: 'ℹ️',
  };
  
  const typeClasses = {
    query: 'status-processing',
    error: 'status-failed',
    info: 'status-pending',
  };

  msg.innerHTML = `
    <div class="message-block">
      <div class="message-header">
        <div class="message-title">
          <span>${typeIcons[type] || '📝'} ${type.charAt(0).toUpperCase() + type.slice(1)}</span>
        </div>
        <div class="message-status ${typeClasses[type] || 'status-pending'}">${type}</div>
      </div>
      <div class="message-content expanded">
        <p style="margin: 0; color: var(--text-secondary);">${escapeHtml(text)}</p>
      </div>
    </div>
  `;
  
  messagesContainer.appendChild(msg);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

/* ============================================================================
   UTILITY FUNCTIONS
   ============================================================================ */

function escapeHtml(text) {
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;',
  };
  return String(text).replace(/[&<>"']/g, m => map[m]);
}

function clearChat() {
  messagesContainer.innerHTML = `
    <div class="welcome-message">
      <h2>Chat Cleared</h2>
      <p>Ask a new question to start fresh.</p>
    </div>
  `;
  statLlm.textContent = '0';
  statTime.textContent = '--';
  statStatus.textContent = 'Ready';
  statStatus.className = 'stat-value status-ready';
}

function exportTrace() {
  if (!state.currentRun) {
    alert('No run to export. Please run a query first.');
    return;
  }

  const trace = state.currentRun;
  const json = JSON.stringify(trace, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `rag-trace-${state.selectedSystem}-${new Date().toISOString()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

function closeModal() {
  modal.classList.remove('active');
}

/* ============================================================================
   INITIALIZATION
   ============================================================================ */

document.addEventListener('DOMContentLoaded', init);
