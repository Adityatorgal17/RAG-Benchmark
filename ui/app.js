const systemOptions = [
  'S-1', 'S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6a', 'S6b', 'S6c', 'A1'
];

const systemLabels = {
  'S-1': 'Reference baseline',
  S0: 'Naive RAG',
  S1: 'Pre-retrieval',
  S2: 'Post-retrieval',
  S3: 'Pre + post retrieval',
  S4: 'LLM-centric',
  S5: 'Single-agent',
  S6a: 'Acquisition-driven multi-agent',
  S6b: 'Utilization-driven multi-agent',
  S6c: 'Hybrid multi-agent',
  A1: 'Ablation: no routing',
};

const state = {
  selectedSystem: 'S0',
};

function getSystemDisplay(system) {
  return `${system} — ${systemLabels[system] || 'RAG system'}`;
}

const systemToggle = document.getElementById('system-toggle');
const questionInput = document.getElementById('question-input');
const runButton = document.getElementById('run-button');
const statusPill = document.getElementById('status');
const selectedSystem = document.getElementById('selected-system');
const answerOutput = document.getElementById('answer-output');
const thinkingOutput = document.getElementById('thinking-output');
const cotOutput = document.getElementById('cot-output');
const preOutput = document.getElementById('pre-output');
const postOutput = document.getElementById('post-output');
const evidenceOutput = document.getElementById('evidence-output');

function renderSystemToggle() {
  systemToggle.innerHTML = '';
  systemOptions.forEach((system) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `toggle-button ${system === state.selectedSystem ? 'active' : ''}`;
    button.textContent = getSystemDisplay(system);
    button.title = getSystemDisplay(system);
    button.addEventListener('click', () => {
      state.selectedSystem = system;
      selectedSystem.textContent = getSystemDisplay(system);
      renderSystemToggle();
    });
    systemToggle.appendChild(button);
  });
}

function renderList(id, values) {
  const node = document.getElementById(id);
  if (!values || values.length === 0) {
    node.textContent = '—';
    return;
  }
  node.textContent = values.map((item) => typeof item === 'string' ? item : JSON.stringify(item, null, 2)).join('\n\n');
}

function renderEvidence(items) {
  if (!items || items.length === 0) {
    evidenceOutput.innerHTML = '<div class="evidence-item"><p>No evidence was selected for this run.</p></div>';
    return;
  }

  evidenceOutput.innerHTML = items
    .map((item) => `
      <div class="evidence-item">
        <h4>${(item.header || item.chunk_id || 'Evidence item')}</h4>
        <p><strong>ID:</strong> ${item.chunk_id || 'n/a'}</p>
        <p><strong>Source:</strong> ${item.source_type || 'unknown'}</p>
        <p>${(item.note ? item.note + '\n\n' : '') + (item.text || '')}</p>
      </div>
    `)
    .join('');
}

async function runSystem() {
  const question = questionInput.value.trim();
  if (!question) {
    statusPill.textContent = 'Enter a prompt first';
    statusPill.style.color = '#b76b00';
    return;
  }

  statusPill.textContent = 'Running…';
  statusPill.style.color = '#3a70ff';

  try {
    const response = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ system: state.selectedSystem, question })
    });

    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const data = await response.json();
    const activeSystem = data.system || state.selectedSystem;
    selectedSystem.textContent = getSystemDisplay(activeSystem);
    answerOutput.textContent = data.answer || 'No answer received.';
    renderList('thinking-output', data.model_thinking_state || []);
    renderList('cot-output', data.chain_of_thought || []);
    renderList('pre-output', data.pre_retrieval || []);
    renderList('post-output', data.post_retrieval || []);
    renderEvidence(data.evidence || []);
    statusPill.textContent = 'Completed';
    statusPill.style.color = '#1a9b72';
  } catch (error) {
    statusPill.textContent = 'Error';
    statusPill.style.color = '#c62828';
    answerOutput.textContent = error.message;
    console.error(error);
  }
}

runButton.addEventListener('click', runSystem);
questionInput.value = 'What is the main threat in this corpus?';
renderSystemToggle();
selectedSystem.textContent = getSystemDisplay(state.selectedSystem);
