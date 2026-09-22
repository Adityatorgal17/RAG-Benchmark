const systemOptions = [
  'S-1', 'S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6a', 'S6b', 'S6c', 'A1', 'A2a', 'A2b'
];

const systemLabels = {
  'S-1': ['No retrieval', 'Provider answers without corpus evidence'],
  S0: ['Naive RAG', 'Dense retrieval plus fixed generation'],
  S1: ['Pre-retrieval', 'Rewrite, HyDE, and rank fusion'],
  S2: ['Post-retrieval', 'Hybrid search, rerank, filter, compress'],
  S3: ['Decomposition', 'Sub-questions plus post-retrieval refinement'],
  S4: ['LLM-centric', 'Gate, notes, critique, and revision'],
  S5: ['Single-agent', 'Search loop with agent state'],
  S6a: ['Acquisition agents', 'Decision, query, and sufficiency roles'],
  S6b: ['Utilization agents', 'Evaluate, note, generate, reconcile'],
  S6c: ['Hybrid agents', 'Plan, route sources, refine, decide'],
  A1: ['No routing ablation', 'S6c without source routing'],
  A2a: ['Rerank-only ablation', 'S2 without filter and compression'],
  A2b: ['No compression ablation', 'S2 filtering without compression'],
};

const state = {
  selectedSystem: 'S0',
  runtime: null,
};

const systemToggle = document.getElementById('system-toggle');
const questionInput = document.getElementById('question-input');
const runButton = document.getElementById('run-button');
const statusPill = document.getElementById('status');
const selectedSystem = document.getElementById('selected-system');
const runtimeMode = document.getElementById('runtime-mode');
const runtimeBadge = document.getElementById('runtime-badge');
const answerOutput = document.getElementById('answer-output');
const preOutput = document.getElementById('pre-output');
const postOutput = document.getElementById('post-output');
const traceOutput = document.getElementById('trace-output');
const candidateOutput = document.getElementById('candidate-output');
const evidenceOutput = document.getElementById('evidence-output');
const llmCalls = document.getElementById('llm-calls');
const latency = document.getElementById('latency');
const citationCount = document.getElementById('citation-count');
const abstentionBadge = document.getElementById('abstention-badge');

function systemTitle(system) {
  const label = systemLabels[system]?.[0] || 'RAG system';
  return `${system} - ${label}`;
}

function setStatus(text, tone = 'neutral') {
  statusPill.textContent = text;
  const colors = {
    neutral: '#08666a',
    success: '#13795b',
    warning: '#a15c00',
    danger: '#b42318',
  };
  statusPill.style.color = colors[tone] || colors.neutral;
}

function renderSystemToggle() {
  systemToggle.innerHTML = '';
  systemOptions.forEach((system) => {
    const button = document.createElement('button');
    const [title, subtitle] = systemLabels[system] || ['RAG system', 'Benchmark variant'];
    button.type = 'button';
    button.className = `toggle-button ${system === state.selectedSystem ? 'active' : ''}`;
    button.title = `${system}: ${subtitle}`;

    const strong = document.createElement('strong');
    strong.textContent = `${system} - ${title}`;
    const span = document.createElement('span');
    span.textContent = subtitle;
    button.append(strong, span);

    button.addEventListener('click', () => {
      state.selectedSystem = system;
      selectedSystem.textContent = systemTitle(system);
      renderSystemToggle();
    });
    systemToggle.appendChild(button);
  });
}

function renderTextBlock(node, values, emptyText) {
  node.classList.toggle('empty', !values || values.length === 0);
  if (!values || values.length === 0) {
    node.textContent = emptyText;
    return;
  }
  node.textContent = values
    .map((item) => (typeof item === 'string' ? item : JSON.stringify(item, null, 2)))
    .join('\n\n');
}

function renderTrace(steps) {
  traceOutput.innerHTML = '';
  traceOutput.classList.toggle('empty', !steps || steps.length === 0);
  if (!steps || steps.length === 0) {
    traceOutput.textContent = 'No trace yet.';
    return;
  }

  steps.forEach((step, index) => {
    const item = document.createElement('div');
    item.className = 'trace-item';

    const title = document.createElement('strong');
    title.textContent = `${index + 1}. ${step.tag || 'step'}`;
    const body = document.createElement('span');
    body.textContent = step.summary || step.system || '';
    item.append(title, body);
    traceOutput.appendChild(item);
  });
}

function renderChips(node, values, emptyText) {
  node.innerHTML = '';
  if (!values || values.length === 0) {
    const empty = document.createElement('span');
    empty.className = 'empty';
    empty.textContent = emptyText;
    node.appendChild(empty);
    return;
  }
  values.forEach((value) => {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = value;
    node.appendChild(chip);
  });
}

function renderEvidence(items) {
  evidenceOutput.innerHTML = '';
  if (!items || items.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'evidence-item';
    const text = document.createElement('p');
    text.textContent = 'No evidence was selected for this run.';
    empty.appendChild(text);
    evidenceOutput.appendChild(empty);
    return;
  }

  items.forEach((item) => {
    const card = document.createElement('div');
    card.className = 'evidence-item';

    const title = document.createElement('h4');
    title.textContent = item.header || item.chunk_id || 'Evidence item';

    const meta = document.createElement('div');
    meta.className = 'evidence-meta';
    [item.chunk_id || 'n/a', item.source_type || 'unknown'].forEach((value) => {
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.textContent = value;
      meta.appendChild(chip);
    });

    const body = document.createElement('p');
    body.textContent = `${item.note ? `${item.note}\n\n` : ''}${item.text || ''}`;
    card.append(title, meta, body);
    evidenceOutput.appendChild(card);
  });
}

function renderRuntime(runtime) {
  state.runtime = runtime;
  const isReal = runtime?.llm_mode === 'real';
  runtimeMode.textContent = isReal
    ? `${runtime.provider || 'provider'} / ${runtime.model || 'configured model'}`
    : 'FakeLLM dry run';
  runtimeBadge.textContent = isReal ? 'Real LLM' : 'Fake LLM';
  runtimeBadge.className = `status-badge ${isReal ? 'real' : 'fake'}`;
}

function renderUsage(data) {
  const usage = data.usage || {};
  llmCalls.textContent = `${usage.llm_calls || 0} calls`;
  latency.textContent = `${Number(usage.wall_s || usage.latency_s || 0).toFixed(2)}s`;
  citationCount.textContent = `${(data.cited_ids || []).length} citations`;
  abstentionBadge.textContent = data.abstained ? 'Insufficient evidence' : 'Answered';
  abstentionBadge.className = `status-badge ${data.abstained ? 'warning' : 'success'}`;
}

async function loadRuntimeStatus() {
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error(`Status failed with ${response.status}`);
    renderRuntime(await response.json());
  } catch (error) {
    runtimeMode.textContent = 'Status unavailable';
    runtimeBadge.textContent = 'Check server';
    runtimeBadge.className = 'status-badge danger';
  }
}

async function runSystem() {
  const question = questionInput.value.trim();
  if (!question) {
    setStatus('Enter a question first', 'warning');
    return;
  }

  runButton.disabled = true;
  setStatus('Running...', 'neutral');
  answerOutput.textContent = 'Running retrieval and generation...';

  try {
    const response = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ system: state.selectedSystem, question }),
    });

    if (!response.ok) {
      throw new Error(`Request failed with status ${response.status}`);
    }

    const data = await response.json();
    const activeSystem = data.system || state.selectedSystem;
    selectedSystem.textContent = systemTitle(activeSystem);
    answerOutput.textContent = data.answer || 'No answer received.';
    renderRuntime(data.runtime || state.runtime);
    renderUsage(data);
    renderTextBlock(preOutput, data.pre_retrieval || [], 'No pre-retrieval steps for this system.');
    renderTextBlock(postOutput, data.post_retrieval || [], 'No post-retrieval steps for this system.');
    renderTrace(data.steps || []);
    renderChips(candidateOutput, data.candidate_ids || [], 'No candidates reported.');
    renderEvidence(data.evidence || []);
    setStatus('Completed', 'success');
  } catch (error) {
    setStatus('Error', 'danger');
    answerOutput.textContent = error.message;
    console.error(error);
  } finally {
    runButton.disabled = false;
  }
}

runButton.addEventListener('click', runSystem);
questionInput.value = 'Which Palo Alto Networks PAN-OS issue is the main focus of the corpus, and what makes it risky?';
renderSystemToggle();
selectedSystem.textContent = systemTitle(state.selectedSystem);
renderChips(candidateOutput, [], 'No candidates yet.');
renderEvidence([]);
loadRuntimeStatus();
