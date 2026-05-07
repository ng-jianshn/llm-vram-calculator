// =========================================
//  LLM VRAM Calculator — Frontend Logic
// =========================================

let currentData = null;

// ---- Precision mode management ----
function getSelectedFormats() {
    const checked = document.querySelectorAll('#precision-grid input[type="checkbox"]:checked');
    return Array.from(checked).map((cb) => cb.value);
}

function togglePill(e) {
    const label = e.target.closest('.precision-pill');
    if (!label) return;
    const cb = label.querySelector('input[type="checkbox"]');
    // The checkbox toggles automatically; we just sync the class
    setTimeout(() => {
        label.classList.toggle('selected', cb.checked);
    }, 0);
}

// Attach click handlers to precision pills
document.getElementById('precision-grid').addEventListener('change', togglePill);

function selectAllFormats() {
    document.querySelectorAll('#precision-grid input[type="checkbox"]').forEach((cb) => {
        cb.checked = true;
        cb.closest('.precision-pill').classList.add('selected');
    });
}

function selectNoFormats() {
    document.querySelectorAll('#precision-grid input[type="checkbox"]').forEach((cb) => {
        cb.checked = false;
        cb.closest('.precision-pill').classList.remove('selected');
    });
}

function selectPresetFormats(preset) {
    const presets = {
        common: ['FP16', 'BF16', 'FP8', 'INT8', 'INT4'],
        gguf: ['GGUF_Q8', 'GGUF_Q6', 'GGUF_Q5', 'GGUF_Q4', 'GGUF_Q3', 'GGUF_Q2'],
    };
    const selected = presets[preset] || [];
    document.querySelectorAll('#precision-grid input[type="checkbox"]').forEach((cb) => {
        cb.checked = selected.includes(cb.value);
        cb.closest('.precision-pill').classList.toggle('selected', cb.checked);
    });
}

// ---- Quick example buttons ----
function setModel(name) {
    document.getElementById('model-input').value = name;
    analyzeModel();
}

// Enter key support
document.getElementById('model-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') analyzeModel();
});

// ---- Inference param helpers ----
function getInferenceParams() {
    return {
        sequence_length: parseInt(document.getElementById('param-seq-length').value) || 2048,
        batch_size: parseInt(document.getElementById('param-batch-size').value) || 1,
        concurrent_users: parseInt(document.getElementById('param-concurrent').value) || 1,
    };
}

function syncSlider(type) {
    if (type === 'seq') {
        document.getElementById('param-seq-length').value = document.getElementById('param-seq-slider').value;
        highlightSeqPreset();
    } else if (type === 'batch') {
        document.getElementById('param-batch-size').value = document.getElementById('param-batch-slider').value;
    } else if (type === 'concurrent') {
        document.getElementById('param-concurrent').value = document.getElementById('param-concurrent-slider').value;
    }
}

function syncInput(type) {
    if (type === 'seq') {
        const v = parseInt(document.getElementById('param-seq-length').value) || 2048;
        document.getElementById('param-seq-slider').value = Math.min(v, 131072);
        highlightSeqPreset();
    } else if (type === 'batch') {
        const v = parseInt(document.getElementById('param-batch-size').value) || 1;
        document.getElementById('param-batch-slider').value = Math.min(v, 64);
    } else if (type === 'concurrent') {
        const v = parseInt(document.getElementById('param-concurrent').value) || 1;
        document.getElementById('param-concurrent-slider').value = Math.min(v, 128);
    }
}

function setSeqLength(val) {
    document.getElementById('param-seq-length').value = val;
    document.getElementById('param-seq-slider').value = val;
    highlightSeqPreset();
}

function highlightSeqPreset() {
    const val = parseInt(document.getElementById('param-seq-length').value);
    document.querySelectorAll('.param-presets button').forEach((btn) => btn.classList.remove('active'));
    const presetMap = { 512:0, 2048:1, 4096:2, 8192:3, 16384:4, 32768:5, 65536:6, 131072:7 };
    if (val in presetMap) {
        const btns = document.querySelectorAll('.param-presets button');
        if (btns[presetMap[val]]) btns[presetMap[val]].classList.add('active');
    }
}

// ---- Main analysis flow ----
async function analyzeModel() {
    const input = document.getElementById('model-input').value.trim();
    if (!input) return;

    const selectedFormats = getSelectedFormats();
    if (selectedFormats.length === 0) {
        showError('Please select at least one precision mode.');
        return;
    }

    const btn = document.getElementById('analyze-btn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const errorSection = document.getElementById('error-section');

    btn.disabled = true;
    btn.querySelector('.btn-text').textContent = 'Analyzing…';
    results.classList.add('hidden');
    errorSection.classList.add('hidden');
    loading.classList.remove('hidden');

    const params = getInferenceParams();

    try {
        const resp = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: input, formats: selectedFormats, ...params }),
        });

        const data = await resp.json();

        if (!resp.ok) {
            showError(data.error || 'Something went wrong.');
            return;
        }

        currentData = data;
        renderResults(data);
    } catch (err) {
        showError('Network error. Is the server running?');
        console.error(err);
    } finally {
        btn.disabled = false;
        btn.querySelector('.btn-text').textContent = 'Analyze Model';
        loading.classList.add('hidden');
    }
}

function showError(msg) {
    const section = document.getElementById('error-section');
    document.getElementById('error-message').textContent = msg;
    section.classList.remove('hidden');
}

// ---- Render full results ----
function renderResults(data) {
    const results = document.getElementById('results');

    // Model summary
    document.getElementById('result-model-id').textContent = data.model_id;
    document.getElementById('result-arch').textContent = data.architecture;
    document.getElementById('result-pipeline').textContent = data.pipeline_tag;
    document.getElementById('result-library').textContent = data.library;
    document.getElementById('result-params').textContent = formatParams(data.param_billions);

    // MoE / MLA tags
    const tagsEl = document.getElementById('model-tags');
    tagsEl.innerHTML = '';
    if (data.kv_info && (data.kv_info.is_moe || data.kv_info.is_mla)) {
        if (data.kv_info.is_moe) {
            const t = document.createElement('span');
            t.className = 'tag tag-moe';
            t.textContent = 'MoE' + (data.kv_info.active_param_billions ? ' · ' + data.kv_info.active_param_billions + 'B active' : '');
            tagsEl.appendChild(t);
        }
        if (data.kv_info.is_mla) {
            const t = document.createElement('span');
            t.className = 'tag tag-mla';
            t.textContent = 'MLA';
            tagsEl.appendChild(t);
        }
        tagsEl.classList.remove('hidden');
    } else {
        tagsEl.classList.add('hidden');
    }

    // Context length
    const ctxEl = document.getElementById('result-context');
    if (data.context_length) {
        document.getElementById('result-ctx-val').textContent = data.context_length.toLocaleString();
        ctxEl.classList.remove('hidden');
    } else {
        ctxEl.classList.add('hidden');
    }

    // Config warning banner
    const configWarningEl = document.getElementById('config-warning');
    if (data.config_warning) {
        document.getElementById('config-warning-text').textContent = data.config_warning;
        configWarningEl.classList.remove('hidden');
    } else {
        configWarningEl.classList.add('hidden');
    }

    // KV-cache breakdown
    if (data.kv_info) {
        const kv = data.kv_info;
        document.getElementById('kv-cache-val').textContent = kv.kv_cache_gb + ' GB';
        document.getElementById('kv-overhead-val').textContent = kv.activation_overhead_gb + ' GB';
        document.getElementById('kv-sequences-val').textContent = kv.total_sequences;
        document.getElementById('kv-arch-val').textContent =
            kv.num_layers + 'L / ' + kv.num_kv_heads + ' KV-heads / dim ' + kv.head_dim;
        document.getElementById('kv-arch-note').textContent =
            kv.arch_source === 'estimated'
                ? '⚠ Architecture estimated from parameter count (config unavailable)'
                : 'Architecture from model config';
        document.getElementById('kv-breakdown').classList.remove('hidden');
    }

    // VRAM table (already filtered by backend)
    renderVRAMTable(data.vram_table, data.gpu_compatibility);

    // GPU format selector (only selected formats)
    const select = document.getElementById('gpu-format-select');
    select.innerHTML = '';
    data.vram_table.forEach((row) => {
        const opt = document.createElement('option');
        opt.value = row.format;
        opt.textContent = row.label;
        if (row.format === 'FP16') opt.selected = true;
        select.appendChild(opt);
    });
    // Default to first if FP16 not in list
    if (!select.value && data.vram_table.length > 0) {
        select.value = data.vram_table[0].format;
    }
    renderGPUCompat();

    // LLM analysis (markdown)
    renderLLMAnalysis(data.llm_analysis);

    results.classList.remove('hidden');
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderLLMAnalysis(markdown) {
    const card = document.getElementById('llm-analysis-card');
    const content = document.getElementById('llm-analysis-content');
    if (!card || !content) return;
    if (!markdown) {
        card.classList.add('hidden');
        content.innerHTML = '';
        return;
    }
    if (typeof marked !== 'undefined' && marked.parse) {
        content.innerHTML = marked.parse(markdown);
    } else {
        // Fallback: plain text in <pre>
        const pre = document.createElement('pre');
        pre.textContent = markdown;
        content.innerHTML = '';
        content.appendChild(pre);
    }
    card.classList.remove('hidden');
    // Reset chat history each time a new analysis is rendered
    clearChat();
}

// ---- Follow-up chat ----
let chatHistory = [];

function clearChat() {
    chatHistory = [];
    const msgs = document.getElementById('llm-chat-messages');
    if (msgs) msgs.innerHTML = '';
}

function appendChatMessage(role, markdown, opts) {
    opts = opts || {};
    const msgs = document.getElementById('llm-chat-messages');
    if (!msgs) return null;
    const wrap = document.createElement('div');
    wrap.className = 'llm-chat-msg llm-chat-msg-' + role;
    if (opts.pending) wrap.classList.add('pending');
    const bubble = document.createElement('div');
    bubble.className = 'llm-chat-bubble';
    if (role === 'assistant' && typeof marked !== 'undefined' && marked.parse) {
        bubble.innerHTML = marked.parse(markdown || '');
    } else {
        bubble.textContent = markdown || '';
    }
    wrap.appendChild(bubble);
    msgs.appendChild(wrap);
    msgs.scrollTop = msgs.scrollHeight;
    return wrap;
}

async function sendChat(e) {
    if (e) e.preventDefault();
    const input = document.getElementById('llm-chat-input');
    const sendBtn = document.getElementById('llm-chat-send');
    const text = (input.value || '').trim();
    if (!text) return;
    if (!currentData) {
        appendChatMessage('assistant', '_Run an analysis first before asking follow-ups._');
        return;
    }

    // Render user message
    appendChatMessage('user', text);
    chatHistory.push({ role: 'user', content: text });
    input.value = '';
    input.disabled = true;
    sendBtn.disabled = true;

    // Pending placeholder
    const pending = appendChatMessage('assistant', '…thinking…', { pending: true });

    try {
        const resp = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: chatHistory,
                context: {
                    model_id: currentData.model_id,
                    architecture: currentData.architecture,
                    param_billions: currentData.param_billions,
                    kv_info: currentData.kv_info,
                    vram_table: currentData.vram_table,
                    llm_analysis: currentData.llm_analysis,
                },
            }),
        });
        const data = await resp.json();
        if (pending) pending.remove();
        if (!resp.ok) {
            appendChatMessage('assistant', '⚠️ ' + (data.error || 'Chat request failed.'));
            chatHistory.pop(); // remove the user message that failed
        } else {
            const reply = data.reply || '(empty response)';
            appendChatMessage('assistant', reply);
            chatHistory.push({ role: 'assistant', content: reply });
        }
    } catch (err) {
        if (pending) pending.remove();
        appendChatMessage('assistant', '⚠️ Network error talking to /api/chat.');
        console.error(err);
        chatHistory.pop();
    } finally {
        input.disabled = false;
        sendBtn.disabled = false;
        input.focus();
    }
}

// ---- VRAM Table ----
function renderVRAMTable(vramTable, gpuCompat) {
    const tbody = document.getElementById('vram-table-body');
    tbody.innerHTML = '';

    vramTable.forEach((row) => {
        const tr = document.createElement('tr');
        const fitsCount = gpuCompat[row.format]
            ? gpuCompat[row.format].filter((g) => g.fits).length
            : 0;

        tr.innerHTML = `
            <td><span class="format-name">${row.label}</span></td>
            <td>${row.bits}-bit</td>
            <td>${row.model_size_gb} GB</td>
            <td>${row.kv_cache_gb} GB</td>
            <td><span class="vram-value ${vramColor(row.vram_required_gb)}">${row.vram_required_gb} GB</span></td>
            <td class="desc-col">${row.description}</td>
            <td><span class="gpu-count-badge ${gpuCountClass(fitsCount)}">${fitsCount} VMs</span></td>
        `;
        tbody.appendChild(tr);
    });
}

function vramColor(gb) {
    if (gb <= 8) return 'vram-low';
    if (gb <= 24) return 'vram-medium';
    if (gb <= 48) return 'vram-high';
    return 'vram-extreme';
}

function gpuCountClass(count) {
    if (count >= 15) return 'gpu-many';
    if (count >= 5) return 'gpu-some';
    return 'gpu-few';
}

// ---- GPU Compatibility ----
function renderGPUCompat() {
    if (!currentData) return;

    const format = document.getElementById('gpu-format-select').value;
    const gpus = currentData.gpu_compatibility[format] || [];
    const grid = document.getElementById('gpu-grid');
    grid.innerHTML = '';

    const sorted = [...gpus].sort((a, b) => {
        if (a.fits !== b.fits) return a.fits ? -1 : 1;
        return a.vram_gb - b.vram_gb;   // smallest first for fits
    });

    sorted.forEach((gpu) => {
        const item = document.createElement('div');
        item.className = `gpu-item ${gpu.fits ? 'fits' : 'no-fit'}`;

        const statusIcon = gpu.fits ? '✓' : '✗';
        const statusClass = gpu.fits ? 'ok' : 'fail';
        const headroomText = gpu.fits
            ? `+${gpu.headroom_gb} GB free`
            : `${gpu.headroom_gb} GB short`;
        const multiGpuNote = gpu.gpus > 1 ? `<span class="gpu-multi-note">Requires tensor parallelism</span>` : '';

        let throughputHtml = '';
        if (gpu.throughput && gpu.throughput.tpm > 0) {
            const t = gpu.throughput;
            throughputHtml = `
                <div class="gpu-throughput">
                    <div class="throughput-main">${formatTPM(t.tpm)} <span class="throughput-unit">TPM</span></div>
                    <div class="throughput-details">
                        <span>${t.output_tok_per_sec} tok/s</span>
                        <span class="throughput-sep">·</span>
                        <span>${t.per_user_tok_per_sec} tok/s/user</span>
                        <span class="throughput-sep">·</span>
                        <span class="throughput-bottleneck">${t.bottleneck}-bound</span>
                    </div>
                </div>`;
        }

        const benchBtnHtml = (gpu.fits && BENCHMARK_ENABLED_VMS.has(gpu.name))
            ? `<button class="bench-run-btn" onclick="openBenchModal('${gpu.name.replace(/'/g, "\\'")}')">
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
                      <polygon points="5 3 19 12 5 21 5 3"/>
                  </svg>
                  Run Benchmark
               </button>`
            : '';

        item.innerHTML = `
            <div class="gpu-info">
                <div class="gpu-name">${gpu.name}</div>
                <div class="gpu-config">${gpu.gpu} — ${gpu.vram_gb} GB total</div>
                <span class="gpu-series-badge">${gpu.series}</span>
                ${multiGpuNote}
                ${throughputHtml}
                ${benchBtnHtml}
            </div>
            <div class="gpu-status ${statusClass}">
                ${statusIcon} ${headroomText}
            </div>
        `;
        grid.appendChild(item);
    });
}

// ---- Helpers ----
function formatParams(billions) {
    if (billions >= 1) return `${billions}B`;
    const millions = Math.round(billions * 1000);
    return `${millions}M`;
}

function formatTPM(tpm) {
    if (tpm >= 1_000_000) return (tpm / 1_000_000).toFixed(1) + 'M';
    if (tpm >= 1_000) return (tpm / 1_000).toFixed(1) + 'K';
    return tpm.toLocaleString();
}

// =========================================
//  Benchmark Modal
// =========================================
const BENCHMARK_ENABLED_VMS = new Set(['NC40ads H100 v5', 'NC80adis H100 v5']);

function openBenchModal(gpuName) {
    if (!currentData) return;
    const modelId = currentData.model_id;
    document.getElementById('bench-target-vm').textContent = gpuName;
    document.getElementById('bench-model').value = modelId;
    document.getElementById('bench-tokenizer').value = modelId;
    document.getElementById('bench-dataset').value = 'random';
    document.getElementById('bench-result').classList.add('hidden');
    document.getElementById('bench-result').textContent = '';

    const submitBtn = document.getElementById('bench-submit-btn');
    submitBtn.disabled = false;
    submitBtn.textContent = 'Run Benchmark';
    submitBtn.dataset.gpuSku = gpuName;

    updateBenchDatasetFields();

    const modal = document.getElementById('bench-modal');
    modal.classList.remove('hidden');
    modal.setAttribute('aria-hidden', 'false');
}

function closeBenchModal() {
    const modal = document.getElementById('bench-modal');
    modal.classList.add('hidden');
    modal.setAttribute('aria-hidden', 'true');
}

function updateBenchDatasetFields() {
    const ds = document.getElementById('bench-dataset').value;
    document.getElementById('bench-random-fields').classList.toggle('hidden', ds !== 'random');
    document.getElementById('bench-sharegpt-fields').classList.toggle('hidden', ds !== 'sharegpt');
    refreshBenchCmdPreview();
}

function collectBenchPayload() {
    const ds = document.getElementById('bench-dataset').value;
    const submitBtn = document.getElementById('bench-submit-btn');
    const payload = {
        gpu_sku: submitBtn.dataset.gpuSku,
        model: document.getElementById('bench-model').value,
        tokenizer: document.getElementById('bench-tokenizer').value,
        dataset_name: ds,
        num_prompts: parseInt(document.getElementById('bench-num-prompts').value) || 1000,
        max_concurrency: parseInt(document.getElementById('bench-concurrency').value) || 32,
        seed: 42,
    };
    if (ds === 'random') {
        payload.random_input_len = parseInt(document.getElementById('bench-rand-input').value) || 1024;
        payload.random_output_len = parseInt(document.getElementById('bench-rand-output').value) || 128;
        payload.random_prefix_len = parseInt(document.getElementById('bench-rand-prefix').value) || 0;
        payload.random_range_ratio = parseFloat(document.getElementById('bench-rand-ratio').value) || 0.0;
    } else {
        payload.dataset_path = document.getElementById('bench-dataset-path').value
            || './ShareGPT_V3_unfiltered_cleaned_split.json';
    }
    return payload;
}

function buildBenchCommand(p) {
    const parts = [
        'vllm bench serve',
        `--base-url http://localhost:8000`,
        `--model ${p.model}`,
        `--tokenizer ${p.tokenizer}`,
        `--dataset-name ${p.dataset_name}`,
    ];
    if (p.dataset_name === 'random') {
        parts.push(`--random-input-len ${p.random_input_len}`);
        parts.push(`--random-output-len ${p.random_output_len}`);
        parts.push(`--random-prefix-len ${p.random_prefix_len}`);
        parts.push(`--random-range-ratio ${p.random_range_ratio}`);
    } else {
        parts.push(`--dataset-path ${p.dataset_path}`);
    }
    parts.push(`--num-prompts ${p.num_prompts}`);
    parts.push(`--max-concurrency ${p.max_concurrency}`);
    parts.push(`--seed ${p.seed}`);
    return parts.join(' \\\n    ');
}

function refreshBenchCmdPreview() {
    const preview = document.getElementById('bench-cmd-preview');
    if (!preview) return;
    try {
        preview.textContent = buildBenchCommand(collectBenchPayload());
    } catch (_) { /* ignore until modal fully rendered */ }
}

async function submitBenchmark() {
    const payload = collectBenchPayload();
    const submitBtn = document.getElementById('bench-submit-btn');
    const resultEl = document.getElementById('bench-result');
    submitBtn.disabled = true;
    submitBtn.textContent = 'Submitting…';
    resultEl.classList.remove('hidden');
    resultEl.className = 'bench-result info';
    resultEl.textContent = 'Submitting benchmark job to Kubernetes…';

    try {
        const resp = await fetch('/api/benchmark', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await resp.json();
        if (!resp.ok) {
            resultEl.className = 'bench-result error';
            resultEl.textContent = data.error || 'Failed to submit benchmark.';
            submitBtn.disabled = false;
            submitBtn.textContent = 'Run Benchmark';
            return;
        }
        resultEl.className = 'bench-result success';
        resultEl.innerHTML =
            `<strong>Benchmark submitted.</strong><br>` +
            `Run ID: <code>${data.run_id}</code><br>` +
            `Namespace: <code>${data.namespace}</code><br>` +
            `Deployment: <code>${data.deployment}</code><br>` +
            `Service: <code>${data.service}</code><br>` +
            `Job: <code>${data.job}</code><br>` +
            `<br>Tail logs: <code>kubectl logs -n ${data.namespace} -f job/${data.job}</code>`;
        submitBtn.textContent = 'Submitted';
    } catch (err) {
        console.error(err);
        resultEl.className = 'bench-result error';
        resultEl.textContent = 'Network error. Is the server running?';
        submitBtn.disabled = false;
        submitBtn.textContent = 'Run Benchmark';
    }
}

// Refresh command preview as user edits inputs
document.addEventListener('input', (e) => {
    if (e.target && e.target.closest && e.target.closest('#bench-modal')) {
        refreshBenchCmdPreview();
    }
});

// Esc closes modal
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        const modal = document.getElementById('bench-modal');
        if (modal && !modal.classList.contains('hidden')) closeBenchModal();
    }
});

// ============== View routing (sidebar) ==============
function showView(name, ev) {
    if (ev) ev.preventDefault();
    document.querySelectorAll('.sidebar-link').forEach(a => {
        a.classList.toggle('active', a.dataset.view === name);
    });
    const cal = document.getElementById('view-calculator');
    const res = document.getElementById('view-results');
    if (name === 'results') {
        cal.classList.add('hidden'); cal.classList.remove('view-active');
        res.classList.remove('hidden'); res.classList.add('view-active');
        showResultsList();
        loadBenchmarkRuns();
        startResultsAutoRefresh();
    } else {
        res.classList.add('hidden'); res.classList.remove('view-active');
        cal.classList.remove('hidden'); cal.classList.add('view-active');
        stopResultsAutoRefresh();
    }
}

// ============== Signed-in user ==============
async function loadCurrentUser() {
    try {
        const r = await fetch('/api/me');
        const d = await r.json();
        const el = document.getElementById('sidebar-user-name');
        if (el) el.textContent = d.user || 'anonymous';
    } catch (_) {
        const el = document.getElementById('sidebar-user-name');
        if (el) el.textContent = 'anonymous';
    }
}

// ============== Results list ==============
let _resultsRefreshTimer = null;

function startResultsAutoRefresh() {
    stopResultsAutoRefresh();
    _resultsRefreshTimer = setInterval(() => {
        const detail = document.getElementById('results-detail-view');
        if (detail && !detail.classList.contains('hidden')) {
            const rid = detail.dataset.runId;
            if (rid) loadBenchmarkDetail(rid, /*silent*/true);
        } else {
            loadBenchmarkRuns(/*silent*/true);
        }
    }, 5000);
}
function stopResultsAutoRefresh() {
    if (_resultsRefreshTimer) { clearInterval(_resultsRefreshTimer); _resultsRefreshTimer = null; }
}

function _stateBadgeHtml(state) {
    const s = (state || 'unknown').toLowerCase();
    return `<span class="state-badge state-${s}">${s}</span>`;
}

function _fmtTime(iso) {
    if (!iso) return '�';
    try {
        const d = new Date(iso);
        return d.toLocaleString();
    } catch (_) { return iso; }
}

async function loadBenchmarkRuns(silent) {
    const tbody = document.getElementById('results-table-body');
    if (!silent && tbody) tbody.innerHTML = '<tr><td colspan="6" class="results-empty">Loading�</td></tr>';
    try {
        const r = await fetch('/api/benchmarks');
        const data = await r.json();
        if (!r.ok) {
            tbody.innerHTML = `<tr><td colspan="6" class="results-empty error">${data.error || 'Failed to load runs.'}</td></tr>`;
            return;
        }
        const runs = data.runs || [];
        if (!runs.length) {
            tbody.innerHTML = '<tr><td colspan="6" class="results-empty">No benchmark runs yet.</td></tr>';
        } else {
            tbody.innerHTML = runs.map(run => `
                <tr class="results-row" onclick="openBenchmarkDetail('${run.run_id}')">
                    <td><code>${run.run_id}</code></td>
                    <td>${_stateBadgeHtml(run.state)}</td>
                    <td title="${run.model || ''}">${run.model || '�'}</td>
                    <td>${run.gpu_sku || '�'}</td>
                    <td>${run.requestor || 'unknown'}</td>
                    <td>${_fmtTime(run.submitted_at)}</td>
                </tr>
            `).join('');
        }
        const stamp = document.getElementById('results-last-refresh');
        if (stamp) stamp.textContent = 'Last updated: ' + new Date().toLocaleTimeString();
    } catch (err) {
        if (!silent) tbody.innerHTML = '<tr><td colspan="6" class="results-empty error">Network error loading runs.</td></tr>';
    }
}

function showResultsList() {
    document.getElementById('results-list-view').classList.remove('hidden');
    document.getElementById('results-detail-view').classList.add('hidden');
}
function showResultsDetail() {
    document.getElementById('results-list-view').classList.add('hidden');
    document.getElementById('results-detail-view').classList.remove('hidden');
}

function openBenchmarkDetail(runId) {
    const detail = document.getElementById('results-detail-view');
    detail.dataset.runId = runId;
    document.getElementById('detail-run-id').textContent = runId;
    document.getElementById('detail-state-badge').outerHTML =
        `<span id="detail-state-badge" class="state-badge">loading�</span>`;
    document.getElementById('detail-meta').innerHTML = '';
    document.getElementById('detail-error-block').classList.add('hidden');
    document.getElementById('detail-error').textContent = '';
    document.getElementById('detail-logs').textContent = 'Loading�';
    showResultsDetail();
    loadBenchmarkDetail(runId);
}

async function loadBenchmarkDetail(runId, silent) {
    try {
        const r = await fetch(`/api/benchmark/${encodeURIComponent(runId)}`);
        const d = await r.json();
        if (!r.ok) {
            document.getElementById('detail-logs').textContent = d.error || 'Failed to load.';
            return;
        }
        // State badge
        const badge = document.getElementById('detail-state-badge');
        if (badge) {
            const s = (d.state || 'unknown').toLowerCase();
            badge.className = `state-badge state-${s}`;
            badge.textContent = s;
        }
        // Meta
        const metaEl = document.getElementById('detail-meta');
        const metaItems = [
            ['Model', d.model],
            ['GPU SKU', d.gpu_sku],
            ['Dataset', d.dataset],
            ['Requestor', d.requestor],
            ['Submitted', _fmtTime(d.submitted_at)],
            ['Namespace', d.namespace],
            ['Job', d.job && d.job.name],
            ['Pod', d.pod && d.pod.name],
        ].filter(([, v]) => v);
        metaEl.innerHTML = metaItems.map(([k, v]) =>
            `<div class="meta-item"><span class="meta-key">${k}</span><span class="meta-val">${v}</span></div>`
        ).join('');

        // Error
        const errBlock = document.getElementById('detail-error-block');
        if (d.error) {
            errBlock.classList.remove('hidden');
            document.getElementById('detail-error').textContent = d.error;
        } else {
            errBlock.classList.add('hidden');
        }

        // Logs
        const logsEl = document.getElementById('detail-logs');
        if (d.logs && d.logs.trim()) {
            logsEl.textContent = d.logs;
        } else if (d.logs_error) {
            logsEl.textContent = `(logs not available: ${d.logs_error})`;
        } else if (d.state === 'provisioning') {
            logsEl.textContent = 'Waiting for vLLM server to become ready�';
        } else {
            logsEl.textContent = '(no output)';
        }
    } catch (err) {
        if (!silent) document.getElementById('detail-logs').textContent = 'Network error.';
    }
}

// Init
document.addEventListener('DOMContentLoaded', loadCurrentUser);
