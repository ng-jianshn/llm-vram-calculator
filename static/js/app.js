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

    results.classList.remove('hidden');
    results.scrollIntoView({ behavior: 'smooth', block: 'start' });
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

        item.innerHTML = `
            <div class="gpu-info">
                <div class="gpu-name">${gpu.name}</div>
                <div class="gpu-config">${gpu.gpu} — ${gpu.vram_gb} GB total</div>
                <span class="gpu-series-badge">${gpu.series}</span>
                ${multiGpuNote}
                ${throughputHtml}
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
