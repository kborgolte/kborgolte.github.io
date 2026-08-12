// Oberfläche: Status, Jobliste, Upload, Auswahl einer Szene.

const $ = (id) => document.getElementById(id);

let viewer = null;        // wird beim ersten Öffnen einer Szene erzeugt
let selectedJob = null;
let lastSignature = '';   // verhindert unnötiges Neuzeichnen der Liste

const STATUS_LABEL = {
  queued: 'wartet',
  running: 'läuft',
  done: 'fertig',
  failed: 'fehlgeschlagen',
  cancelled: 'abgebrochen',
};

const formatNumber = (value) => (value ?? 0).toLocaleString('de-DE');

function formatDuration(seconds) {
  if (!seconds && seconds !== 0) return '–';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${Math.round(seconds % 60)}s`;
}

// ── Status oben rechts ──────────────────────────────────────────────────────

async function refreshStatus() {
  try {
    const status = await (await fetch('/api/status')).json();

    const device = $('device-badge');
    device.textContent = status.device.label;
    device.className = 'badge ' + (status.device.device === 'cpu' ? 'warn' : 'ok');
    device.title = status.device.note || '';

    const model = $('model-badge');
    const ready = status.model_ready && status.lingbot_ready;
    model.textContent = ready ? 'Modell bereit' : 'Modell fehlt';
    model.className = 'badge ' + (ready ? 'ok' : 'err');
    if (!ready) model.title = 'setup.sh ausführen — ' + status.model_path;

    $('mobile-url').textContent = status.mobile_url;

    const queue = $('queue-badge');
    queue.hidden = status.queue_depth === 0;
    queue.textContent = `${status.queue_depth} in der Warteschlange`;
  } catch {
    $('device-badge').textContent = 'Server nicht erreichbar';
    $('device-badge').className = 'badge err';
  }
}

async function loadQr() {
  const box = $('qr');
  try {
    const response = await fetch('/api/qr');
    box.innerHTML = response.ok
      ? await response.text()
      : '<span style="color:#555;font-size:12px;padding:20px">QR nicht verfügbar</span>';
  } catch {
    box.textContent = '—';
  }
}

// ── Jobliste ────────────────────────────────────────────────────────────────

function renderJobs(jobs) {
  const container = $('jobs');

  if (!jobs.length) {
    container.innerHTML = '<p style="color:var(--muted);font-size:13px">Noch nichts aufgenommen.</p>';
    return;
  }

  container.innerHTML = jobs.map((job) => {
    const active = job.id === selectedJob ? ' active' : '';
    const busy = job.status === 'running' || job.status === 'queued';

    const points = job.scene?.points
      ? `<span class="badge">${formatNumber(job.scene.points)} Punkte</span>` : '';
    const progressBar = busy
      ? `<div class="bar"><span style="width:${job.progress}%"></span></div>` : '';
    const error = job.error
      ? `<div class="error">${escapeHtml(job.error)}</div>` : '';
    const action = busy
      ? `<button class="ghost" data-cancel="${job.id}" title="Abbrechen">✕</button>`
      : `<button class="ghost" data-delete="${job.id}" title="Löschen">🗑</button>`;

    return `
      <div class="job ${job.status}${active}" data-job="${job.id}">
        <div class="job-head">
          ${busy ? '<div class="spin"></div>' : ''}
          <span class="job-name" title="${escapeHtml(job.name)}">${escapeHtml(job.name)}</span>
          ${action}
        </div>
        <div class="job-stage">${escapeHtml(job.stage)} · ${STATUS_LABEL[job.status] || job.status}</div>
        ${progressBar}
        <div style="margin-top:7px;display:flex;gap:5px;flex-wrap:wrap">
          ${points}
          ${job.duration ? `<span class="badge">${formatDuration(job.duration)}</span>` : ''}
        </div>
        ${error}
      </div>`;
  }).join('');
}

function escapeHtml(value) {
  const div = document.createElement('div');
  div.textContent = value ?? '';
  return div.innerHTML;
}

async function refreshJobs() {
  let jobs;
  try {
    jobs = (await (await fetch('/api/jobs')).json()).jobs;
  } catch {
    return;
  }

  // Nur neu zeichnen, wenn sich wirklich etwas geändert hat — sonst verliert
  // man bei jedem Tick den Hover-Zustand und die Liste flackert.
  const signature = jobs.map((j) => `${j.id}:${j.status}:${j.progress}:${j.stage}`).join('|')
    + `|${selectedJob}`;
  if (signature !== lastSignature) {
    lastSignature = signature;
    renderJobs(jobs);
  }

  // Frisch fertig gewordene Szene automatisch öffnen — man wartet ja darauf.
  const finished = jobs.find((job) => job.status === 'done');
  if (finished && selectedJob === null) openScene(finished);
}

// ── Szene öffnen ────────────────────────────────────────────────────────────

async function openScene(job) {
  if (job.status !== 'done') return;

  selectedJob = job.id;
  lastSignature = '';

  $('hint').innerHTML = '<div class="spin"></div><span>Punktwolke wird geladen …</span>';
  $('hint').hidden = false;

  try {
    if (!viewer) {
      const { SceneViewer } = await import('/static/viewer.js');
      viewer = new SceneViewer($('stage'));
    }

    const count = await viewer.load(job.id, job.scene);

    $('hint').hidden = true;
    $('scene-info').hidden = false;
    $('controls').hidden = false;

    $('scene-title').textContent = job.name;
    $('i-points').textContent = formatNumber(count);
    $('i-frames').textContent = job.scene?.frames ?? '–';
    $('i-device').textContent = job.device?.label ?? '–';
    $('i-duration').textContent = formatDuration(job.duration);
    $('download-glb').href = `/api/scene/${job.id}/scene.glb`;
    $('download-glb').hidden = !job.has_glb;
  } catch (error) {
    $('hint').innerHTML =
      `<strong>Szene lässt sich nicht anzeigen</strong><span>${escapeHtml(error.message)}</span>` +
      (error.message.includes('Failed to fetch dynamically imported module')
        ? '<span style="font-size:13px">three.js fehlt im Ordner static/vendor. '
          + 'Einmal <code>./setup.sh</code> laufen lassen.</span>'
        : '');
  }
}

// ── Upload vom Notebook ─────────────────────────────────────────────────────

async function uploadFile(file) {
  if (!file) return;

  const zone = $('dropzone');
  const original = zone.innerHTML;
  zone.innerHTML = `<div>${escapeHtml(file.name)} wird übertragen …</div>`;

  const body = new FormData();
  body.append('file', file);

  try {
    const response = await fetch('/api/upload', { method: 'POST', body });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      zone.innerHTML = `<div style="color:var(--err)">${escapeHtml(payload.detail || 'Upload fehlgeschlagen')}</div>`;
      setTimeout(() => { zone.innerHTML = original; }, 4000);
      return;
    }
    zone.innerHTML = original;
    lastSignature = '';
    refreshJobs();
  } catch {
    zone.innerHTML = '<div style="color:var(--err)">Upload fehlgeschlagen</div>';
    setTimeout(() => { zone.innerHTML = original; }, 4000);
  }
}

function wireUpload() {
  const zone = $('dropzone');
  const input = $('file-input');

  input.addEventListener('change', (event) => uploadFile(event.target.files[0]));

  for (const type of ['dragenter', 'dragover']) {
    zone.addEventListener(type, (event) => {
      event.preventDefault();
      zone.classList.add('hot');
    });
  }
  for (const type of ['dragleave', 'drop']) {
    zone.addEventListener(type, () => zone.classList.remove('hot'));
  }
  zone.addEventListener('drop', (event) => {
    event.preventDefault();
    uploadFile(event.dataTransfer.files[0]);
  });
}

// ── Verdrahtung ─────────────────────────────────────────────────────────────

$('jobs').addEventListener('click', async (event) => {
  const cancelId = event.target.dataset?.cancel;
  const deleteId = event.target.dataset?.delete;

  if (cancelId) {
    event.stopPropagation();
    await fetch(`/api/jobs/${cancelId}/cancel`, { method: 'POST' });
    lastSignature = '';
    return refreshJobs();
  }

  if (deleteId) {
    event.stopPropagation();
    await fetch(`/api/jobs/${deleteId}`, { method: 'DELETE' });
    if (selectedJob === deleteId) {
      selectedJob = null;
      viewer?.clear();
      $('hint').hidden = false;
      $('scene-info').hidden = true;
      $('controls').hidden = true;
    }
    lastSignature = '';
    return refreshJobs();
  }

  const card = event.target.closest('[data-job]');
  if (!card) return;

  const jobs = (await (await fetch('/api/jobs')).json()).jobs;
  const job = jobs.find((candidate) => candidate.id === card.dataset.job);
  if (job) openScene(job);
});

$('point-size').addEventListener('input', (event) => viewer?.setPointSize(+event.target.value));
$('show-path').addEventListener('change', (event) => viewer?.setPathVisible(event.target.checked));
$('reset-view').addEventListener('click', () => viewer?.resetView());

wireUpload();
loadQr();
refreshStatus();
refreshJobs();
setInterval(refreshJobs, 1500);
setInterval(refreshStatus, 10000);
