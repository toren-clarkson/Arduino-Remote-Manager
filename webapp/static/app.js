'use strict';

const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const enc = encodeURIComponent;
const fileURL = (path, inline = false) => `/api/file?path=${enc(path)}${inline ? '&inline=1' : ''}`;
const filesURL = path => `/files?path=${enc(path)}`;
const jobURL = id => `/jobs?id=${enc(id)}`;
const outputURL = path => /\.[a-z0-9]+$/i.test(path) ? fileURL(path) : filesURL(path);
const artifactURL = (id, name, inline = false) => `/api/jobs/${id}/artifact?name=${enc(name)}${inline ? '&inline=1' : ''}`;
const isActive = job => ['starting','running','stop_requested'].includes(job.status);
const size = value => value >= 1024**3 ? `${(value/1024**3).toFixed(1)} GB` : value >= 1024**2 ? `${(value/1024**2).toFixed(1)} MB` : value >= 1024 ? `${(value/1024).toFixed(1)} KB` : `${value} B`;
const date = value => new Date(value).toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
const status = value => `<span class="badge ${esc(value)}">${esc(value.replaceAll('_',' '))}</span>`;
const empty = (title, text='') => `<div class="empty"><strong>${esc(title)}</strong>${esc(text)}</div>`;
const page = location.pathname.slice(1) || 'home';
let state;
let chosenJob = new URLSearchParams(location.search).get('id');
let editor = null;
let toastTimer;
let polling = false;

async function api(path, options = {}) {
  const headers = {'X-CSRF-Token': $('meta[name="csrf-token"]').content, ...options.headers};
  if (options.body && !(options.body instanceof FormData)) { headers['Content-Type'] = 'application/json'; options.body = JSON.stringify(options.body); }
  const response = await fetch(path, {...options, headers});
  let data;
  try { data = await response.json(); } catch { throw new Error(`Server returned HTTP ${response.status}.`); }
  if (!response.ok) throw new Error(data.error || `Request failed: ${response.status}`);
  return data;
}

function toast(message, error=false) {
  clearTimeout(toastTimer);
  const node = $('#toast'); node.textContent = message; node.className = error ? 'error' : ''; node.hidden = false;
  toastTimer = setTimeout(() => node.hidden = true, error ? 12000 : 6500);
}

function modal(title, html, wide=false) {
  $('#dialog-title').textContent = title; $('#dialog-body').innerHTML = html;
  $('#dialog').classList.toggle('wide', wide); if (!$('#dialog').open) $('#dialog').showModal();
  addFieldHelp($('#dialog'));
}

function heading(title, subtitle, action='', eyebrow='MAIN WORKSPACE') {
  $('#breadcrumb').textContent = title; document.title = `${title} · Main`;
  return `<div class="page-title"><div><div class="eyebrow">${esc(eyebrow)}</div><h1>${esc(title)}</h1><p class="subtitle">${esc(subtitle)}</p></div>${action ? `<div class="actions">${action}</div>` : ''}</div>`;
}

function docs(path, label='Read the guide') { return `<button class="secondary" data-action="read" data-path="${esc(path)}">${esc(label)} ↗</button>`; }
function panel(title, subtitle, html) { return `<section class="panel"><div class="panel-header"><div><h2>${esc(title)}</h2>${subtitle ? `<p>${esc(subtitle)}</p>` : ''}</div></div><div class="panel-body">${html}</div></section>`; }
function options(items, selected) { return items.map(item => `<option value="${esc(item.value)}" ${item.value === selected ? 'selected' : ''}>${esc(item.label)}</option>`).join(''); }

function uploadForm(folder='datasets/uploads', accept='') {
  return `<form data-form="upload" class="upload"><label class="field"><span>Destination folder</span><input name="folder" required value="${esc(folder)}" aria-label="Upload destination"></label><label class="field"><span>Choose images, videos, weights or a dataset ZIP</span><input type="file" name="files" multiple required ${accept ? `accept="${esc(accept)}"` : ''}></label><button class="primary" type="submit">Upload files ↑</button><span class="inline-help">Up to 2 GB per request. Existing files are never overwritten.</span></form>`;
}

function jobField(field) {
  const value = field.default ?? '';
  let input;
  const models = state.models.map(m => ({value:m.path, label:`${m.label || m.name} · ${m.provider}`}));
  const media = state.media.filter(m => field.kind === 'video' ? m.video : field.kind === 'image' ? m.image : field.kind === 'archive' ? m.path.endsWith('.zip') : m.image || m.video).map(m => ({value:m.path,label:m.path}));
  if (field.kind === 'select') input = `<select name="${field.key}" required>${options(field.options.map(v => ({value:v,label:v})), value)}</select>`;
  else if (['model','video','image','media','dataset','archive'].includes(field.kind)) {
    const choices = field.kind === 'model' ? models : field.kind === 'dataset' ? state.datasets.map(d=>({value:d.path,label:d.path})) : media;
    input = `<select name="${field.key}" required><option value="">Select ${esc(field.kind)}…</option>${options(choices, value)}</select>`;
  } else input = `<input name="${field.key}" type="${field.kind === 'number' ? 'number' : 'text'}" value="${esc(value)}" required ${field.kind === 'number' ? `step="${field.integer ? '1' : 'any'}" min="${field.min ?? ''}" max="${field.max ?? ''}"` : ''}>`;
  return `<label class="field"><span>${esc(field.label)} ${helpIcon(field.help)}</span>${input}</label>`;
}

function jobForms(group) {
  return Object.entries(state.catalog.jobs).filter(([id, spec]) => spec.group === group && id !== 'mission').map(([id,spec]) => panel(spec.label, `Outputs: ${spec.outputs.join(' · ')}`, `<form data-form="job" data-kind="${id}"><div class="form-grid">${spec.fields.map(jobField).join('')}</div><div class="form-actions"><button class="primary" type="submit">Start job →</button><span class="muted">Runs in a separate Python process.</span></div></form>`)).join('');
}

function jobsTable(jobs, limit=100) {
  if (!jobs.length) return empty('No jobs yet', 'Start a training, inference, tag or mission job to see it here.');
  return `<div class="table-wrap"><table><thead><tr><th>Job</th><th>Status</th><th>Started</th><th>Controls</th></tr></thead><tbody>${jobs.slice(0,limit).map(j=>`<tr><td><a href="${jobURL(j.id)}" class="file-name">${esc(j.label)}</a><small class="mono">${esc(j.id.slice(0,8))} · ${esc(j.kind)}</small></td><td>${status(j.status)}</td><td>${date(j.created)}</td><td><div class="actions"><a href="${jobURL(j.id)}">View log →</a>${j.kind==='mission' ? `<a href="${liveURL(j.id)}">Monitor →</a>` : ''}${isActive(j) ? `<button class="text-button danger-text" data-action="stop" data-id="${j.id}">Stop</button>` : ''}</div></td></tr>`).join('')}</tbody></table></div>`;
}

const folderDescriptions = {datasets:'Images, videos and labeled YOLO exports.', models:'Weight files, names, notes and model classes.', training:'Frame extraction, dataset import and training tools.', missions:'Mission controllers, configuration and guides.', demos:'Earlier color, YOLO, AprilTag and motor experiments.', tools:'Printable AprilTag generation.', assets:'Markers and other project media.', runs:'Training results, mission logs and snapshots.', tests:'Hardware-free checks for project behavior.', robotics_lab:'Shared motion, navigation and path helpers.', docs:'Project documentation and file map.', webapp:'This control room, its plan and extension guide.'};

const folderPages={datasets:'/datasets',models:'/models',training:'/training',missions:'/missions',runs:'/jobs',assets:'/tags',tools:'/tags'};

function home() {
  return heading('Your robotics workspace', 'One place to prepare data, train models, configure missions and inspect every run.', docs('webapp/README.md','App guide')) +
  `<section class="hero"><div><div class="eyebrow">FROM DATA TO A ROBOT MISSION</div><h2>Prepare. Train. Observe. Move.</h2><p>Build a model library, set up your markers, then run controlled stop-and-stare missions. Every action has a job, a configuration and a log.</p><div class="actions"><a class="button" href="/missions">Configure a mission →</a><a class="button secondary" href="/training">Train a model</a></div></div><div class="route-diagram"><div><span class="route-number">01</span><span><strong>Prepare the scene</strong>Dataset + model + AprilTag</span></div><div><span class="route-number">02</span><span><strong>Choose the behavior</strong>Mission + tracker + settings</span></div><div><span class="route-number">03</span><span><strong>Run & review</strong>Preview, logs and artifacts</span></div></div></section>` +
  `<div class="stats">${[['MODELS',state.models.length],['DATASET CONFIGS',state.datasets.length],['APRILTAG IMAGES',state.tags.length],['ACTIVE JOBS',state.jobs.filter(isActive).length]].map(([label,value])=>`<div class="stat"><span class="stat-label">${label}</span><strong>${value}</strong></div>`).join('')}</div>` +
  `<div class="section-heading"><div><h2>Project directories</h2><p>Browse the actual folders inside Main.</p></div><a href="/files">All files →</a></div><div class="grid">${state.folders.map(f=>`<a class="card folder-card" href="${folderPages[f.name] || filesURL(f.path)}"><div class="folder-top"><span class="folder-glyph">▱</span><span class="muted">↗</span></div><h3>${esc(f.name)}</h3><p>${esc(folderDescriptions[f.name] || 'Browse and manage files in this directory.')}</p><span class="folder-path">Main / ${esc(f.path)}</span></a>`).join('')}</div><div class="section-heading"><h2>Recent jobs</h2><a href="/jobs">All jobs →</a></div><section class="panel">${jobsTable(state.jobs,5)}</section>`;
}

function datasetsPage() {
  return heading('Datasets', 'Collect footage and images, extract frames and import labeled training data.', docs('training/README.md'))+
  `<div class="notice">A tracking video is optional. Train the detector on labeled images; ByteTrack and BoT-SORT associate its detections without a separate tracker-training video. Use Roboflow to annotate and export YOLO detection labels.</div>`+
  panel('Upload data', 'Keep each collection in its own folder. Upload a ZIP for a complete exported dataset.',uploadForm())+
  `<div class="grid two">${Object.entries(state.catalog.jobs).filter(([,s])=>s.group==='datasets').map(([id,spec])=>`<div>${panel(spec.label, spec.outputs.join(' · '), `<form data-form="job" data-kind="${id}"><div class="form-grid">${spec.fields.map(jobField).join('')}</div><div class="form-actions"><button class="primary">Start job →</button></div></form>`)}</div>`).join('')}</div>`+
  panel('Available dataset configurations', 'Inspect or edit data.yaml before training.', state.datasets.length ? `<div class="table-wrap"><table><thead><tr><th>Configuration</th><th>Manage</th></tr></thead><tbody>${state.datasets.map(d=>`<tr><td class="mono">${esc(d.path)}</td><td><div class="actions"><button class="text-button" data-action="edit" data-path="${esc(d.path)}">Edit YAML</button><a href="${filesURL(d.path.split('/').slice(0,-1).join('/'))}">Browse dataset →</a></div></td></tr>`).join('')}</tbody></table></div>` : empty('No labeled datasets yet', 'Import an export ZIP or download a Roboflow version.'));
}

function modelsPage() {
  return heading('Model library', 'Manage weights from multiple providers. Capabilities determine which jobs and missions can use them.', docs('webapp/EXTENDING.md','Providers & extensions'))+
  `<div class="notice small">Load weights only from sources you trust. Ultralytics models can run detection, segmentation, pose, OBB and classification inference. Navigation requires compatible boxes; other artifacts can be cataloged for custom executors.</div>`+
  panel('Add model weights', 'Upload .pt, .onnx, .engine, .tflite, .pth, .bin or .safetensors artifacts.', uploadForm('models', '.pt,.onnx,.engine,.tflite,.pth,.bin,.safetensors'))+
  `<div class="section-heading"><h2>Available models <span class="chip">${state.models.length}</span></h2><input id="model-filter" class="search-input" placeholder="Filter models…" aria-label="Filter models"></div>`+
  `<div class="grid two" id="model-grid">${state.models.map(m=>`<article class="card model-card" data-filter="${esc(`${m.name} ${m.label || ''} ${m.provider} ${m.inspection?.task || ''}`.toLowerCase())}"><div class="folder-top"><span class="badge neutral">${esc(m.provider)}</span><span class="muted">${size(m.bytes)}</span></div><h2>${esc(m.label || m.name)}</h2><span class="source-path">${esc(m.path)}</span><p class="subsection">${esc(m.notes || 'No notes yet. Inspect this model to discover its task and class names.')}</p>${m.inspection ? `<span class="badge">${esc(m.inspection.task)}</span><div class="model-classes subsection">${Object.values(m.inspection.classes || {}).map(c=>`<span class="chip">${esc(c)}</span>`).join('')}</div>` : ''}<div class="card-footer"><div class="actions"><button class="text-button" data-action="model-edit" data-path="${esc(m.path)}">Details</button><button class="text-button" data-action="inspect" data-path="${esc(m.path)}">Inspect</button><button class="text-button" data-action="rename" data-path="${esc(m.path)}">Rename file</button></div><button class="text-button danger-text" data-action="trash" data-path="${esc(m.path)}">Trash</button></div></article>`).join('') || empty('No models yet','Upload a model weight file.')}</div>`+
  `<div class="section-heading"><h2>Model jobs</h2></div><div class="grid two">${jobForms('models')}</div>`;
}

function trainingPage() {
  return heading('Training', 'Train a local YOLO model, then promote its best weights into the model library.', docs('training/README.md'))+
  `<div class="notice">Choose labeled YOLO data and .pt weights matching the selected task: detection, segmentation, pose or OBB. Training saves metrics, plots and checkpoints under runs/training; the best weights are copied to your chosen model filename.</div><div class="split"><div>${jobForms('training')}</div><div>${panel('Before you train','A short preparation checklist.',`<p><strong>01 · Collect varied images</strong><br>Include different positions, lighting, distances and backgrounds.</p><p><strong>02 · Label and split</strong><br>Annotate the exact classes you want. Keep validation images separate from training images.</p><p><strong>03 · Import your export</strong><br>Use Datasets to import labeled YOLO data. Unlabeled video frames are not ready for training.</p><p><strong>04 · Inspect and replay</strong><br>Inspect the resulting classes, then test inference or a video replay before a physical mission.</p><a href="/datasets">Manage datasets →</a>`)}</div></div><div class="section-heading"><h2>Training history</h2><a href="${filesURL('runs/training')}">Training files →</a></div><section class="panel">${jobsTable(state.jobs.filter(j=>j.kind==='train'))}</section>`;
}

function tagsPage() {
  return heading('AprilTags', 'Generate markers, verify detection and prepare a tag destination for your mission.', docs('assets/tags/README.md'))+
  `<div class="notice">The mission uses tag36h11, IDs 0–586. Keep the full white border visible, use a flat matte print, and select the same ID in Missions. Tag arrival uses image width fraction, not a measured distance.</div>`+
  `<div class="grid two">${jobForms('tags')}</div>`+
  panel('Upload a marker or test photo', 'Photos uploaded here also appear in detector-test inputs.', uploadForm('datasets/tag-photos','.png,.jpg,.jpeg,.webp,.bmp'))+
  `<div class="section-heading"><h2>Printable marker library</h2><a href="${filesURL('assets/tags')}">Manage marker files →</a></div><div class="grid">${state.tags.map(t=>`<article class="card"><img class="tag-image" src="${fileURL(t.path,true)}" alt="${esc(t.name)}"><h3>${esc(t.name)}</h3><div class="actions"><a class="button secondary" href="${fileURL(t.path)}">Download</a><a class="button secondary" href="/print-tag?path=${enc(t.path)}" target="_blank" rel="noopener">Print ↗</a><button class="text-button danger-text" data-action="trash" data-path="${esc(t.path)}">Trash</button></div></article>`).join('') || empty('Generate your first marker')}</div>`;
}

const fieldHelp = {
  weights:'Model task must provide axis-aligned boxes. Not needed for tag-only.', target_class:'Exact class name from the selected model. Inspect it in Model library.',
  arduino_ip:'Actual UNO R4 address from Serial Monitor; no automatic discovery.', phone_ip:'Optional PC interface address shared with the phone. Blank = PhoneSense default.',
  left_pins:'IN1, IN2, PWM', right_pins:'IN1, IN2, PWM', object_stop_area:'Box area / full image area; 0.12 = 12%.', tag_stop_width:'Mean tag edge / image width; 0.24 = 24%.',
  settle_seconds:'Wait after all PWM reaches zero, before taking fresh frames.', discard_frames:'Discard this many new frame arrivals after settling.', observe_frames:'Distinct stationary images in one observation burst.', confirmations:'Minimum agreeing detections in the burst.',
  max_result_age:'Max seconds from local image receipt to completed detection.', search_seconds:'Search hold duration; ramp-up/down add time.', min_pwm:'Minimum PWM needed to overcome motor friction. Calibrate on the robot.',
  tracker:'Track IDs reset between movement steps; no identity guarantee across camera jumps.', tag_quad_decimate:'1 keeps full resolution; larger values trade small-tag accuracy for speed.'
};

function helpIcon(text) {
  return text ? `<span class="field-help" tabindex="0" role="img" aria-label="Help: ${esc(text)}">?<span class="help-text" role="tooltip">${esc(text)}</span></span>` : '';
}

function addFieldHelp(root=document) {
  const descriptions={folder:'Project subfolder receiving the uploaded files. Existing files will not be overwritten.',
    files:'Choose local files to copy into the selected project folder.', label:'Friendly display name for this model; changing it does not rename the weight file.',
    provider:'Runtime provider responsible for loading this model. A catalog entry alone does not add execution support.',
    notes:'Notes about the model, its classes, data or intended use. These do not change the weights.',
    name:'Name for this file, folder or saved profile. Use a distinct name to keep previous versions.',
    video:'Live uses PhoneSense. Replay uses the selected video and always disables physical motors.',
    max_batches:'Stop after this many observation batches, or use 0 to continue until completion or Stop.',
    save_frames:'Save accepted stationary source images alongside the session CSV for later inspection.',
    drive:'Enable real motor commands to the configured Arduino. Leave unchecked for a preview.',
    mission_mode:'Choose which target sequence the mission should complete.',
    'model-filter':'Filter model cards by filename, display name, provider or task.',
    'profile-select':'Load saved mission settings into this form. Loading a profile does not start a mission.',
    'monitor-select':'Choose the mission whose camera, terminal output and movement requests to display.',
    'follow-log':'Keep the terminal scrolled to the newest output. Uncheck to read earlier lines.',
    editor:'Project file contents. Saving preserves a revision backup; read-only guides can be opened for editing.'};
  for(const input of $$('input, select, textarea',root)) {
    if(input.type==='hidden' || input.dataset.helpAdded) continue;
    const label=input.closest('label'), target=label?.querySelector(':scope > span') || label;
    input.dataset.helpAdded='true';
    if(label?.querySelector('.field-help')) continue;
    const key=input.name || input.id;
    const text=descriptions[key] || state?.config_schema.find(f=>f.key===key)?.help ||
      Object.values(state?.catalog.jobs || {}).flatMap(s=>s.fields).find(f=>f.key===key)?.help;
    if(!text) continue;
    const icon=document.createElement('span'); icon.innerHTML=helpIcon(text);
    if(target) target.append(icon); else input.insertAdjacentElement('afterend',icon);
  }
}

document.addEventListener('click',event=>{
  const help=event.target.closest('.field-help');
  if(help) { event.preventDefault(); event.stopPropagation(); help.focus(); }
});

function configField(key, values) {
  const spec = state.config_schema.find(f=>f.key===key); if (!spec) return '';
  const value = values[key] ?? '';
  let input;
  if (key === 'weights') input = `<select name="weights"><option value="">No model (tag-only)</option>${options(state.models.map(m=>({value:m.path,label:m.label || m.name})),value)}</select>`;
  else if (key === 'tracker') input = `<select name="tracker">${options(state.catalog.trackers.map(t=>({value:t.id,label:t.label})),value)}</select>`;
  else if (spec.options) input = `<select name="${key}">${options(spec.options.map(v=>({value:v,label:v})),value)}</select>`;
  else if (key === 'target_class') input = `<input name="target_class" value="${esc(value)}" list="class-options" required>`;
  else input = `<input name="${key}" type="${spec.kind==='number' ? 'number' : 'text'}" ${spec.kind==='number' ? `step="${spec.integer ? '1' : 'any'}"` : ''} value="${esc(spec.kind==='pins' ? value.join(', ') : value)}" ${key==='phone_ip' ? '' : 'required'}>`;
  return `<label class="field"><span>${esc(key.replaceAll('_',' '))} ${helpIcon(spec.help || fieldHelp[key])}</span>${input}</label>`;
}

function missionValues() {
  try { return {...state.defaults, ...JSON.parse(localStorage.getItem('main-mission-config') || '{}')}; } catch { return {...state.defaults}; }
}

function missionsPage() {
  const values = missionValues();
  const groups = {
    'Object & tag inputs':['object_provider','weights','target_class','tracker','tag_id','object_stop_area','tag_stop_width','center_tolerance'],
    'OpenCV color settings':['color_h','color_s','color_v','color_h_tolerance','color_s_tolerance','color_v_tolerance','color_min_area'],
    'Connections & motor wiring':['arduino_ip','phone_ip','left_pins','right_pins'],
    'Stationary observations & detection':['settle_seconds','discard_frames','observe_frames','confirmations','confidence','image_size','device','tag_min_margin','tag_quad_decimate','tag_threads'],
    'Timed movement & ramps':['motion_driver','forward_speed','turn_speed','search_speed','forward_min_seconds','forward_max_seconds','turn_min_seconds','turn_max_seconds','search_seconds','min_pwm','acceleration','deceleration','motor_interval','ramp_timeout'],
    'Steering & position filtering':['steering_controller','pid_kp','pid_ki','pid_kd','pid_integral_limit','position_estimator','filter_process_noise','filter_measurement_noise','filter_motion_noise','filter_max_uncertainty'],
    'Timeouts & target memory':['observation_timeout','inference_timeout','camera_timeout','camera_reconnect_timeout','max_result_age','empty_batches_before_search','position_memory_seconds']
  };
  const known = Object.values(groups).flat().concat('mission_mode');
  const additional = state.config_schema.map(f=>f.key).filter(k=>!known.includes(k));
  if(additional.length) groups['Additional plugin settings']=additional;
  return heading('Missions', 'Choose a behavior, configure its inputs and run a hardware-free preview or an explicit drive session.', docs('missions/stop_and_stare/README.md'))+
  `<div class="notice">All three templates use stop-and-stare: stop → settle → fresh images → confirm → one movement pulse → stop. The robot never plans from images taken during a pulse.</div>`+
  `<form data-form="mission" id="mission-form"><div class="radio-cards">${state.catalog.missions.map(m=>`<label class="radio-card"><input type="radio" name="mission_mode" value="${m.id}" ${values.mission_mode===m.id ? 'checked' : ''}><strong>${esc(m.label)}</strong><span>${esc(m.description)}</span></label>`).join('')}</div>`+
  `<div class="panel"><div class="panel-header"><div><h2>Mission configuration</h2><p>Saved profiles and job snapshots make runs reproducible.</p></div><div class="actions"><select id="profile-select" aria-label="Saved profile"><option value="">Load profile…</option>${options(state.profiles.map(p=>({value:p.id,label:p.name})), '')}</select><button type="button" class="secondary" data-action="save-profile">Save profile</button></div></div><div class="panel-body">${Object.entries(groups).map(([label,keys],i)=>`<details ${i<2 ? 'open' : ''}><summary>${esc(label)}</summary><div class="form-grid">${keys.map(k=>configField(k,values)).join('')}</div></details>`).join('')}<datalist id="class-options">${[...new Set(state.models.flatMap(m=>Object.values(m.inspection?.classes || {})))].map(c=>`<option value="${esc(c)}"></option>`).join('')}</datalist></div></div>`+
  panel('Launch options','Preview is the default. Camera URLs appear in the job log once PhoneSense starts.',`<div class="form-grid"><label class="field"><span>Input source</span><select name="video"><option value="">Live PhoneSense camera</option>${options(state.media.filter(m=>m.video).map(m=>({value:m.path,label:`Replay: ${m.path}`})), '')}</select></label><label class="field"><span>Observation batch limit (0 = until complete / Stop)</span><input name="max_batches" type="number" min="0" max="100000" step="1" value="0" required></label></div><div class="form-actions"><label><input type="checkbox" name="save_frames" checked> Save stationary observations</label></div><div class="form-actions"><label class="drive-check"><input type="checkbox" name="drive"><span><strong>Enable physical robot driving</strong><small>Connects to the configured Arduino. Leave unchecked for preview. Video replay always disables motors.</small></span></label></div><div class="form-actions"><button class="primary" type="submit">Start mission →</button><button class="secondary" type="button" data-action="reset-mission">Reset defaults</button><span class="muted">One mission / compute job at a time.</span></div>`)+'</form>'+
  jobForms('missions')+
  `<div class="section-heading"><h2>Mission runs</h2><a href="${filesURL('runs/stop_and_stare')}">Session files →</a></div><section class="panel">${jobsTable(state.jobs.filter(j=>j.kind==='mission'),8)}</section>`;
}

function collectConfig(form) {
  const data = new FormData(form); const result = {};
  for (const spec of state.config_schema) {
    let value = data.get(spec.key);
    if (value === null) continue;
    if(spec.kind==='number') value = Number(value);
    if(spec.kind==='pins') value = value.split(',').map(v=>Number(v.trim()));
    if(spec.key==='phone_ip' && !value) value=null;
    if(spec.key==='weights' && !value) value=state.defaults.weights;
    result[spec.key]=value;
  }
  return result;
}

function jobsPage() {
  return heading('Jobs & logs','Processes, parameters, results and artifacts for every task.',docs('webapp/README.md','Job controls'))+
  `<div id="job-detail" class="job-detail">${chosenJob ? '<div class="loading">Loading job…</div>' : ''}</div><section class="panel" id="job-table">${jobsTable(state.jobs)}</section>`;
}

async function jobDetail(id) {
  const job=await api(`/api/jobs/${id}`);
  const detail=$('#job-detail');
  if(detail) renderJobDetail(job,detail);
}

async function filesPage() {
  const path = new URLSearchParams(location.search).get('path') || '';
  const listing = await api(`/api/files?path=${enc(path)}`);
  const parts=path.split('/').filter(Boolean);
  const breadcrumb=`<div class="breadcrumbs"><a href="/files">Main</a>${parts.map((p,i)=>`<span>/</span><a href="${filesURL(parts.slice(0,i+1).join('/'))}">${esc(p)}</a>`).join('')}</div>`;
  return heading('Files & editor','Browse Main, edit text, organize files and recover items from Trash.', `<button class="secondary" data-action="show-trash">Trash & recovery</button>`)+breadcrumb+
  `<section class="panel"><div class="panel-header"><div><h2>${esc(parts.at(-1) || 'Main')}</h2><p>${listing.entries.length} items · hidden and environment paths excluded</p></div><div class="actions"><button class="secondary" data-action="new-file" data-path="${esc(path)}">New text file</button><button class="secondary" data-action="mkdir" data-path="${esc(path)}">New folder</button></div></div>${listing.entries.length ? `<div class="table-wrap"><table><thead><tr><th>Name</th><th>Size</th><th>Modified</th><th>Manage</th></tr></thead><tbody>${listing.entries.map(f=>`<tr><td>${f.directory ? `<a class="file-name" href="${filesURL(f.path)}">▱ ${esc(f.name)}</a>` : `<a class="file-name" href="${fileURL(f.path)}">${esc(f.name)}</a>`}</td><td>${f.directory ? '—' : size(f.bytes)}</td><td>${date(f.modified*1000)}</td><td><div class="actions">${f.editable ? `<button class="text-button" data-action="edit" data-path="${esc(f.path)}">Edit</button>` : ''}${f.image || f.video ? `<button class="text-button" data-action="preview" data-path="${esc(f.path)}" data-video="${f.video}">Preview</button>` : ''}${!f.directory ? `<a href="${fileURL(f.path)}">Download</a>` : ''}<button class="text-button" data-action="rename" data-path="${esc(f.path)}">Rename</button><button class="text-button danger-text" data-action="trash" data-path="${esc(f.path)}">Trash</button></div></td></tr>`).join('')}</tbody></table></div>` : empty('This folder is empty')}</section>`+
  panel('Upload into this folder','Choose a subfolder when browsing the Main root.',uploadForm(path || 'datasets/uploads'));
}

function settingsPage() {
  return heading('Profiles & settings','Reusable mission configurations and the capabilities available in this workspace.', docs('webapp/EXTENDING.md','Extension guide'))+
  `<div class="split"><div>${panel('Saved mission profiles','Load a profile, adjust its inputs and launch from Missions.',state.profiles.map(p=>`<div class="profile-row"><div><strong>${esc(p.name)}</strong><span class="source-path">${esc(p.config.mission_mode)} · ${esc(p.config.tracker)} · tag ${esc(p.config.tag_id)}</span></div><div class="actions"><button class="text-button" data-action="load-profile" data-id="${esc(p.id)}">Load →</button><button class="text-button danger-text" data-action="delete-profile" data-id="${esc(p.id)}">Delete</button></div></div>`).join('') || empty('No profiles saved','Configure a mission, then select Save profile.'))}</div><div>${panel('Local runtime','Uses the same Python environment as the project.',`<dl class="runtime-list"><dt>Managed folder</dt><dd class="mono">${esc(state.root)}</dd><dt>Python</dt><dd>${esc(state.runtime.python)}</dd><dt>Interpreter</dt><dd class="mono">${esc(state.runtime.executable)}</dd><dt>Roboflow key</dt><dd>${state.runtime.roboflow_key ? '<span class="badge">Available in environment</span>' : '<span class="badge warning">Not configured</span>'}</dd><dt>API key setup</dt><dd>Set <code>ROBOFLOW_API_KEY</code> in PowerShell before starting the server. Keys are never stored in profiles or entered in this UI.</dd></dl>`)}</div></div>`+
  panel('Capability catalog','Registries describe providers, trackers, job inputs, resource requirements and outputs.',`<div class="grid two">${state.catalog.providers.map(p=>`<div><h3>${esc(p.label)}</h3><p>${esc(p.tasks.join(' · '))}</p>${p.extensions.map(e=>`<span class="chip">${esc(e)}</span>`).join('')}</div>`).join('')}<div><h3>Trackers</h3>${state.catalog.trackers.map(t=>`<p><strong>${esc(t.label)}</strong><br>${esc(t.description)}</p>`).join('')}</div><div><h3>Job formats</h3>${Object.entries(state.catalog.jobs).map(([id,s])=>`<p><strong>${esc(id)}</strong> → ${esc(s.outputs.join(', '))}</p>`).join('')}</div></div>`);
}

async function render() {
  $$('.sidebar a[data-page]').forEach(a=>a.classList.toggle('active',a.dataset.page===page));
  const pages={home, datasets:datasetsPage, models:modelsPage, training:trainingPage, tags:tagsPage, missions:missionsPage, jobs:jobsPage, live:missionMonitorPage, files:filesPage, settings:settingsPage};
  $('#content').innerHTML=await (pages[page] || home)();
  addFieldHelp();
  if(page==='missions') syncMissionInputs();
  updateJobStatus();
  if(['jobs','live'].includes(page) && chosenJob) await jobDetail(chosenJob);
}

function syncMissionInputs() {
  const form=$('#mission-form'); if(!form) return;
  const mode=new FormData(form).get('mission_mode');
  const spec=state.catalog.missions.find(m=>m.id===mode);
  const object=mode!=='tag_only', hsv=form.elements.object_provider?.value==='hsv';
  const sets={model:['target_class','object_stop_area','object_provider'],
    apriltag:['tag_id','tag_stop_width','tag_min_margin','tag_quad_decimate','tag_threads']};
  for(const [capability,keys] of Object.entries(sets)) for(const key of keys) {
    const input=form.elements[key]; if(!input) continue;
    const visible=(spec?.inputs || ['model','apriltag']).includes(capability);
    input.disabled=!visible;
    input.closest('.field').hidden=!visible;
  }
  for(const key of ['weights','tracker','confidence','image_size','device',...state.config_schema.filter(f=>f.key.startsWith('color_')).map(f=>f.key)]) {
    const input=form.elements[key]; if(!input) continue;
    const visible=object && (key.startsWith('color_') ? hsv : !hsv);
    input.disabled=!visible; input.closest('.field').hidden=!visible;
  }
  for(const key of ['pid_kp','pid_ki','pid_kd','pid_integral_limit','filter_process_noise','filter_measurement_noise','filter_motion_noise','filter_max_uncertainty']) {
    const input=form.elements[key]; if(!input) continue;
    const visible=key.startsWith('pid_') ? form.elements.steering_controller.value==='pid' : form.elements.position_estimator.value==='kalman';
    input.disabled=!visible; input.closest('.field').hidden=!visible;
  }
}

function updateJobStatus() {
  const active=state.jobs.filter(isActive);
  $('#job-count').textContent=active.length;
  $('#resource-status').textContent=active.length ? `${active.length} active job${active.length>1 ? 's' : ''}` : 'Workspace idle';
  $('#resource-status').className=`badge ${active.length ? '' : 'neutral'}`;
}

async function refresh() { state=await api('/api/state'); await render(); }
async function launch(kind,params) { const job=await api('/api/jobs',{method:'POST',body:{kind,params}}); location.href=kind==='mission' ? liveURL(job.id) : jobURL(job.id); }

async function openText(path, editable=false) {
  editor=await api(`/api/text?path=${enc(path)}`);
  modal(path, `<textarea class="editor" id="editor" spellcheck="false" ${editable ? '' : 'readonly'} aria-label="File contents"></textarea><div class="form-actions">${editable ? '<button class="primary" data-action="save-text">Save changes</button><span class="muted">Previous contents are backed up. Conflicting edits are rejected.</span>' : `<button class="secondary" data-action="edit" data-path="${esc(path)}">Edit file</button>`}</div>`,true);
  $('#editor').value=editor.text;
}

async function nameDialog(title, action, path, defaultValue='', label='Name') {
  modal(title,`<form data-form="name" data-action="${action}" data-path="${esc(path)}"><label class="field"><span>${esc(label)}</span><input name="name" value="${esc(defaultValue)}" required maxlength="100"></label><div class="form-actions"><button class="primary">Save</button></div></form>`);
}

document.addEventListener('click', async event=>{
  const button=event.target.closest('[data-action]'); if(!button) return;
  const {action,path,id}=button.dataset;
  try {
    if(action==='read' || action==='edit') return await openText(path,action==='edit');
    if(action==='preview') return modal(path,button.dataset.video==='true' ? `<video class="preview" controls src="${fileURL(path,true)}"></video>` : `<img class="preview" src="${fileURL(path,true)}" alt="${esc(path)}">`,true);
    if(action==='save-text') { editor=await api('/api/text',{method:'PUT',body:{path:editor.path,text:$('#editor').value,revision:editor.revision}}); toast('Saved. Previous contents retained in .control/history.'); return; }
    if(action==='rename') return nameDialog('Rename file or folder','rename',path,path.split('/').at(-1));
    if(action==='mkdir') return nameDialog('Create a folder','mkdir',path);
    if(action==='new-file') return nameDialog('Create a text file','new-file',path,'notes.md','Filename with extension');
    if(action==='trash') {
      modal('Move to Trash',`<p>Move <strong>${esc(path)}</strong> to local Trash? You can restore it from Files & editor.</p><div class="form-actions"><button class="danger" data-action="confirm-trash" data-path="${esc(path)}">Move to Trash</button></div>`); return;
    }
    if(action==='confirm-trash') { await api('/api/trash',{method:'POST',body:{path}}); $('#dialog').close(); await refresh(); toast('Moved to Trash. Restore it from Files & editor.'); return; }
    if(action==='show-trash') {
      const data=await api('/api/trash'); modal('Trash & recovery',data.records.map(r=>`<div class="profile-row"><div><strong>${esc(r.path)}</strong><small class="source-path">${date(r.time)}</small></div><button class="secondary" data-action="restore" data-id="${r.id}">Restore</button></div>`).join('') || empty('Trash is empty')); return;
    }
    if(action==='restore') { await api('/api/restore',{method:'POST',body:{id}}); $('#dialog').close(); await refresh(); toast('Restored.'); return; }
    if(action==='inspect') return await launch('inspect_model',{model:path});
    if(action==='model-edit') {
      const model=state.models.find(m=>m.path===path);
      modal('Model details',`<form data-form="model" data-path="${esc(path)}"><div class="form-grid"><label class="field"><span>Display name</span><input name="label" value="${esc(model.label || model.name)}" maxlength="100" required></label><label class="field"><span>Provider</span><select name="provider">${options(state.catalog.providers.map(p=>({value:p.id,label:p.label})),model.provider)}</select></label><label class="field wide"><span>Notes</span><textarea name="notes" rows="5" maxlength="4000">${esc(model.notes || '')}</textarea></label></div><div class="form-actions"><button class="primary">Save details</button><a href="${fileURL(path)}">Download weights</a></div></form>`); return;
    }
    if(action==='stop') { await api(`/api/jobs/${id}/stop`,{method:'POST',body:{}}); toast('Stop requested. Waiting for the process to exit.'); if(['jobs','live'].includes(page)) await jobDetail(id); return; }
    if(action==='copy-artifact') { return nameDialog('Copy artifact into datasets','copy-artifact',id,`synthetic-${id.slice(0,8)}.png`,'New image filename'); }
    if(action==='save-profile') {
      localStorage.setItem('main-mission-config',JSON.stringify(collectConfig($('#mission-form'))));
      return nameDialog('Save mission profile','save-profile','','my-mission','Profile name (reusing a name updates it)');
    }
    if(action==='load-profile') { const profile=state.profiles.find(p=>p.id===id); localStorage.setItem('main-mission-config',JSON.stringify(profile.config)); location.href='/missions'; return; }
    if(action==='delete-profile') { await api(`/api/profiles/${enc(id)}`,{method:'DELETE'}); await refresh(); toast('Profile deleted.'); return; }
    if(action==='reset-mission') { localStorage.removeItem('main-mission-config'); await render(); return; }
  } catch(error) { toast(error.message,true); }
});

document.addEventListener('submit',async event=>{
  const form=event.target.closest('form[data-form]'); if(!form) return; event.preventDefault();
  const submit=$('button[type="submit"],button:not([type])',form); if(submit) submit.disabled=true;
  const submitLabel=submit?.textContent;
  if(submit) submit.textContent=form.dataset.form==='upload' ? 'Uploading…' : 'Working…';
  try {
    const values=Object.fromEntries(new FormData(form));
    if(form.dataset.form==='upload') { const data=await api('/api/upload',{method:'POST',body:new FormData(form)}); await refresh(); toast(`Uploaded ${data.files.length} file(s).`); }
    if(form.dataset.form==='job') {
      for(const f of state.catalog.jobs[form.dataset.kind].fields) if(f.kind==='number') values[f.key]=Number(values[f.key]);
      await launch(form.dataset.kind,values);
    }
    if(form.dataset.form==='mission') {
      const config=collectConfig(form); localStorage.setItem('main-mission-config',JSON.stringify(config));
      await launch('mission',{config,video:values.video,drive:!!form.elements.drive.checked,save_frames:!!form.elements.save_frames.checked,max_batches:Number(values.max_batches)});
    }
    if(form.dataset.form==='model') { await api('/api/models',{method:'PUT',body:{path:form.dataset.path,...values}}); $('#dialog').close(); await refresh(); toast('Model details saved.'); }
    if(form.dataset.form==='name') {
      const action=form.dataset.action, path=form.dataset.path;
      if(action==='rename') await api('/api/rename',{method:'POST',body:{path,name:values.name}});
      if(action==='mkdir') await api('/api/folders',{method:'POST',body:{parent:path,name:values.name}});
      if(action==='new-file') { const result=await api('/api/new-file',{method:'POST',body:{parent:path,name:values.name}}); $('#dialog').close(); await refresh(); await openText(result.path,true); return; }
      if(action==='save-profile') await api('/api/profiles',{method:'POST',body:{name:values.name,config:JSON.parse(localStorage.getItem('main-mission-config'))}});
      if(action==='copy-artifact') await api(`/api/jobs/${path}/copy-artifact`,{method:'POST',body:{name:'synthetic-tag.png',output:values.name}});
      $('#dialog').close(); await refresh(); toast('Saved.');
    }
  } catch(error) { toast(error.message,true); }
  finally { if(submit) { submit.disabled=false; submit.textContent=submitLabel; } }
});

document.addEventListener('input',event=>{
  if(event.target.id==='model-filter') $$('.model-card').forEach(card=>card.hidden=!card.dataset.filter.includes(event.target.value.toLowerCase()));
});
document.addEventListener('change',async event=>{
  if(event.target.id==='monitor-select' && event.target.value) { location.href=liveURL(event.target.value); return; }
  if(['mission_mode','object_provider','steering_controller','position_estimator'].includes(event.target.name)) syncMissionInputs();
  if(event.target.id==='profile-select' && event.target.value) {
    const profile=state.profiles.find(p=>p.id===event.target.value);
    localStorage.setItem('main-mission-config',JSON.stringify(profile.config)); await render(); toast(`Loaded ${profile.name}.`);
  }
  if(event.target.name==='video' && page==='missions') {
    const drive=$('#mission-form').elements.drive;
    if(event.target.value) drive.checked=false;
    drive.disabled=!!event.target.value;
  }
});
$('#dialog-close').addEventListener('click',()=>$('#dialog').close());
$('#refresh').addEventListener('click',()=>refresh().then(()=>toast('Project refreshed.')).catch(e=>toast(e.message,true)));

async function pollJobs() {
  if(!state || polling || document.hidden) return; polling=true;
  try {
    const data=await api('/api/jobs');
    const previous=state.jobs;
    state.jobs=data.jobs; updateJobStatus();
    if(page==='jobs') {
      $('#job-table').innerHTML=jobsTable(state.jobs);
    }
    if(['jobs','live'].includes(page) && chosenJob) await jobDetail(chosenJob);
    if(previous.some(j=>isActive(j) && !state.jobs.some(n=>n.id===j.id && isActive(n)))) {
      state=await api('/api/state');
      toast('A job finished. Its results are available in Jobs & logs.');
    }
  } catch(error) { $('#resource-status').textContent='Server unavailable'; }
  finally { polling=false; }
}
refresh().catch(error=>$('#content').innerHTML=`<div class="page-error"><h2>Could not open the workspace</h2><p>${esc(error.message)}</p><button class="secondary" id="retry">Retry</button></div>`).finally(()=>{ const retry=$('#retry'); if(retry) retry.addEventListener('click',()=>location.reload()); });
setInterval(pollJobs,1000);
