/* Mission monitor and stable, followable subprocess terminal output. */
let colorPickSession=null;

async function freezeColorFrame(id) {
  const frame=await api(`/api/jobs/${id}/color-frame`,{method:'POST',body:{}});
  colorPickSession={id,token:frame.token};
  const image=$('#color-pick-image');
  image.src=frame.image; image.hidden=false;
  $('#color-pick-instruction').textContent='Click the target color in this frozen frame. The live feed continues above.';
}

document.addEventListener('click',async event=>{
  const action=event.target.closest('[data-action="color-freeze"]');
  const live=event.target.id==='live-camera' && !$('#color-picker')?.hidden;
  try {
    if(action || live) { await freezeColorFrame(chosenJob); return; }
    if(event.target.id!=='color-pick-image' || !colorPickSession || !event.target.naturalWidth) return;
    const rect=event.target.getBoundingClientRect();
    const x=(event.clientX-rect.left)/rect.width, y=(event.clientY-rect.top)/rect.height;
    if(x<0 || x>1 || y<0 || y>1) return;
    const result=await api(`/api/jobs/${colorPickSession.id}/color`,{method:'POST',body:{token:colorPickSession.token,x,y}});
    $('#color-pick-instruction').textContent=`HSV ${result.hsv.join(', ')} queued. The robot applies it while stopped and takes new observations.`;
    event.target.hidden=true; colorPickSession=null;
    toast(result.message);
  } catch(error) { toast(error.message,true); }
});
function liveURL(id) { return `/live?id=${enc(id)}`; }

function phoneConnectionCard(job) {
  const runtime=job.mission || {}, active=isActive(job);
  if(job.params.video) return '<div class="notice">Video replay: PhoneSense is not started and motors are disabled.</div>';
  const links=runtime.phone_urls || [];
  const reconnect=active && runtime.phase==='WAIT_RECONNECT' ? `<p><strong>Camera interrupted; motors stopped.</strong> PhoneSense is still running. On your phone, tap Connect and Start camera again. The mission waits up to ${esc(job.params.config?.camera_reconnect_timeout ?? 60)} seconds for fresh frames, then starts a new stationary observation.</p>` : '';
  return `<div class="notice ${active ? '' : 'amber'}"><h3>Connect your phone camera</h3>${reconnect}${links.length ? `<p>${active ? 'Open the PC address on the same network as your phone:' : 'This mission has ended. These were its connection addresses; start a new live mission before connecting.'}</p><div class="phone-links">${links.map(url=>active ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(url)} ↗</a>` : `<span class="mono">${esc(url)}</span>`).join('')}</div>` : `<p>${active ? 'Waiting for PhoneSense to start. Model loading and warmup happen first; connection links will appear here automatically.' : 'No PhoneSense address was recorded. Check the terminal output for startup errors.'}</p>`}<p class="inline-help">On the phone: open the HTTPS address, handle its local certificate prompt, allow camera access, then choose Connect and Start camera. Keep the mission running. The management app on port 5000 is a separate server.</p></div>`;
}

function missionMonitorPage() {
  const missions=state.jobs.filter(j=>j.kind==='mission');
  if(!chosenJob) chosenJob=(missions.find(isActive) || missions[0])?.id;
  return heading('Mission monitor','Live camera, phone connection, terminal output and movement requests in one place.',`<a class="button secondary" href="/missions">Configure a mission →</a>`)+
    `<div class="panel"><div class="panel-body"><label class="field"><span>Mission session</span><select id="monitor-select"><option value="">Choose a mission…</option>${options(missions.map(j=>({value:j.id,label:`${j.status} · ${j.params.config?.mission_mode || j.label} · ${date(j.created)} · ${j.id.slice(0,8)}`})),chosenJob)}</select></label></div></div><div id="job-detail">${chosenJob ? '<div class="loading">Loading mission…</div>' : empty('No missions yet','Configure and start a mission. Its monitor opens automatically.')}</div>`;
}

function renderJobDetail(job, container) {
  const monitor=page==='live', mission=job.kind==='mission';
  if(container.dataset.jobId!==job.id) {
    colorPickSession=null;
    container.dataset.jobId=job.id;
    const terminal=`<section class="panel terminal-panel"><div class="panel-header"><h3>${mission ? 'Mission terminal output' : 'Job terminal output'}</h3><div class="actions"><label class="follow-label"><input id="follow-log" type="checkbox" checked> Follow output</label><a href="${artifactURL(job.id,'output.log')}">Download full log ↓</a></div></div><p class="log-caption">stdout + stderr from the job's Python process · latest 60 KB · refreshes every second</p><pre class="logs" id="live-log" tabindex="0" aria-label="Job terminal output"></pre></section>`;
    const camera=`<section class="panel"><div class="panel-header"><h3>Live camera</h3><span id="camera-status" class="badge neutral">Waiting</span></div><div class="panel-body"><p id="camera-phase" class="inline-help"></p><div class="camera-stage"><img id="live-camera" class="preview" alt="Live camera received by the mission" hidden><p id="camera-wait" class="empty">Waiting for the first camera frame…</p></div><p class="inline-help">Display only, up to 5 fps. The robot ignores these frames during motion and plans from fresh stationary observations.</p><div id="color-picker" class="color-picker" hidden><h3>OpenCV color picker</h3><p>Freeze the live camera, then click the object color. An unset color keeps the robot stopped.</p><div class="color-picker-controls"><button type="button" class="secondary" data-action="color-freeze">Freeze frame & pick color</button><span id="selected-color" class="mono"></span></div><p id="color-pick-instruction" class="inline-help">You can also click the live video to freeze a frame.</p><img id="color-pick-image" class="color-pick-image" alt="Frozen camera frame: click the target color" hidden></div></div></section>`;
    container.innerHTML=`<section class="panel"><div class="panel-header"><div><div class="job-title"><h2>${esc(job.label)}</h2><span id="detail-status"></span></div><p class="mono">${esc(job.id)}</p></div><div class="actions">${mission && !monitor ? `<a class="button secondary" href="${liveURL(job.id)}">Open mission monitor →</a>` : ''}<button id="detail-stop" class="danger" data-action="stop" data-id="${job.id}">Stop job</button></div></div><div class="panel-body"><div id="detail-message"></div>${mission ? '<div id="phone-connection"></div><div id="mission-stats" class="stats"></div>' : ''}</div></section>${monitor && mission ? `<div class="monitor-grid">${camera}${terminal}</div>` : terminal}${mission && monitor ? '<section class="panel"><div class="panel-header"><h3>Movement requests</h3><span class="muted">Normalized demands, not measured velocity or Arduino acknowledgements</span></div><div id="command-history"></div></section>' : ''}<section class="panel"><div class="panel-body"><div class="split"><div><h3>Artifacts & outputs</h3><div id="job-artifacts" class="artifact-list"></div><div id="project-outputs" class="artifact-list"></div><div id="artifact-copy"></div><details class="subsection"><summary>Job parameters</summary><pre>${esc(JSON.stringify(job.params,null,2))}</pre></details><details id="result-details" hidden><summary>Result data</summary><pre id="result-data"></pre></details></div><div><h3>${mission ? 'Last stationary detector image' : 'Output preview'}</h3><p class="inline-help">${mission ? 'Processed while stopped. This image stays unchanged during movement.' : 'Annotated or generated image.'}</p><img id="stationary-preview" class="preview" alt="Stationary detector or output image" hidden><p id="stationary-wait" class="muted">No processed image yet.</p></div></div></div></section>`;
    addFieldHelp(container);
  }
  $('#detail-status').innerHTML=status(job.status);
  $('#detail-stop').hidden=!isActive(job);
  $('#detail-stop').textContent=job.status==='stop_requested' ? 'Stop requested…' : 'Stop job';
  $('#detail-message').textContent=job.message || '';
  const log=$('#live-log'), follow=$('#follow-log').checked, oldScroll=log.scrollTop;
  if(log.textContent!==job.log) log.textContent=job.log || '';
  log.scrollTop=follow ? log.scrollHeight : oldScroll;
  $('#job-artifacts').innerHTML=(job.artifacts || []).filter(n=>!['camera.jpg','runtime.json'].includes(n)).map(n=>`<a href="${artifactURL(job.id,n)}">${esc(n)} ↓</a>`).join('') || '<span class="muted">No artifacts yet.</span>';
  $('#project-outputs').innerHTML=(job.result?.files || []).map(p=>`<a href="${outputURL(p)}">${esc(p)} →</a>`).join('');
  $('#artifact-copy').innerHTML=(job.artifacts || []).includes('synthetic-tag.png') ? `<button class="secondary subsection" data-action="copy-artifact" data-id="${job.id}">Copy synthetic image to datasets</button>` : '';
  $('#result-details').hidden=!job.result?.data;
  $('#result-data').textContent=JSON.stringify(job.result?.data || {},null,2);
  if(job.preview) {
    $('#stationary-preview').hidden=false; $('#stationary-wait').hidden=true;
    const previewStamp=String(job.mission?.observation_sequence ?? job.ended ?? '');
    if($('#stationary-preview').dataset.stamp!==previewStamp) {
      $('#stationary-preview').src=`/api/jobs/${job.id}/observation?t=${enc(previewStamp)}`;
      $('#stationary-preview').dataset.stamp=previewStamp;
    }
  }
  if(!mission) return;
  const runtime=job.mission || {}, fresh=runtime.updated_at && Date.now()-Date.parse(runtime.updated_at)<4000;
  $('#phone-connection').innerHTML=phoneConnectionCard(job);
  const latest=runtime.last_command;
  $('#mission-stats').innerHTML=[['PHASE',runtime.phase || (isActive(job) ? 'Starting / older runner' : job.status)],['LEG',runtime.leg || '—'],['OUTPUT',job.params.drive ? 'DRIVE' : 'PREVIEW'],['LAST REQUEST',latest?.step || 'None']].map(([label,value])=>`<div class="stat"><span class="stat-label">${esc(label)}</span><strong class="telemetry-value">${esc(value)}</strong></div>`).join('');
  if(!monitor) return;
  const colorPicker=$('#color-picker');
  colorPicker.hidden=!(isActive(job) && job.params.config?.object_provider==='hsv' && job.params.config?.mission_mode!=='tag_only' && !['tag','done'].includes(runtime.leg));
  $('#selected-color').textContent=runtime.selected_hsv ? `Active HSV: ${runtime.selected_hsv.join(', ')}` : job.params.config?.color_h>=0 ? `Configured HSV: ${job.params.config.color_h}, ${job.params.config.color_s}, ${job.params.config.color_v}` : 'No color selected';
  $('[data-action="color-freeze"]').disabled=!cameraActiveForColor(runtime,job);
  const cameraActive=isActive(job) && fresh && runtime.camera_connected;
  $('#camera-status').textContent=cameraActive ? 'Receiving frames' : isActive(job) ? 'Waiting / stale' : 'Mission ended';
  $('#camera-status').className=`badge ${cameraActive ? '' : 'neutral'}`;
  $('#camera-phase').textContent=`${runtime.phase || 'Starting'} · ${cameraActive ? 'Camera packets arriving' : 'No current camera stream'}`;
  if(runtime.camera_available) {
    const image=$('#live-camera'); image.hidden=false; $('#camera-wait').hidden=true;
    if(!image.dataset.connected) {
      image.src=`/api/jobs/${job.id}/camera`; image.dataset.connected='true';
    }
  } else $('#camera-wait').textContent=isActive(job) ? 'Waiting for the phone camera. Use the connection links above.' : 'No live preview was recorded for this session. New missions publish video to this monitor.';
  const commands=(runtime.commands || []).slice().reverse();
  $('#command-history').innerHTML=commands.length ? `<div class="table-wrap"><table><thead><tr><th>Time</th><th>Request</th><th>Left</th><th>Right</th><th>Hold</th><th>Leg</th></tr></thead><tbody>${commands.map(c=>`<tr><td>${esc(new Date(c.requested_at).toLocaleTimeString())}</td><td>${esc(c.step)}</td><td class="mono">${Number(c.left).toFixed(3)}</td><td class="mono">${Number(c.right).toFixed(3)}</td><td>${Number(c.hold_seconds).toFixed(3)} s</td><td>${esc(c.leg)}</td></tr>`).join('')}</tbody></table></div>` : empty('No movement requests yet','The mission may be preparing the model or waiting for camera frames.');
}

function cameraActiveForColor(runtime,job) {
  return isActive(job) && !!runtime.camera_available;
}
