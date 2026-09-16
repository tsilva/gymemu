import { PanelRuntime } from './panels/runtime.js';
import { PANEL_TYPES, panelDefinition } from './panels/catalog.js';
import { defaultWorkspace, normalizeWorkspace, loadWorkspace, saveWorkspace, STORAGE_KEY, compareWorkspaceRevisions } from './panels/workspace.js';
import { bindTimelineRange } from './chart-range.js';
import { METRICS } from './panels/telemetry.js';
import { mountPlaybackControls } from './playback-settings.js';
import { setSvgUseHref } from './panels/shared.js';

const $ = selector => document.querySelector(selector);
const token = new URLSearchParams(location.hash.slice(1)).get('token') || sessionStorage.getItem('gymemu-token');
if (token) sessionStorage.setItem('gymemu-token', token);
const windowId = location.pathname === '/workspace/stats' ? 'stats' : 'main';
const isPlayer = windowId === 'main', writer = crypto.randomUUID();
window.name = `gymemu-${windowId}-${location.host}`;
document.body.classList.toggle('stats-window', !isPlayer);
document.body.classList.toggle('player-window', isPlayer);
document.title = isPlayer ? 'Gymemu player' : 'Gymemu diagnostics';
const companion = $('#companion-tab'), companionUrl = new URL(location.href);
companionUrl.pathname = isPlayer ? '/workspace/stats' : '/';
companionUrl.hash = new URLSearchParams({token:token || ''}).toString();
companion.href = companionUrl.href;
companion.target = `gymemu-${isPlayer ? 'stats' : 'main'}-${location.host}`;
companion.title = isPlayer ? 'Open diagnostics tab' : 'Open player tab';
companion.setAttribute('aria-label',companion.title);
companion.querySelector('span').textContent = isPlayer ? 'Diagnostics' : 'Player';
$('#inspection-position').hidden = isPlayer;
$('#mode').hidden = !isPlayer;
$('#add-panel').hidden = isPlayer;
$('.timeline').hidden = !isPlayer;
$('.timeline-actions').hidden = !isPlayer;
const workspaceChannel = typeof BroadcastChannel === 'function' ? new BroadcastChannel('gymemu-player-workspace') : null;
let workspace = loadWorkspace(), snapshot = null, bitmaps = {}, active = true, syncing = false, pendingCommand = Promise.resolve();
let runtime, grid, toastTimer, saveTimer, chartRange = null, chartHoverStep = null, scrubbing = false;
let rangeRevision = {clock:0,writer:''};
const RANGE_KEY = 'gymemu-chart-range';
const chartIdentity = s => s ? [s.checkpoint,s.generation || 0,s.mode,s.selection,s.history_epoch].join(':') : null;
const headers = { 'X-Player-Token': token || '', 'Content-Type': 'application/json' };
const showToast = message => {
  $('#toast').textContent = message; $('#toast').hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { $('#toast').hidden = true; }, 6000);
};
async function request(path, options={}) {
  const response = await fetch(path,{...options,headers});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
  return result;
}
function command(value) {
  if (!isPlayer && !['seek','reorder_history','edit_history','revert_history','pause'].includes(value.type)) return Promise.resolve();
  pendingCommand = pendingCommand.then(() => request('/api/command',{method:'POST',body:JSON.stringify(value)})).catch(error => { showToast(error.message); return {error:error.message}; });
  return pendingCommand;
}
function setChartHoverStep(step) {
  chartHoverStep = Number.isFinite(step) ? step : null;
  if(snapshot) runtime?.renderSnapshot(snapshot,{bitmaps,chartRange,chartHoverStep});
}
function renderChartRange() {
  const band=$('#timeline-zoom-band'), label=$('#timeline-zoom');
  $('.timeline').hidden = !isPlayer && !chartRange;
  document.body.classList.toggle('has-chart-range',Boolean(chartRange));
  band.hidden=label.hidden=!chartRange;
  if(chartRange && snapshot) {
    const total=Math.max(1,timelineEnd(snapshot));
    band.style.left=(100*chartRange.first/total)+'%';
    band.style.width=(100*(chartRange.last-chartRange.first)/total)+'%';
    label.textContent='Steps '+chartRange.first+'–'+chartRange.last+' · Reset zoom';
    for(const handle of band.querySelectorAll('.timeline-range-handle')) {
      const first=handle.dataset.edge==='first';
      handle.setAttribute('aria-valuemin',first?0:chartRange.first+1);
      handle.setAttribute('aria-valuemax',first?chartRange.last-1:timelineEnd(snapshot));
      handle.setAttribute('aria-valuenow',chartRange[handle.dataset.edge]);
      handle.setAttribute('aria-valuetext','Step '+chartRange[handle.dataset.edge]);
    }
  }
  if(snapshot) runtime?.renderSnapshot(snapshot,{bitmaps,chartRange,chartHoverStep});
}
function rangeMessage() {return {type:'chart-range',identity:chartIdentity(snapshot),range:chartRange,revision:rangeRevision};}
function setChartRange(range) {
  chartRange=range;
  rangeRevision={clock:Math.max(Date.now(),rangeRevision.clock+1),writer};
  renderChartRange();
  const message=rangeMessage();workspaceChannel?.postMessage(message);
  try {localStorage.setItem(RANGE_KEY,JSON.stringify(message));} catch { /* Storage can be disabled. */ }
}
function receiveRange(message) {
  if(message.identity!==chartIdentity(snapshot) || compareWorkspaceRevisions(message.revision,rangeRevision)<=0) return;
  const range=message.range;
  if(range && (!Number.isInteger(range.first)||!Number.isInteger(range.last)||range.first<0||range.last>timelineEnd(snapshot)||range.first>=range.last)) return;
  rangeRevision=message.revision;chartRange=range;renderChartRange();
}
function inspectStep(step, episodeId, historyEpoch) {
  if(snapshot?.mode==='teacher-forcing' && !snapshot.loading)
    return command({type:'seek',position:step,episode_id:episodeId,history_epoch:historyEpoch});
}
function persist(immediate=false) {
  workspace.revision = {clock:Math.max(Date.now(),workspace.revision.clock+1),writer};
  saveWorkspace(workspace);
  workspaceChannel?.postMessage({type:'layout',layout:workspace});
  clearTimeout(saveTimer);
  const save=()=>{saveTimer=null;return request('/api/workspace',{method:'POST',body:JSON.stringify(workspace),keepalive:true}).catch(e=>showToast(e.message));};
  if(immediate) return save();
  saveTimer=setTimeout(save,250);
}

function placement(id) {
  const panel = workspace.panels[id], min=panelDefinition(workspace,id).minimum;
  return {...panel.placement,minW:min.w,minH:min.h,id};
}
function shelf() {
  const target=$('#panel-shelf-items'); target.replaceChildren();
  Object.entries(workspace.panels).filter(([,p])=>p.placement.window===windowId&&!p.placement.visible).forEach(([id,p])=>{
    const button=document.createElement('button'); button.textContent=p.title; button.setAttribute('aria-label',`Show ${p.title}`);
    button.onclick=()=>{p.placement.visible=true;syncWorkspace();persist();};target.append(button);
  });
  if (!target.children.length) target.textContent='Every panel is visible.';
}
async function syncWorkspace() {
  syncing=true;
  try { await runtime.sync(workspace,windowId); }
  finally { syncing=false; }
  shelf();
  if(snapshot) runtime.renderSnapshot(snapshot,{bitmaps,chartRange,chartHoverStep});
}
async function updatePanel(id,value) {
  Object.assign(workspace.panels[id],value); await syncWorkspace(); persist();
}
function menu(id,anchor) {
  const target=$('#panel-menu'), panel=workspace.panels[id];target.replaceChildren();
  const button=(label,callback)=>{const b=document.createElement('button');b.textContent=label;b.onclick=()=>{target.hidden=true;callback();};target.append(b);};
  button(panel.enabled ? 'Disable widget' : 'Enable widget',()=>updatePanel(id,{enabled:!panel.enabled}));
  button('Hide widget',()=>{panel.placement.visible=false;syncWorkspace();persist();});
  if(panel.type==='telemetry') {
    button('Edit widget',()=>openEditor(id));
    button('Duplicate widget',()=>{const copy=structuredClone(panel);copy.builtin=false;copy.title+=' copy';copy.placement={...copy.placement,y:40};workspace.panels[`panel-${crypto.randomUUID()}`]=copy;syncWorkspace();persist();});
    if(!panel.builtin) button('Delete widget',()=>{delete workspace.panels[id];syncWorkspace();persist();});
  }
  const rect=anchor.getBoundingClientRect();target.style.left=`${Math.min(rect.left,innerWidth-200)}px`;target.style.top=`${Math.min(rect.bottom+5,innerHeight-220)}px`;target.hidden=false;
}
let editingId=null;
function openEditor(id=null) {
  editingId=id;const panel=id ? workspace.panels[id] : null;
  $('#editor-heading').textContent=panel ? 'Edit widget' : 'Add widget';
  $('#editor-title').value=panel?.title||'Telemetry';$('#editor-view').value=panel?.config.view||'stats';
  const fields=$('#editor-metrics');fields.replaceChildren();const legend=document.createElement('legend');legend.textContent='Metrics';fields.append(legend);
  Object.entries(METRICS).forEach(([key,value])=>{const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.value=key;input.checked=(panel?.config.metrics||['mse','inference_ms']).includes(key);label.append(input,document.createTextNode(value.label));fields.append(label);});
  $('#panel-editor').showModal();$('#editor-title').focus();
}
$('#editor-cancel').onclick=()=>$('#panel-editor').close();
$('#panel-editor-form').onsubmit=event=>{
  event.preventDefault();const metrics=[...$('#editor-metrics').querySelectorAll('input:checked')].map(e=>e.value);
  if(!metrics.length) return showToast('Select at least one metric.');
  const title=$('#editor-title').value.trim();if(!title) return;
  const config={view:$('#editor-view').value,metrics};
  if(editingId) updatePanel(editingId,{title,config});
  else {workspace.panels[`panel-${crypto.randomUUID()}`]={type:'telemetry',title,config,builtin:false,enabled:true,placement:{x:0,y:40,w:6,h:8,visible:true,window:'stats'}};syncWorkspace();persist();}
  $('#panel-editor').close();
};
$('#add-panel').onclick=()=>openEditor();
$('#panels-toggle').onclick=()=>{$('#panel-shelf').hidden=!$('#panel-shelf').hidden;};
$('#reset-layout').onclick=async()=>{const revision=workspace.revision;workspace=defaultWorkspace();workspace.revision=revision;await syncWorkspace();await persist(true);showToast('Default workspace restored.');};
document.addEventListener('click',event=>{if(!event.target.closest('[data-panel-menu],#panel-menu')) $('#panel-menu').hidden=true;});
$('#mode').onchange=()=>{showToast('Loading playback mode…');command({type:'mode',mode:$('#mode').value});};
$('#play-toggle').onclick=()=>command({type:snapshot?.playing?'pause':'play'});
$('#reset-playback').onclick=()=>command({type:'reset'});
const playbackSettings = mountPlaybackControls({ services: { getState:()=>snapshot, command } });
const previousStep=document.createElement('button');
previousStep.textContent='Previous step';previousStep.title='Previous recorded frame';
previousStep.onclick=()=>command({type:'seek',position:Math.max(0,snapshot.step-1)});
playbackSettings.element.querySelector('.play-actions').prepend(previousStep);
$('#playback-settings-content').append(playbackSettings.element);
const settingsDialog=$('#playback-settings'), settingsToggle=$('#playback-settings-toggle');
settingsToggle.onclick=()=>{settingsDialog.showModal();settingsToggle.setAttribute('aria-expanded','true');};
$('#playback-settings-close').onclick=()=>settingsDialog.close();
settingsDialog.addEventListener('close',()=>{settingsToggle.setAttribute('aria-expanded','false');settingsToggle.focus();});
function timelineEnd(s) {
  return (s.history || []).reduce((last, point) => Math.max(last, point.step), s.step || 0);
}
function timelinePosition(s, step=s.step) {
  const name=s.mode==='teacher-forcing' ? `EPISODE ${s.selection}` : s.name.toUpperCase();
  return `${name} · STEP ${step}${s.finished&&step===s.step?' · END':''}`;
}
function updateScrubberProgress(slider) {
  slider.style.setProperty('--timeline-progress',`${Number(slider.max)>0?100*Number(slider.value)/Number(slider.max):0}%`);
}
$('#timeline-zoom').onclick=()=>setChartRange(null);
bindTimelineRange($('#timeline-zoom-band'),()=>snapshot?{first:0,last:timelineEnd(snapshot)}:null,()=>chartRange,setChartRange);
$('#timeline-scrubber').addEventListener('pointerdown',()=>{scrubbing=true;});
for(const name of ['pointerup','pointercancel']) document.addEventListener(name,()=>{scrubbing=false;});
$('#timeline-scrubber').oninput=()=>{ $('#position').textContent=timelinePosition(snapshot,Number($('#timeline-scrubber').value));updateScrubberProgress($('#timeline-scrubber')); };
$('#timeline-scrubber').onchange=()=>command({type:'seek',position:Number($('#timeline-scrubber').value)});
function keyName(event) { return event.key === ' ' ? 'space' : event.key.toLowerCase().replace(/^arrow/,''); }
const pressed=new Set();
document.addEventListener('keydown',event=>{
  if(!isPlayer || event.target.closest('input,select,textarea,dialog') || !snapshot) return;
  const key=keyName(event);
  if(!['tab','r','c','escape',...Object.keys(snapshot.keymap)].includes(key)) return;
  event.preventDefault(); if(event.repeat || pressed.has(key)) return;
  pressed.add(key);command({type:'key',key,down:true});
});
document.addEventListener('keyup',event=>{const key=keyName(event);if(pressed.delete(key)) command({type:'key',key,down:false});});
const blur=()=>{scrubbing=false;pressed.clear();if(isPlayer) command({type:'blur'});};
window.addEventListener('blur',blur);
document.addEventListener('visibilitychange',()=>{if(document.hidden) blur();});
window.addEventListener('pagehide',()=>{if(saveTimer) persist(true);active=false;workspaceChannel?.close();if(isPlayer) fetch('/api/command',{method:'POST',headers,body:JSON.stringify({type:'blur'}),keepalive:true}).catch(()=>{});});
setInterval(()=>{if(isPlayer && active && !document.hidden) request('/api/command',{method:'POST',body:JSON.stringify({type:'heartbeat'})}).catch(()=>{});},400);
function chrome(s) {
  $('#checkpoint').textContent=s.checkpoint_label || s.checkpoint.split('/').slice(-2).join('/');$('#checkpoint').title=s.checkpoint;
  $('#mode').value=s.mode;$('#mode').disabled=!s.available_modes?.length||Boolean(s.loading);
  $('#position').textContent=timelinePosition(s);
  $('#inspection-position').textContent=timelinePosition(s);
  const play=$('#play-toggle'), action=s.playing?'pause':'play', label=s.playing?'Pause':'Play';
  play.dataset.action=action;play.setAttribute('aria-label',label);play.title=label;
  setSvgUseHref($('#play-toggle-icon'),`/assets/tabler-icons.svg#ti-player-${action}`);
  play.disabled=s.finished||Boolean(s.loading);
  $('#reset-playback').disabled=Boolean(s.loading);settingsToggle.disabled=Boolean(s.loading);
  previousStep.disabled=s.mode!=='teacher-forcing'||s.step===0||Boolean(s.loading);
  playbackSettings.render(s);
  const slider=$('#timeline-scrubber');slider.disabled=s.mode!=='teacher-forcing'||Boolean(s.loading);slider.max=timelineEnd(s);if(!scrubbing) slider.value=s.step;
  updateScrubberProgress(slider);
}
async function decode(images) {
  return Object.fromEntries(await Promise.all(Object.entries(images).map(async([key,src])=>{
    const image=new Image();image.src=src;await image.decode();return [key,image];
  })));
}
async function boot() {
  if(!token) throw new Error('Open the complete player URL printed in your terminal.');
  try { const saved=await request('/api/workspace'); if(saved) workspace=normalizeWorkspace(saved); } catch(e) { showToast(e.message); }
  if (!isPlayer) {
    grid=window.GridStack.init({alwaysShowResizeHandle:true,animate:false,cellHeight:32,column:12,columnOpts:{breakpointForWindow:true,breakpoints:[{w:640,c:1},{w:960,c:6}]},draggable:{handle:'.panel-drag',scroll:true},float:false,margin:5,maxRow:240,resizable:{handles:'se'}},$('#dashboard'));
    grid.on('change',(_event,nodes)=>{if(syncing)return;nodes.forEach(node=>{const p=workspace.panels[node.id];if(p && grid.getColumn()===12) for(const key of ['x','y','w','h']) p.placement[key]=node[key];});persist();});
  }
  runtime=new PanelRuntime({definitionFor:panelDefinition,container:$('#dashboard'),services:{getState:()=>snapshot,command,inspectStep,setChartRange,setChartHoverStep,showToast,updatePanel},
    onMount:(element,id,_definition,item)=>{
      if(grid) grid.makeWidget(item,placement(id));
      else {item.style.order=['original','prediction','difference'].indexOf(id);element.querySelector('.panel-drag').remove();}
      element.querySelector('[data-panel-menu]').onclick=e=>{e.stopPropagation();menu(id,e.currentTarget);};
    },
    onLayout:(element,id,_place,item,definition)=>{grid?.update(item,placement(id));element.classList.toggle('disabled',!definition.enabled);},
    onUnmount:(_element,_id,item)=>grid?.removeWidget(item,false),onError:(id,error)=>showToast(`${id}: ${error.message}`)});
  await syncWorkspace();
  const receiveLayout = saved => {
    const next=normalizeWorkspace(saved);
    if(compareWorkspaceRevisions(next.revision,workspace.revision)<=0) return;
    clearTimeout(saveTimer);saveTimer=null;
    workspace=next;saveWorkspace(workspace);syncWorkspace();
  };
  workspaceChannel?.addEventListener('message',({data})=>{
    if(data?.type==='chart-range') receiveRange(data);
    else if(data?.type==='request-chart-range' && snapshot) workspaceChannel.postMessage(rangeMessage());
    else if(data?.type==='layout') receiveLayout(data.layout);
    else if(data?.type==='request-layout') workspaceChannel.postMessage({type:'layout',layout:workspace});
  });
  window.addEventListener('storage',event=>{
    if(event.key===RANGE_KEY && event.newValue) {try {receiveRange(JSON.parse(event.newValue));} catch { /* Ignore invalid ranges. */ }}
    if(event.key===STORAGE_KEY && event.newValue) {
      try { receiveLayout(JSON.parse(event.newValue)); } catch { /* Ignore invalid stored layouts. */ }
    }
  });
  workspaceChannel?.postMessage({type:'request-layout'});
  let revision=-1;
  while(active) {
    try {
      const next=await request(`/api/state?after=${revision}`);
      if(next.catalog && !next.checkpoint) { location.href=`/browse#${new URLSearchParams({token})}`; return; }
      if(next.catalog) {
        const browse=$('#browse-checkpoints'); browse.hidden=false;
        const saved=sessionStorage.getItem('gymemu-catalog-route');
        const destination=new URL(saved || '/browse',location.href);
        if(destination.origin!==location.origin) destination.href=new URL('/browse',location.href).href;
        destination.hash=new URLSearchParams({token}).toString();
        browse.href=destination.href;
        browse.onclick=async event=>{event.preventDefault();await command({type:'pause'});location.href=browse.href;};
        if(!isPlayer) companionUrl.pathname='/player';
        companion.href=companionUrl.href;
      }
      if(snapshot && next.generation!==snapshot.generation) { location.reload(); return; }
      if(next.revision!==revision) {
        const decoded=await decode(next.images);
        const changed=chartIdentity(snapshot)!==chartIdentity(next);
        bitmaps=decoded;snapshot=next;revision=next.revision;
        if(changed) {
          chartHoverStep=null;chartRange=null;rangeRevision={clock:0,writer:''};renderChartRange();
          try {const saved=JSON.parse(localStorage.getItem(RANGE_KEY));if(saved)receiveRange(saved);} catch { /* Ignore unavailable storage. */ }
          workspaceChannel?.postMessage({type:'request-chart-range'});
        }
        chrome(next);runtime.renderSnapshot(next,{bitmaps,chartRange,chartHoverStep});
        document.body.dataset.revision=String(next.revision);
        if(next.error) showToast(next.error);
      }
    } catch(error) {
      showToast(error.message);
      await new Promise(resolve=>setTimeout(resolve,1000));
    }
  }
}
boot().catch(error=>showToast(error.message));
