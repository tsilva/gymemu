import { PanelRuntime } from './panels/runtime.js';
import { PANEL_TYPES, panelDefinition } from './panels/catalog.js';
import { defaultWorkspace, normalizeWorkspace, loadWorkspace, saveWorkspace } from './panels/workspace.js';
import { METRICS } from './panels/telemetry.js';
import { mountPlaybackControls } from './playback-settings.js';
import { setSvgUseHref } from './panels/shared.js';

const $ = selector => document.querySelector(selector);
const token = new URLSearchParams(location.hash.slice(1)).get('token') || sessionStorage.getItem('gymemu-token');
if (token) sessionStorage.setItem('gymemu-token', token);
let workspace = loadWorkspace(), snapshot = null, bitmaps = {}, active = true, syncing = false, pendingCommand = Promise.resolve();
let runtime, grid, toastTimer, saveTimer;
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
  pendingCommand = pendingCommand.then(() => request('/api/command',{method:'POST',body:JSON.stringify(value)})).catch(error => showToast(error.message));
  return pendingCommand;
}
function persist(immediate=false) {
  saveWorkspace(workspace);
  clearTimeout(saveTimer);
  const save=()=>{saveTimer=null;return request('/api/workspace',{method:'POST',body:JSON.stringify(workspace),keepalive:true}).catch(e=>showToast(e.message));};
  if(immediate) return save();
  saveTimer=setTimeout(save,250);
}

function placement(id) {
  const panel = workspace.panels[id], min=PANEL_TYPES[panel.type].minimum;
  return {...panel.placement,minW:min.w,minH:min.h,id};
}
function shelf() {
  const target=$('#panel-shelf-items'); target.replaceChildren();
  Object.entries(workspace.panels).filter(([,p])=>!p.placement.visible).forEach(([id,p])=>{
    const button=document.createElement('button'); button.textContent=p.title; button.setAttribute('aria-label',`Show ${p.title}`);
    button.onclick=()=>{p.placement.visible=true;syncWorkspace();persist();};target.append(button);
  });
  if (!target.children.length) target.textContent='Every panel is visible.';
}
async function syncWorkspace() {
  syncing=true;
  try { await runtime.sync(workspace,'main'); }
  finally { syncing=false; }
  shelf();
  if(snapshot) runtime.renderSnapshot(snapshot,{bitmaps});
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
  else {workspace.panels[`panel-${crypto.randomUUID()}`]={type:'telemetry',title,config,builtin:false,enabled:true,placement:{x:0,y:40,w:6,h:8,visible:true,window:'main'}};syncWorkspace();persist();}
  $('#panel-editor').close();
};
$('#add-panel').onclick=()=>openEditor();
$('#panels-toggle').onclick=()=>{$('#panel-shelf').hidden=!$('#panel-shelf').hidden;};
$('#reset-layout').onclick=async()=>{workspace=defaultWorkspace();await syncWorkspace();await persist(true);showToast('Default workspace restored.');};
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
function timelinePosition(s, step=s.step) {
  const name=s.mode==='teacher-forcing' ? `EPISODE ${s.selection}` : s.name.toUpperCase();
  return `${name} · STEP ${step}${s.finished&&step===s.step?' · END':''}`;
}
function updateScrubberProgress(slider) {
  slider.style.setProperty('--timeline-progress',`${Number(slider.max)>0?100*Number(slider.value)/Number(slider.max):0}%`);
}
$('#timeline-scrubber').oninput=()=>{ $('#position').textContent=timelinePosition(snapshot,Number($('#timeline-scrubber').value));updateScrubberProgress($('#timeline-scrubber')); };
$('#timeline-scrubber').onchange=()=>command({type:'seek',position:Number($('#timeline-scrubber').value)});
function keyName(event) { return event.key === ' ' ? 'space' : event.key.toLowerCase().replace(/^arrow/,''); }
const pressed=new Set();
document.addEventListener('keydown',event=>{
  if(event.target.closest('input,select,textarea,dialog') || !snapshot) return;
  const key=keyName(event);
  if(!['tab','r','c','escape',...Object.keys(snapshot.keymap)].includes(key)) return;
  event.preventDefault(); if(event.repeat || pressed.has(key)) return;
  pressed.add(key);command({type:'key',key,down:true});
});
document.addEventListener('keyup',event=>{const key=keyName(event);if(pressed.delete(key)) command({type:'key',key,down:false});});
const blur=()=>{pressed.clear();command({type:'blur'});};
window.addEventListener('blur',blur);
document.addEventListener('visibilitychange',()=>{if(document.hidden) blur();});
window.addEventListener('pagehide',()=>{if(saveTimer) persist(true);active=false;fetch('/api/command',{method:'POST',headers,body:JSON.stringify({type:'blur'}),keepalive:true}).catch(()=>{});});
setInterval(()=>{if(active && !document.hidden) request('/api/command',{method:'POST',body:JSON.stringify({type:'heartbeat'})}).catch(()=>{});},400);
function chrome(s) {
  $('#checkpoint').textContent=s.checkpoint.split('/').slice(-2).join('/');$('#checkpoint').title=s.checkpoint;
  $('#mode').value=s.mode;$('#mode').disabled=!s.available_modes?.length||Boolean(s.loading);
  $('#position').textContent=timelinePosition(s);
  const play=$('#play-toggle'), action=s.playing?'pause':'play', label=s.playing?'Pause':'Play';
  play.dataset.action=action;play.setAttribute('aria-label',label);play.title=label;
  setSvgUseHref($('#play-toggle-icon'),`/assets/tabler-icons.svg#ti-player-${action}`);
  play.disabled=s.finished||Boolean(s.loading);
  $('#reset-playback').disabled=Boolean(s.loading);settingsToggle.disabled=Boolean(s.loading);
  previousStep.disabled=s.mode!=='teacher-forcing'||s.step===0||Boolean(s.loading);
  playbackSettings.render(s);
  const slider=$('#timeline-scrubber');slider.disabled=s.mode!=='teacher-forcing'||Boolean(s.loading);slider.max=s.total_steps||0;if(document.activeElement!==slider) slider.value=s.step;
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
  grid=window.GridStack.init({alwaysShowResizeHandle:true,animate:false,cellHeight:32,column:12,columnOpts:{breakpointForWindow:true,breakpoints:[{w:640,c:1},{w:960,c:6}]},draggable:{handle:'.panel-drag',scroll:true},float:false,margin:5,maxRow:240,resizable:{handles:'se'}},$('#dashboard'));
  grid.on('change',(_event,nodes)=>{if(syncing)return;nodes.forEach(node=>{const p=workspace.panels[node.id];if(p && grid.getColumn()===12) for(const key of ['x','y','w','h']) p.placement[key]=node[key];});persist();});
  runtime=new PanelRuntime({definitionFor:panelDefinition,container:$('#dashboard'),services:{getState:()=>snapshot,command,showToast,updatePanel},
    onMount:(element,id,_definition,item)=>{grid.makeWidget(item,placement(id));element.querySelector('[data-panel-menu]').onclick=e=>{e.stopPropagation();menu(id,e.currentTarget);};},
    onLayout:(element,id,_place,item,definition)=>{grid.update(item,placement(id));element.classList.toggle('disabled',!definition.enabled);},
    onUnmount:(_element,_id,item)=>grid.removeWidget(item,false),onError:(id,error)=>showToast(`${id}: ${error.message}`)});
  await syncWorkspace();
  let revision=-1;
  while(active) {
    try {
      const next=await request(`/api/state?after=${revision}`);
      if(next.revision!==revision) {
        const decoded=await decode(next.images);bitmaps=decoded;snapshot=next;revision=next.revision;
        chrome(next);runtime.renderSnapshot(next,{bitmaps});
        if(next.error) showToast(next.error);
      }
      $('#connection').textContent=snapshot?.loading?'Loading':snapshot?.playing?'Playing':'Paused';$('#connection').classList.remove('error');
    } catch(error) {
      $('#connection').textContent='Disconnected';$('#connection').classList.add('error');showToast(error.message);
      await new Promise(resolve=>setTimeout(resolve,1000));
    }
  }
}
boot().catch(error=>{$('#connection').textContent='Connection failed';$('#connection').classList.add('error');showToast(error.message);});
