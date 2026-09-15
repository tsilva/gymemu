import { createPanel, setStats, drawLines, themeColor, lineCursorIndex } from './shared.js';
import { bindChartRange } from '../chart-range.js';
export const METRICS = Object.freeze({
  mse: { label: 'RGB MSE', format: v => v.toFixed(7) },
  inference_ms: { label: 'Inference', format: v => `${v.toFixed(1)} ms` },
  step: { label: 'Frame', format: String },
  action: { label: 'Executed action', format: String },
  history_length: { label: 'RGB history', format: v => `${v} frames` },
  action_history: { label: 'Action history', format: v => `${v} actions` },
});
export function measuredHistory(snapshot, range = null) {
  if (snapshot?.mode !== 'teacher-forcing') return [];
  return (snapshot.history || []).filter(p => Number.isInteger(p.step) && Number.isFinite(p.mse)
    && (!range || (p.step >= range.first && p.step <= range.last)));
}
export function highestMse(history) {
  return history.reduce((peak, point) => !peak || point.mse > peak.mse ? point : peak, null);
}
export function chartPoint(history, geometry, x) {
  const plot = geometry?.plot;
  if (!plot || x < plot.left || x > plot.right) return null;
  return history[lineCursorIndex(plot, x, history.length)] || null;
}
export function mount({ definition, services }) {
  const element = createPanel({id: definition.id, label: definition.title, body: `
    <div class="stats-grid"></div>
    <div class="mse-chart-controls"><button data-peak type="button" disabled>Highest measured MSE</button><button data-reset-zoom type="button" hidden>Reset zoom</button><span data-coverage></span></div>
    <div class="chart-wrap"><canvas aria-label="RGB MSE history" tabindex="0"></canvas><p class="chart-empty"></p></div>
    <footer class="chart-footer"><span class="chart-readout"></span><span>Hover to inspect · Click to select · Drag to zoom · Double-click to reset</span></footer>`});
  const chart = element.querySelector('.chart-wrap'), canvas = element.querySelector('canvas');
  const controls = element.querySelector('.mse-chart-controls'), footer = element.querySelector('.chart-footer');
  const peakButton = element.querySelector('[data-peak]'), resetZoom = element.querySelector('[data-reset-zoom]');
  let latest = null, view = {}, geometry = null, hoverX = null, clickTimer = null, contextKey = null;
  const points = () => measuredHistory(latest, view.chartRange);
  const draw = () => {
    if (!latest || definition.config.view !== 'chart') return;
    const history = points(), all = measuredHistory(latest);
    const options = {steps:history.map(p=>p.step),showStepTicks:true,connectGaps:false,
      referenceStep:latest.step};
    geometry = drawLines(canvas,[{values:history.map(p=>p.mse),color:themeColor('seriesViolet')}],options);
    const hovered = hoverX === null ? null : chartPoint(history,geometry,hoverX);
    if (hovered) geometry = drawLines(canvas,[{values:history.map(p=>p.mse),color:themeColor('seriesViolet')}],
      {...options,cursorStep:hovered.step,cursorLabel:`Step ${hovered.step}`});
    const current = hovered || all.find(p=>p.step===latest.step);
    element.querySelector('.chart-readout').textContent = current ? `Step ${current.step} · MSE ${current.mse.toFixed(7)}` : 'No measured point selected';
    canvas.dataset.hoverStep = hovered ? String(hovered.step) : '';
    canvas.dataset.selectedStep = String(latest.step);
    element.querySelector('.chart-empty').textContent = latest.mode !== 'teacher-forcing' ? 'RGB MSE requires recorded targets.'
      : !all.length ? 'Step or play to measure prediction errors.' : !history.length ? 'No measured steps in this zoom range.' : '';
    element.querySelector('[data-coverage]').textContent = latest.mode === 'teacher-forcing' ? `${all.length} / ${latest.total_steps} transitions measured` : '';
    const peak = highestMse(all);
    peakButton.disabled = !peak || Boolean(latest.loading);
    peakButton.textContent = peak ? `Highest MSE: ${peak.mse.toFixed(7)} · Step ${peak.step}` : 'Highest measured MSE';
    resetZoom.hidden = !view.chartRange;
    resetZoom.textContent = view.chartRange ? `Steps ${view.chartRange.first}–${view.chartRange.last} · Reset zoom` : 'Reset zoom';
  };
  const inspect = (step, identity = latest) => services.inspectStep?.(step, identity.selection, identity.history_epoch);
  peakButton.onclick = () => {const peak=highestMse(measuredHistory(latest));if(peak) inspect(peak.step);};
  resetZoom.onclick = () => services.setChartRange?.(null);
  bindChartRange(canvas,()=>geometry,()=>({history:points()}),services);
  canvas.addEventListener('pointermove',event=>{const b=canvas.getBoundingClientRect();hoverX=(event.clientX-b.left)*canvas.clientWidth/b.width;draw();});
  canvas.addEventListener('pointerleave',()=>{hoverX=null;draw();});
  canvas.addEventListener('click',event=>{
    const b=canvas.getBoundingClientRect();
    const point=chartPoint(points(),geometry,(event.clientX-b.left)*canvas.clientWidth/b.width);
    clearTimeout(clickTimer);
    if(point) {const identity=latest;clickTimer=setTimeout(()=>inspect(point.step,identity),250);}
  });
  canvas.addEventListener('dblclick',()=>clearTimeout(clickTimer));
  canvas.addEventListener('keydown',event=>{
    if (!['ArrowLeft','ArrowRight','Home','End','Enter',' '].includes(event.key)) return;
    const history=points();if(!history.length)return;
    event.preventDefault();
    const hovered=hoverX===null?null:chartPoint(history,geometry,hoverX);
    let index=history.findIndex(p=>p.step===(hovered?.step ?? latest.step));
    if(event.key==='Home') index=0;
    else if(event.key==='End') index=history.length-1;
    else if(event.key==='ArrowLeft') index=Math.max(0,index-1);
    else if(event.key==='ArrowRight') index=Math.min(history.length-1,index+1);
    else {if(index>=0) inspect(history[index].step);return;}
    hoverX=geometry?.plot.positions[index] ?? null;draw();
  });
  chart.hidden = controls.hidden = footer.hidden = definition.config.view !== 'chart';
  return {element, render(s,nextView={}) {
    if (!s) return;
    const nextKey=`${s.mode}:${s.selection}:${s.history_epoch}`;
    if(nextKey!==contextKey){hoverX=null;clearTimeout(clickTimer);contextKey=nextKey;}
    latest=s;view=nextView;
    setStats(element.querySelector('.stats-grid'), (definition.config.metrics||[]).map(key => {
      const m=METRICS[key]; return [m?.label||key,s[key] == null ? '—' : m?.format(s[key])];
    })); draw();
  },resize:draw,destroy(){clearTimeout(clickTimer);}};
}
