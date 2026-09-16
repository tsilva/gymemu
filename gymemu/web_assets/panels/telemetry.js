import { createPanel, setStats, drawLines, themeColor, lineCursorIndex } from './shared.js';
import { createChartTooltip } from './chart-tooltip.js';
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
export function chartPoint(history, geometry, x) {
  const plot = geometry?.plot;
  if (!plot || x < plot.left || x > plot.right) return null;
  return history[lineCursorIndex(plot, x, history.length)] || null;
}
export function mount({ definition, services }) {
  const element = createPanel({id: definition.id, label: definition.title, body: `
    <div class="stats-grid"></div>
    <div class="chart-wrap"><canvas aria-label="RGB MSE history" tabindex="0"></canvas><p class="chart-empty"></p></div>`});
  const chart = element.querySelector('.chart-wrap'), canvas = element.querySelector('canvas');
  const renderTooltip = createChartTooltip(canvas);
  let latest = null, view = {}, geometry = null, clickTimer = null, contextKey = null;
  const points = () => measuredHistory(latest, view.chartRange);
  const draw = () => {
    if (!latest || definition.config.view !== 'chart') return;
    const history = points(), all = measuredHistory(latest);
    const steps = history.map(p=>p.step);
    const hoverStep = view.chartHoverStep;
    const series = [{label:'RGB MSE',values:history.map(p=>p.mse),color:themeColor('seriesViolet')}];
    geometry = drawLines(canvas,series,{steps,showStepTicks:true,connectGaps:false,
      cursorStep:Number.isFinite(hoverStep)?hoverStep:latest.step});
    renderTooltip({steps,series,step:hoverStep,geometry});
    canvas.dataset.hoverStep = Number.isFinite(hoverStep) ? String(hoverStep) : '';
    canvas.dataset.selectedStep = String(latest.step);
    element.querySelector('.chart-empty').textContent = latest.mode !== 'teacher-forcing' ? 'RGB MSE requires recorded targets.'
      : !all.length ? 'Step or play to measure prediction errors.' : !history.length ? 'No measured steps in this zoom range.' : '';

  };
  const inspect = (step, identity = latest) => services.inspectStep?.(step, identity.selection, identity.history_epoch);
  bindChartRange(canvas,()=>geometry,()=>({history:points(),view}),services);
  canvas.addEventListener('pointermove',event=>{
    const bounds=canvas.getBoundingClientRect(), plot=geometry?.plot, history=points();
    if(!plot || !history.length || !(bounds.width>0)) return;
    const x=(event.clientX-bounds.left)*canvas.clientWidth/bounds.width;
    const fraction=Math.max(0,Math.min(1,(x-plot.left)/(plot.right-plot.left)));
    services.setChartHoverStep?.(history[0].step+fraction*(history.at(-1).step-history[0].step));
  });
  for(const event of ['pointerleave','pointercancel']) canvas.addEventListener(event,()=>services.setChartHoverStep?.(null));
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
    const hovered=Number.isFinite(view.chartHoverStep) ? history.reduce((best,p)=>Math.abs(p.step-view.chartHoverStep)<Math.abs(best.step-view.chartHoverStep)?p:best) : null;
    let index=history.findIndex(p=>p.step===(hovered?.step ?? latest.step));
    if(event.key==='Home') index=0;
    else if(event.key==='End') index=history.length-1;
    else if(event.key==='ArrowLeft') index=Math.max(0,index-1);
    else if(event.key==='ArrowRight') index=Math.min(history.length-1,index+1);
    else {if(index>=0) inspect(history[index].step);return;}
    services.setChartHoverStep?.(history[index]?.step ?? null);
  });
  chart.hidden = definition.config.view !== 'chart';
  return {element, render(s,nextView={}) {
    if (!s) return;
    const nextKey=`${s.mode}:${s.selection}:${s.history_epoch}`;
    if(nextKey!==contextKey){clearTimeout(clickTimer);contextKey=nextKey;}
    latest=s;view=nextView;
    setStats(element.querySelector('.stats-grid'), (definition.config.metrics||[]).map(key => {
      const m=METRICS[key]; return [m?.label||key,s[key] == null ? '—' : m?.format(s[key])];
    })); draw();
  },resize:draw,destroy(){clearTimeout(clickTimer);}};
}
