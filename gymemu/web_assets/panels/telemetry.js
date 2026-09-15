import { createPanel, setStats, drawLines, themeColor } from './shared.js';
export const METRICS = Object.freeze({
  mse: { label: 'RGB MSE', format: v => v.toFixed(7) },
  inference_ms: { label: 'Inference', format: v => `${v.toFixed(1)} ms` },
  step: { label: 'Frame', format: String },
  action: { label: 'Executed action', format: String },
  history_length: { label: 'RGB history', format: v => `${v} frames` },
  action_history: { label: 'Action history', format: v => `${v} actions` },
});
export function mount({ definition }) {
  const element = createPanel({id: definition.id, label: definition.title, body: '<div class="stats-grid"></div><div class="chart-wrap"><canvas aria-label="RGB MSE history"></canvas><p class="chart-empty"></p></div>'});
  const chart = element.querySelector('.chart-wrap'), canvas = element.querySelector('canvas');
  let latest = null;
  const draw = () => {
    if (!latest || definition.config.view !== 'chart') return;
    const history = latest.history;
    element.querySelector('.chart-empty').textContent = latest.mode !== 'teacher-forcing' ? 'RGB MSE requires recorded targets.' : history.length ? '' : 'Step to collect prediction errors.';
    drawLines(canvas, [{values:history.map(p=>p.mse),color:themeColor('seriesViolet')}], {steps:history.map(p=>p.step),showStepTicks:true});
  };
  chart.hidden = definition.config.view !== 'chart';
  return {element, render(s) {
    if (!s) return; latest=s;
    setStats(element.querySelector('.stats-grid'), (definition.config.metrics||[]).map(key => {
      const m=METRICS[key]; return [m?.label||key,s[key] == null ? '—' : m?.format(s[key])];
    })); draw();
  },resize:draw};
}
