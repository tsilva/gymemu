// Adapted from Gradlab's type registry and built-in panel presets.
export const PANEL_TYPES = Object.freeze({
  frame: { module: './frame.js', minimum: { w: 2, h: 6 }, frameKinds: [] },
  history: { module: './history.js', minimum: { w: 4, h: 5 }, frameKinds: [] },
  telemetry: { module: './telemetry.js', minimum: { w: 3, h: 5 }, frameKinds: [] },
});
export const BUILTIN_PANEL_PRESETS = Object.freeze({
  original: { type: 'frame', title: 'Original', config: { source: 'original' }, placement: { x: 0, y: 0, w: 4, h: 14 } },
  prediction: { type: 'frame', title: 'Prediction', config: { source: 'prediction' }, placement: { x: 4, y: 0, w: 4, h: 14 } },
  difference: { type: 'frame', title: 'Prediction − original', config: { source: 'difference', gain: 1 }, placement: { x: 8, y: 0, w: 4, h: 14 } },
  history: { type: 'history', title: 'Input history', config: {}, placement: { x: 0, y: 0, w: 12, h: 16 } },
  metrics: { type: 'telemetry', title: 'Prediction error', config: { view: 'chart', metrics: ['mse', 'inference_ms', 'step'] }, placement: { x: 0, y: 16, w: 8, h: 11 } },
  model: { type: 'telemetry', title: 'Model context', config: { view: 'stats', metrics: ['history_length', 'action_history', 'action'] }, placement: { x: 8, y: 16, w: 4, h: 11 } },
});
export function panelDefinition(workspace, id) {
  const panel = workspace.panels[id];
  if (!panel || !PANEL_TYPES[panel.type]) return null;
  return { ...PANEL_TYPES[panel.type], ...panel, minimum: panel.type==='telemetry' && panel.config.view==='chart' ? {w:3,h:11} : PANEL_TYPES[panel.type].minimum, id, label: panel.title, frameKinds: [], enabled: panel.enabled !== false };
}
