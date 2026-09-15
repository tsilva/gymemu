import { BUILTIN_PANEL_PRESETS, PANEL_TYPES } from './catalog.js';
export const WORKSPACE_VERSION = 3;
export const STORAGE_KEY = 'gymemu.player.workspace.v1';
export const panelWindow = type => type === 'frame' ? 'main' : 'stats';
export function compareWorkspaceRevisions(a, b) {
  return (a?.clock || 0) - (b?.clock || 0) || (a?.writer || '').localeCompare(b?.writer || '');
}
export function defaultWorkspace() {
  return { version: WORKSPACE_VERSION, revision: {clock: 0, writer: ''}, panels: Object.fromEntries(Object.entries(BUILTIN_PANEL_PRESETS).map(([id, panel]) => [id, {
    ...structuredClone(panel), builtin: true, enabled: true, placement: { ...panel.placement, visible: true, window: panelWindow(panel.type) },
  }])) };
}
export function normalizeWorkspace(saved) {
  const defaults = defaultWorkspace();
  if (![1, 2, WORKSPACE_VERSION].includes(saved?.version) || !saved.panels) return defaults;
  if (Number.isSafeInteger(saved.revision?.clock) && saved.revision.clock >= 0 && typeof saved.revision.writer === 'string') {
    defaults.revision = {clock:saved.revision.clock, writer:saved.revision.writer.slice(0,80)};
  }
  for (const [id, value] of Object.entries(saved.panels).slice(0, 40)) {
    if (!/^[a-z][a-z0-9-]{0,60}$/.test(id) || !PANEL_TYPES[value?.type]) continue;
    if (!defaults.panels[id] && value.type !== 'telemetry') continue;
    const fallback = defaults.panels[id] || defaults.panels.metrics;
    const minimum = PANEL_TYPES[value.type].minimum;
    const integer = (v, min, max, d) => Number.isInteger(v) ? Math.max(min, Math.min(max, v)) : d;
    const w = integer(value.placement?.w, minimum.w, 12, fallback.placement.w);
    defaults.panels[id] = {
      type: fallback.type, title: String(value.title || fallback.title).slice(0, 80),
      builtin: Boolean(BUILTIN_PANEL_PRESETS[id]), enabled: value.enabled !== false,
      config: { ...fallback.config,
        ...(fallback.type === 'telemetry' ? { view: value.config?.view === 'stats' ? 'stats' : 'chart',
          metrics: Array.isArray(value.config?.metrics) ? value.config.metrics.filter(v => ['mse','step','inference_ms','action','history_length','action_history'].includes(v)) : fallback.config.metrics } : {}),
        ...(id === 'difference' ? { gain: [1, 2, 4, 8].includes(value.config?.gain) ? value.config.gain : 1 } : {}),
      },
      placement: { x: integer(value.placement?.x, 0, 12-w, 0),
        y: Math.max(0, integer(value.placement?.y, 0, 200, 0) - (saved.version < 3 && fallback.type !== 'frame' ? 14 : 0)),
        w, h: integer(value.placement?.h, minimum.h, 40, fallback.placement.h), visible: value.placement?.visible !== false, window: panelWindow(fallback.type) },
    };
  }
  if (saved.version === 1 && ['prediction', 'original', 'difference'].every((id, index) => {
    const placement = saved.panels[id]?.placement;
    return placement?.x === index * 4 && placement.y === 0 && placement.w === 4 && placement.h === 14;
  })) {
    defaults.panels.original.placement.x = 0;
    defaults.panels.prediction.placement.x = 4;
  }
  // Upgrade the former default diagnostic arrangement without moving custom layouts.
  const history = defaults.panels.history.placement, metrics = defaults.panels.metrics.placement, model = defaults.panels.model.placement;
  if (history.x===0 && history.y===0 && history.w===8 && history.h===10
      && metrics.x===0 && metrics.y===10 && metrics.w===8 && [7,11].includes(metrics.h)
      && model.x===8 && model.y===0 && model.w===4 && model.h===7) {
    for (const id of ['history','metrics','model'])
      Object.assign(defaults.panels[id].placement, BUILTIN_PANEL_PRESETS[id].placement);
  }
  return defaults;
}
export function loadWorkspace(storage = localStorage) {
  try { return normalizeWorkspace(JSON.parse(storage.getItem(STORAGE_KEY))); }
  catch { return defaultWorkspace(); }
}
export function saveWorkspace(workspace, storage = localStorage) {
  try { storage.setItem(STORAGE_KEY, JSON.stringify(workspace)); } catch { /* Layout still works without storage. */ }
}
