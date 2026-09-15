import { BUILTIN_PANEL_PRESETS, PANEL_TYPES } from './catalog.js';
export const WORKSPACE_VERSION = 2;
export const STORAGE_KEY = 'gymemu.player.workspace.v1';
export function defaultWorkspace() {
  return { version: WORKSPACE_VERSION, panels: Object.fromEntries(Object.entries(BUILTIN_PANEL_PRESETS).map(([id, panel]) => [id, {
    ...structuredClone(panel), builtin: true, enabled: true, placement: { ...panel.placement, visible: true, window: 'main' },
  }])) };
}
export function normalizeWorkspace(saved) {
  const defaults = defaultWorkspace();
  if (![1, WORKSPACE_VERSION].includes(saved?.version) || !saved.panels) return defaults;
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
      placement: { x: integer(value.placement?.x, 0, 12-w, 0), y: integer(value.placement?.y, 0, 200, 0),
        w, h: integer(value.placement?.h, minimum.h, 40, fallback.placement.h), visible: value.placement?.visible !== false, window: 'main' },
    };
  }
  if (saved.version === 1 && ['prediction', 'original', 'difference'].every((id, index) => {
    const placement = saved.panels[id]?.placement;
    return placement?.x === index * 4 && placement.y === 0 && placement.w === 4 && placement.h === 14;
  })) {
    defaults.panels.original.placement.x = 0;
    defaults.panels.prediction.placement.x = 4;
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
