import { flushSync, mount } from 'svelte';
import Shell from './Shell.svelte';

window.gymemuPanelModules = {
  './frame.js': () => import('../gymemu/web_assets/panels/frame.js'),
  './history.js': () => import('../gymemu/web_assets/panels/history.js'),
  './telemetry.js': () => import('../gymemu/web_assets/panels/telemetry.js'),
};
mount(Shell, { target: document.body });
flushSync();
await import('../gymemu/web_assets/app.js');
