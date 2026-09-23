import { flushSync, mount } from 'svelte';
import Catalog from './Catalog.svelte';

mount(Catalog, { target: document.body });
flushSync();
await import('../gymemu/web_assets/catalog.js');
