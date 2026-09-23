const $ = selector => document.querySelector(selector);
const hash = new URLSearchParams(location.hash.slice(1));
const token = hash.get('token') || sessionStorage.getItem('gymemu-token');
if (token) sessionStorage.setItem('gymemu-token', token);
const desktopId = new URLSearchParams(location.search).get('desktop');
const headers = { 'X-Player-Token': token || '', 'Content-Type': 'application/json' };
let catalog = null, loading = false;

async function request(path, options = {}) {
  const response = await fetch(path, { ...options, headers });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
  return result;
}
function showError(message) {
  $('#catalog-error').textContent = message;
  $('#catalog-error').hidden = !message;
}
function route(env, run) {
  const query = new URLSearchParams();
  if (env) query.set('env', env);
  if (run) query.set('run', run);
  if (desktopId) query.set('desktop', desktopId);
  return `/browse${query.size ? `?${query}` : ''}#${new URLSearchParams({ token: token || '' })}`;
}
function link(label, href) {
  const a = document.createElement('a');
  a.textContent = label; a.href = href;
  a.onclick = event => {
    if (event.button || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    if (loading) return;
    history.pushState(null, '', href);
    $('#search').value = '';
    showError(''); refresh(false).then(() => $('#catalog-title').focus());
  };
  return a;
}
function cell(value, className = '') {
  const td = document.createElement('td'); td.className = className;
  if (value instanceof Node) td.append(value); else td.textContent = value;
  return td;
}
function date(value) { return value ? new Date(value * 1000).toLocaleString() : '—'; }
async function openCheckpoint(item, button) {
  if (loading) return;
  loading = true; showError('');
  button.textContent = `Loading ${item.name}…`;
  $('#catalog-status').hidden = false;
  $('#catalog-status').textContent = `Loading ${item.name}…`;
  document.querySelectorAll('#catalog-items button, #refresh, #search').forEach(el => { el.disabled = true; });
  try {
    await request('/api/open', { method: 'POST', body: JSON.stringify({ checkpoint: item.id }) });
    sessionStorage.setItem('gymemu-catalog-route', location.href);
    location.href = `/player${desktopId ? `?desktop=${desktopId}` : ''}#${new URLSearchParams({ token: token || '' })}`;
  } catch (error) {
    loading = false;
    $('#refresh').disabled = $('#search').disabled = false;
    showError(error.message); render();
  }
}
function render() {
  if (!catalog) return;
  const params = new URLSearchParams(location.search);
  const env = catalog.environments.find(item => item.id === params.get('env'));
  const run = env?.runs.find(item => item.id === params.get('run'));
  if ((params.has('env') && !env) || (params.has('run') && !run)) {
    history.replaceState(null, '', route(env?.id));
    showError('This selection is no longer available. Choose another item or refresh.');
  }
  const crumbs = $('#breadcrumbs'); crumbs.replaceChildren();
  const levels = [['Environments', route()]];
  if (env) levels.push([env.id, route(env.id)]);
  if (run) levels.push([run.name, route(env.id, run.id)]);
  for (const [index, [label, href]] of levels.entries()) {
    if (index) { const separator = document.createElement('span'); separator.textContent = '/'; separator.setAttribute('aria-hidden', 'true'); crumbs.append(separator); }
    const a = link(label, href);
    if (index === levels.length - 1) a.setAttribute('aria-current', 'page');
    crumbs.append(a);
  }
  const title = run ? 'Checkpoints' : env ? 'Training runs' : 'Environments';
  $('#catalog-title').textContent = title;
  document.title = `${title} · Gymemu`;
  $('#catalog-description').textContent = run ? `${run.name} · ${run.approach}` : env ? env.id : 'Choose an environment to browse its training runs.';
  $('#search').placeholder = `Search ${title.toLowerCase()}`;
  $('#search').setAttribute('aria-label', $('#search').placeholder);
  const columns = run ? ['Checkpoint', 'Source', 'Size', 'Saved'] : env ? ['Training run', 'Source', 'Approach', 'Checkpoints', 'Latest save'] : ['Environment ID', 'Training runs', 'Checkpoints'];
  const heading = document.createElement('tr');
  for (const label of columns) { const th = document.createElement('th'); th.scope = 'col'; th.textContent = label; heading.append(th); }
  $('#catalog-columns').replaceChildren(heading);
  const all = run ? run.checkpoints : env ? env.runs : catalog.environments;
  const query = $('#search').value.trim().toLowerCase();
  const items = all.filter(item => `${item.name || item.id} ${item.approach || ''}`.toLowerCase().includes(query));
  const body = $('#catalog-items'); body.replaceChildren();
  for (const item of items) {
    const row = document.createElement('tr');
    if (run) {
      const button = document.createElement('button'); button.type = 'button'; button.textContent = item.name;
      button.onclick = () => openCheckpoint(item, button);
      row.append(cell(button, 'catalog-name'), cell(item.source || 'Local'), cell(`${(item.size_bytes / 1024 / 1024).toFixed(2)} MB`), cell(date(item.modified), 'catalog-date'));
    } else if (env) {
      row.append(cell(link(item.run_id ? `${item.name} · ${item.run_id.slice(0, 8)}` : item.name, route(env.id, item.id)), 'catalog-name'), cell(item.source || 'Local'), cell(item.approach), cell(item.checkpoint_count ?? item.checkpoints.length), cell(date(item.modified), 'catalog-date'));
    } else {
      row.append(cell(link(item.id, route(item.id)), 'catalog-name'), cell(item.runs.length), cell(item.runs.reduce((sum, r) => sum + (r.checkpoint_count ?? r.checkpoints.length), 0)));
    }
    body.append(row);
  }
  $('#item-count').textContent = `${items.length} of ${all.length}`;
  $('#catalog-status').hidden = true;
  $('.catalog-table').hidden = !items.length;
  $('#catalog-empty').hidden = Boolean(items.length);
  $('#catalog-empty').textContent = query ? 'No items match this filter.' : run ? 'This run has no checkpoints yet. Refresh after the first checkpoint is saved.' : 'No training runs found. Train an emulator or choose a directory with --runs-dir.';
  $('#catalog-root').textContent = `Local runs: ${catalog.root}${catalog.remote ? ' · R2 checkpoints included' : ''}`;
  $('#catalog-warnings').hidden = !catalog.warnings.length;
  const warnings = $('#catalog-warnings ul'); warnings.replaceChildren();
  for (const warning of catalog.warnings) { const li = document.createElement('li'); li.textContent = warning; warnings.append(li); }
}
function setRefreshing(refreshing) {
  const button = $('#refresh');
  button.disabled = refreshing || loading;
  button.classList.toggle('refreshing', refreshing);
  button.setAttribute('aria-label', refreshing ? 'Refreshing' : 'Refresh');
  button.setAttribute('aria-busy', String(refreshing));
  button.title = refreshing ? 'Refreshing this list' : 'Refresh this list';
}
let requestId = 0;
async function refresh(reload = true) {
  const currentRequest = ++requestId;
  if (loading) return;
  setRefreshing(true);
  $('#catalog-status').hidden = false; $('#catalog-status').textContent = 'Loading runs…';
  showError('');
  try {
    const params = new URLSearchParams(location.search);
    const query = new URLSearchParams();
    if(params.has('run')) query.set('run', params.get('run'));
    if(reload) query.set('refresh', '1');
    const result = await request(`/api/catalog?${query}`);
    if(currentRequest !== requestId) return;
    catalog = result; render();
  }
  catch (error) { if(currentRequest === requestId) { showError(error.message); $('#catalog-status').hidden = true; } }
  finally { if(currentRequest === requestId) setRefreshing(false); }
}
$('.app-wordmark').href = route();
$('#search').oninput = render;
$('#search-disclosure').addEventListener('toggle', () => {
  if ($('#search-disclosure').open) requestAnimationFrame(() => $('#search').focus({ preventScroll: true }));
});
$('#search-close').onclick = () => {
  $('#search-disclosure').open = false;
  $('#search').value = '';
  render();
  $('#search-disclosure summary').focus();
};
$('#refresh').onclick = () => refresh(true);
window.addEventListener('popstate', () => { $('#search').value = ''; refresh(false); });
if (!token) { $('#catalog-status').hidden = true; showError('Open the complete navigator URL printed in your terminal.'); }
else refresh();
