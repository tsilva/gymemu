const $ = selector => document.querySelector(selector);
const hash = new URLSearchParams(location.hash.slice(1));
const token = hash.get('token') || sessionStorage.getItem('gymemu-token');
if (token) sessionStorage.setItem('gymemu-token', token);
const desktopId = new URLSearchParams(location.search).get('desktop');
const headers = { 'X-Player-Token': token || '', 'Content-Type': 'application/json' };
let catalog = null;
let loading = false;
let requestId = 0;
let nextCursor = null;

function mergePage(page) {
  if (!catalog) catalog = { environments: [], warnings: [] };
  for (const incomingEnv of page.environments) {
    let env = catalog.environments.find(item => item.id === incomingEnv.id);
    if (!env) { env = { id: incomingEnv.id, goals: [] }; catalog.environments.push(env); }
    for (const incomingGoal of incomingEnv.goals) {
      let goal = env.goals.find(item => item.id === incomingGoal.id);
      if (!goal) { goal = { ...incomingGoal, revisions: [] }; env.goals.push(goal); }
      for (const incomingRevision of incomingGoal.revisions) {
        let revision = goal.revisions.find(item => item.id === incomingRevision.id);
        if (!revision) { revision = { id: incomingRevision.id, variants: [], runs: [] }; goal.revisions.push(revision); }
        for (const incomingVariant of incomingRevision.variants) {
          let variant = revision.variants.find(item => item.id === incomingVariant.id);
          if (!variant) { variant = { ...incomingVariant, runs: [] }; revision.variants.push(variant); }
          for (const run of incomingVariant.runs) {
            const old = variant.runs.findIndex(item => item.id === run.id);
            if (old < 0) variant.runs.push(run);
            else if (run.checkpoints.length || !variant.runs[old].checkpoints.length) variant.runs[old] = run;
          }
          variant.runs.sort((a, b) => b.modified - a.modified);
          variant.run_count = variant.runs.length;
          variant.first_activity = Math.min(...variant.runs.map(item => item.created));
          variant.last_activity = Math.max(...variant.runs.map(item => item.modified));
        }
        revision.runs = revision.variants.flatMap(item => item.runs).sort((a, b) => b.modified - a.modified);
        revision.variants.sort((a, b) => a.id.localeCompare(b.id));
      }
      goal.revisions.sort((a, b) => a.id.localeCompare(b.id));
    }
    env.goals.sort((a, b) => a.id.localeCompare(b.id));
  }
  catalog.environments.sort((a, b) => a.id.localeCompare(b.id));
  catalog.warnings.push(...page.warnings);
  if (nextCursor) {
    for (const env of catalog.environments) for (const goal of env.goals) for (const revision of goal.revisions) for (const run of revision.runs) run.rank = null;
  }
  if (!nextCursor) {
    const groups = new Map();
    for (const env of catalog.environments) for (const goal of env.goals) for (const revision of goal.revisions) for (const run of revision.runs) {
      run.rank = null;
      if (run.status === 'complete' && run.comparability && run.best_mse != null && run.has_final_prediction) {
        if (!groups.has(run.comparability)) groups.set(run.comparability, []);
        groups.get(run.comparability).push(run);
      }
    }
    for (const runs of groups.values()) runs.sort((a, b) => a.best_mse - b.best_mse || a.id.localeCompare(b.id)).forEach((run, index) => { run.rank = index + 1; });
  }
}

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
function route(selection = {}) {
  const query = new URLSearchParams(selection);
  if (desktopId) query.set('desktop', desktopId);
  return `/browse${query.size ? `?${query}` : ''}#${new URLSearchParams({ token: token || '' })}`;
}
function link(label, href) {
  const a = document.createElement('a');
  a.textContent = label;
  a.href = href;
  a.onclick = event => {
    if (event.button || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    if (loading) return;
    history.pushState(null, '', href);
    $('#search').value = '';
    showError('');
    refresh(false).then(() => $('#catalog-title').focus());
  };
  return a;
}
function cell(value, className = '') {
  const td = document.createElement('td');
  td.className = className;
  if (value instanceof Node) td.append(value); else td.textContent = value;
  return td;
}
function date(value) { return value ? new Date(value * 1000).toLocaleString() : 'Missing'; }
function count(goal) { return goal.revisions.reduce((total, revision) => total + revision.runs.length, 0); }
async function openCheckpoint(item, button) {
  if (loading) return;
  loading = true;
  showError('');
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
    showError(error.message);
    render();
  }
}
function renderDetails(goal, revision, variant, run) {
  const details = $('#catalog-details');
  details.replaceChildren();
  if (goal && !revision) {
    const p = document.createElement('p');
    p.textContent = `Current revision: ${goal.current || 'Missing'}. Checked-in recipes: ${goal.recipes.length ? goal.recipes.join(', ') : 'none'}.`;
    details.append(p);
  }
  if (variant) {
    const p = document.createElement('p');
    p.textContent = `${variant.run_count} ${nextCursor ? 'loaded ' : ''}Runs. First activity: ${date(variant.first_activity)}. Last activity: ${date(variant.last_activity)}.`;
    details.append(p);
    if (variant.diff?.length) {
      const changes = document.createElement('ul');
      for (const change of variant.diff) {
        const item = document.createElement('li');
        item.textContent = `${change.path}: ${JSON.stringify(change.before)} → ${JSON.stringify(change.after)}`;
        changes.append(item);
      }
      details.append(changes);
    }
  }
  if (run) {
    const p = document.createElement('p');
    p.textContent = `State: ${run.status}. Held-out float32 RGB MSE: ${run.best_mse ?? 'Missing'}. Comparability: ${run.comparability || 'Missing'}.`;
    details.append(p);
    const overrides = document.createElement('p');
    overrides.textContent = `Launch overrides: ${run.launch_overrides?.length ? run.launch_overrides.join(', ') : 'none recorded'}.`;
    details.append(overrides);
    if (run.wandb_url?.startsWith('https://wandb.ai/')) {
      const tracked = document.createElement('a');
      tracked.href = run.wandb_url;
      tracked.textContent = 'Open W&B metric history';
      tracked.rel = 'noopener noreferrer';
      tracked.target = '_blank';
      details.append(tracked);
    }
    if (run.recovery?.length) {
      const recovery = document.createElement('p');
      recovery.textContent = `Recovery files: ${run.recovery.join(', ')}`;
      details.append(recovery);
    }
    if (run.stage_history?.length) {
      const heading = document.createElement('h2');
      heading.textContent = 'Stage losses';
      details.append(heading);
      const table = document.createElement('table');
      const head = document.createElement('tr');
      for (const label of ['Stage', 'Epoch', 'Train loss', 'Held-out loss', 'Final RGB MSE']) {
        const th = document.createElement('th'); th.textContent = label; head.append(th);
      }
      table.append(head);
      const finalStage = run.stages?.at(-1) || run.stage_history.at(-1).stage;
      for (const record of run.stage_history) {
        const row = document.createElement('tr');
        for (const value of [record.stage, record.epoch, record.train?.loss ?? 'Missing', record.validation?.loss ?? 'Missing', record.stage === finalStage ? record.validation?.mse ?? 'Missing' : '—']) {
          row.append(cell(value));
        }
        table.append(row);
      }
      details.append(table);
    }
    if (run.probe_manifest) {
      const probes = document.createElement('p');
      probes.textContent = `${run.probe_manifest.starts?.length || 0} fixed rollout probe starts, horizon ${run.probe_manifest.horizon}. Probe metrics are diagnostics, not evidence of playable long rollouts.`;
      details.append(probes);
    }
    for (const [label, value] of [['Goal contract', run.goal_yaml], ['Resolved Run configuration', run.config_yaml]]) {
      if (!value) continue;
      const block = document.createElement('details');
      const heading = document.createElement('summary');
      heading.textContent = label;
      const pre = document.createElement('pre');
      pre.textContent = value;
      block.append(heading, pre);
      details.append(block);
    }
  }
}
function render() {
  if (!catalog) return;
  const params = new URLSearchParams(location.search);
  const env = catalog.environments.find(item => item.id === params.get('env'));
  const goal = env?.goals.find(item => item.id === params.get('goal'));
  const revision = goal?.revisions.find(item => item.id === params.get('revision'));
  const variant = revision?.variants.find(item => item.id === params.get('variant'));
  const run = variant?.runs.find(item => item.id === params.get('run'));
  const selected = {};
  const levels = [['Environments', route()]];
  for (const [key, item] of [['env', env], ['goal', goal], ['revision', revision], ['variant', variant], ['run', run]]) {
    if (!item) break;
    selected[key] = item.id;
    levels.push([key === 'run' ? item.name : item.id, route(selected)]);
  }
  if ([['env', env], ['goal', goal], ['revision', revision], ['variant', variant], ['run', run]]
      .some(([key, item]) => params.has(key) && !item)) {
    history.replaceState(null, '', route(selected));
    showError('This selection is no longer available. Choose another item or refresh.');
  }
  const crumbs = $('#breadcrumbs');
  crumbs.replaceChildren();
  for (const [index, [label, href]] of levels.entries()) {
    if (index) {
      const separator = document.createElement('span');
      separator.textContent = '/';
      separator.setAttribute('aria-hidden', 'true');
      crumbs.append(separator);
    }
    const a = link(label, href);
    if (index === levels.length - 1) a.setAttribute('aria-current', 'page');
    crumbs.append(a);
  }
  const title = run ? 'Checkpoints' : variant ? 'Runs' : revision ? 'Goal Variants' : goal ? 'Goal Revisions' : env ? 'Research Goals' : 'Environments';
  $('#catalog-title').textContent = title;
  document.title = `${title} · Gymemu`;
  $('#catalog-description').textContent = run ? `${run.name} · ${run.approach || 'Unknown Approach'}` : goal ? goal.id : env ? env.id : 'Choose an Environment.';
  $('#search').placeholder = `Search ${title.toLowerCase()}`;
  $('#search').setAttribute('aria-label', $('#search').placeholder);
  const columns = run ? ['Checkpoint', 'Stage', 'Size', 'Saved'] : variant ? ['Run', 'Approach', 'State', 'Held-out RGB MSE', 'Comparable rank'] : revision ? ['Goal Variant', 'Runs', 'First activity', 'Last activity'] : goal ? ['Goal Revision', 'Runs', 'Current'] : env ? ['Research Goal', 'Runs', 'Recipes'] : ['Environment', 'Research Goals', 'Runs'];
  const heading = document.createElement('tr');
  for (const label of columns) {
    const th = document.createElement('th'); th.scope = 'col'; th.textContent = label; heading.append(th);
  }
  $('#catalog-columns').replaceChildren(heading);
  const all = run ? run.checkpoints : variant ? variant.runs : revision ? revision.variants : goal ? goal.revisions : env ? env.goals : catalog.environments;
  const query = $('#search').value.trim().toLowerCase();
  const items = all.filter(item => `${item.name || item.id} ${item.approach || ''}`.toLowerCase().includes(query));
  const body = $('#catalog-items'); body.replaceChildren();
  for (const item of items) {
    const row = document.createElement('tr');
    if (run) {
      const button = document.createElement('button'); button.type = 'button'; button.textContent = item.name;
      button.onclick = () => openCheckpoint(item, button);
      row.append(cell(button, 'catalog-name'), cell(item.stage || 'Final prediction'), cell(`${(item.size_bytes / 1024 / 1024).toFixed(2)} MB`), cell(date(item.modified), 'catalog-date'));
    } else if (variant) {
      row.append(cell(link(item.name, route({ ...selected, run: item.id })), 'catalog-name'), cell(item.approach || 'Missing'), cell(item.status), cell(item.best_mse ?? 'Missing'), cell(item.rank ?? 'Unranked'));
    } else if (revision) {
      row.append(cell(link(item.id, route({ ...selected, variant: item.id })), 'catalog-name'), cell(item.run_count), cell(date(item.first_activity)), cell(date(item.last_activity)));
    } else if (goal) {
      row.append(cell(link(item.id, route({ ...selected, revision: item.id })), 'catalog-name'), cell(item.runs.length), cell(item.id === goal.current ? 'Current' : 'Historical'));
    } else if (env) {
      row.append(cell(link(item.id, route({ ...selected, goal: item.id })), 'catalog-name'), cell(count(item)), cell(item.recipes.length));
    } else {
      row.append(cell(link(item.id, route({ env: item.id })), 'catalog-name'), cell(item.goals.length), cell(item.goals.reduce((sum, g) => sum + count(g), 0)));
    }
    body.append(row);
  }
  renderDetails(goal, revision, variant, run);
  $('#item-count').textContent = `${items.length} of ${all.length}${nextCursor ? ' loaded' : ''}`;
  $('#load-more').hidden = !nextCursor;
  $('#catalog-status').hidden = true;
  $('.catalog-table').hidden = !items.length;
  $('#catalog-empty').hidden = Boolean(items.length);
  $('#catalog-empty').textContent = query ? 'No items match this search.' : 'No published items here yet.';
  $('#catalog-root').textContent = 'Private R2 research catalog';
  $('#catalog-warnings').hidden = !catalog.warnings.length;
  const warnings = $('#catalog-warnings ul'); warnings.replaceChildren();
  for (const warning of catalog.warnings) {
    const li = document.createElement('li'); li.textContent = warning; warnings.append(li);
  }
}
function setRefreshing(refreshing) {
  const button = $('#refresh');
  button.disabled = refreshing || loading;
  button.classList.toggle('refreshing', refreshing);
  button.setAttribute('aria-label', refreshing ? 'Refreshing' : 'Refresh');
  button.setAttribute('aria-busy', String(refreshing));
  button.title = refreshing ? 'Refreshing this list' : 'Refresh this list';
}
async function refresh(reload = true) {
  const currentRequest = ++requestId;
  if (loading) return;
  const selectedRun = new URLSearchParams(location.search).get('run');
  if (!reload && catalog && !selectedRun) { render(); return; }
  setRefreshing(true);
  $('#catalog-status').hidden = false;
  $('#catalog-status').textContent = 'Loading catalog…';
  showError('');
  try {
    const params = new URLSearchParams(location.search);
    const query = new URLSearchParams();
    if (params.has('run')) query.set('run', params.get('run'));
    if (reload) query.set('refresh', '1');
    const result = await request(`/api/catalog?${query}`);
    if (currentRequest !== requestId) return;
    const firstPage = !catalog || reload;
    if (reload) catalog = null;
    if (firstPage) nextCursor = result.next_cursor;
    mergePage(result);
    render();
  } catch (error) {
    if (currentRequest === requestId) {
      showError(error.message);
      $('#catalog-status').hidden = true;
    }
  } finally {
    if (currentRequest === requestId) setRefreshing(false);
  }
}
async function loadMore() {
  if (!nextCursor || loading) return;
  const cursor = nextCursor;
  $('#load-more').disabled = true;
  try {
    const result = await request(`/api/catalog?${new URLSearchParams({ cursor })}`);
    nextCursor = result.next_cursor;
    mergePage(result);
    render();
  } catch (error) { showError(error.message); }
  finally { $('#load-more').disabled = false; }
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
$('#load-more').onclick = loadMore;
window.addEventListener('popstate', () => { $('#search').value = ''; refresh(false); });
if (!token) {
  $('#catalog-status').hidden = true;
  showError('Open the complete navigator URL printed in your terminal.');
} else refresh();
