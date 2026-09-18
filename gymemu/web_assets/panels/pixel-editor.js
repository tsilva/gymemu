// The editor works in native pixel coordinates; zoom only changes display size.
export function framePalette(data) {
  const colors = new Map();
  for (let i = 0; i < data.length; i += 4) {
    const rgb = Array.from(data.slice(i, i + 3));
    colors.set(rgb.join(','), rgb);
  }
  return [...colors.values()].sort((a, b) => a.reduce((n,v) => n+v,0) - b.reduce((n,v) => n+v,0));
}

export function changedPixels(before, after) {
  const changes = [];
  for (let i = 0; i < before.length; i += 4) {
    if (before[i] !== after[i] || before[i+1] !== after[i+1] || before[i+2] !== after[i+2])
      changes.push([i/4, after[i], after[i+1], after[i+2]]);
  }
  return changes;
}

export function paintLine(data, width, from, to, color, size = 1) {
  const dx = to[0] - from[0], dy = to[1] - from[1];
  const steps = Math.max(Math.abs(dx), Math.abs(dy));
  const height = data.length / (width * 4), offset = Math.floor(size / 2);
  const rgba = [...color, 255];
  for (let i = 0; i <= steps; i++) {
    const x = Math.round(from[0] + dx * i / (steps || 1));
    const y = Math.round(from[1] + dy * i / (steps || 1));
    for (let row = Math.max(0,y-offset); row < Math.min(height,y-offset+size); row++) {
      for (let col = Math.max(0,x-offset); col < Math.min(width,x-offset+size); col++) {
        data.set(rgba, (row * width + col) * 4);
      }
    }
  }
}

const hex = rgb => '#' + rgb.map(v => v.toString(16).padStart(2,'0')).join('').toUpperCase();

export function editUnavailableReason(snapshot) {
  if (snapshot.mode === 'reconstruction') return 'Reconstruction uses the current recorded frame. Pixel edits are unavailable in this mode.';
  if (!snapshot.history_editable) return 'Restart the player to apply pixel edits. This server does not support them yet.';
  if (!snapshot.has_prediction) return 'Run one prediction before applying pixel edits. Close this popup and step the player once.';
  if (snapshot.loading) return 'Wait for the player to finish loading before applying pixel edits.';
  return null;
}

export function createPixelEditor({ apply }) {
  const dialog = document.createElement('dialog');
  dialog.className = 'pixel-editor';
  dialog.setAttribute('aria-label','Edit history frame');
  dialog.innerHTML = `
    <header class="pixel-editor-header"><h2>Edit history frame</h2><button type="button" class="icon-button" data-close aria-label="Close pixel editor">×</button></header>
    <p class="pixel-editor-help">Choose a brush size and palette color, or Alt-click the image to pick a color.</p>
    <p class="pixel-editor-unavailable" role="note" hidden></p>
    <div class="pixel-editor-layout">
      <div class="pixel-editor-viewport"><canvas tabindex="0" aria-label="Editable history frame"></canvas></div>
      <aside class="pixel-editor-tools">
        <label>Zoom <input type="range" min="2" max="24" step="1" value="4" aria-label="Pixel zoom"><output data-zoom></output></label>
        <label>Brush size <input type="range" min="1" max="32" step="1" value="1" aria-label="Brush size"><output data-brush-size>1 × 1 px</output></label>
        <div class="pixel-editor-selection"><span data-color-preview></span><strong data-color></strong></div>
        <div class="pixel-editor-palette" role="group" aria-label="Frame palette"></div>
        <div class="pixel-editor-pages"><button type="button" data-prev aria-label="Previous palette page">‹</button><span data-page></span><button type="button" data-next aria-label="Next palette page">›</button></div>
        <p data-palette-note></p>
        <button type="button" data-undo>Undo stroke</button><button type="button" data-reset>Reset painting</button>
      </aside>
    </div>
    <div class="pixel-editor-footer"><span role="status"></span><div class="dialog-actions"><button type="button" data-cancel>Cancel</button><button type="button" data-apply>Apply and predict</button></div></div>`;
  document.body.append(dialog);
  const $ = selector => dialog.querySelector(selector);
  const canvas = $('canvas'), context = canvas.getContext('2d');
  const paletteElement = $('.pixel-editor-palette'), zoom = $('[aria-label="Pixel zoom"]'), status = $('[role=status]');
  const brush = $('[aria-label="Brush size"]');
  let state = null, selected = null, page = 0, stroke = null, busy = false;
  function draw() { if (state) context.putImageData(state.image,0,0); }
  function controls() {
    if (!state) return;
    const count = changedPixels(state.original,state.image.data).length;
    status.textContent = busy ? 'Running inference…' : `${count} pixel${count===1?'':'s'} changed`;
    $('[data-apply]').disabled = busy || !count || Boolean(state.unavailable);
    $('[data-undo]').disabled = busy || !state.undo.length;
    $('[data-reset]').disabled = busy || !count;
    canvas.setAttribute('aria-disabled',String(busy));
  }
  function selectColor(color) {
    selected = color;
    $('[data-color]').textContent = hex(color);
    $('[data-color-preview]').style.background = hex(color);
    paletteElement.querySelectorAll('button').forEach(button => button.setAttribute('aria-pressed',String(button.dataset.color===hex(color))));
  }
  function palettePage() {
    paletteElement.replaceChildren();
    state.palette.slice(page*96,(page+1)*96).forEach(color => {
      const button = document.createElement('button');
      button.type = 'button'; button.className = 'pixel-editor-swatch';
      button.dataset.color = hex(color); button.style.background = hex(color);
      button.title = hex(color); button.setAttribute('aria-label',`Paint ${hex(color)}`);
      button.onclick = () => selectColor(color);
      paletteElement.append(button);
    });
    selectColor(selected);
    $('.pixel-editor-pages').hidden = state.palette.length <= 96;
    $('[data-page]').textContent = `${page+1} / ${Math.ceil(state.palette.length/96)}`;
    $('[data-prev]').disabled = page === 0;
    $('[data-next]').disabled = (page+1)*96 >= state.palette.length;
  }
  function setZoom() {
    canvas.style.width = `${canvas.width*Number(zoom.value)}px`;
    canvas.style.height = `${canvas.height*Number(zoom.value)}px`;
    $('[data-zoom]').textContent = `${zoom.value}×`;
  }
  function point(event) {
    const bounds = canvas.getBoundingClientRect();
    const x = Math.floor((event.clientX-bounds.left)*canvas.width/bounds.width);
    const y = Math.floor((event.clientY-bounds.top)*canvas.height/bounds.height);
    return x >= 0 && y >= 0 && x < canvas.width && y < canvas.height ? [x,y] : null;
  }
  function finishStroke() {
    if (!stroke || !state) return;
    if (changedPixels(stroke.before,state.image.data).length) {
      state.undo.push(stroke.before);
      if (state.undo.length > 20) state.undo.shift();
    }
    stroke = null; controls();
  }
  canvas.addEventListener('pointerdown',event => {
    if (event.button !== 0 || busy || !state) return;
    const position = point(event); if (!position) return;
    event.preventDefault(); canvas.focus();
    if (event.altKey) {
      const offset = (position[1]*canvas.width+position[0])*4;
      selectColor(Array.from(state.image.data.slice(offset,offset+3)));
      return;
    }
    stroke = {before:state.image.data.slice(),last:position,size:Number(brush.value)};
    canvas.setPointerCapture(event.pointerId);
    paintLine(state.image.data,canvas.width,position,position,selected,stroke.size); draw();
  });
  canvas.addEventListener('pointermove',event => {
    if (!stroke || busy || !state) return;
    const position = point(event);
    if (position) paintLine(state.image.data,canvas.width,stroke.last || position,position,selected,stroke.size);
    stroke.last = position; draw();
  });
  canvas.addEventListener('pointerup',finishStroke);
  canvas.addEventListener('pointercancel',finishStroke);
  canvas.addEventListener('lostpointercapture',finishStroke);
  zoom.oninput = setZoom;
  brush.oninput = () => {
    $('[data-brush-size]').textContent = `${brush.value} × ${brush.value} px`;
    brush.setAttribute('aria-valuetext',`${brush.value} by ${brush.value} image pixels`);
  };
  $('[data-prev]').onclick = () => {page--; palettePage();};
  $('[data-next]').onclick = () => {page++; palettePage();};
  $('[data-undo]').onclick = () => {state.image.data.set(state.undo.pop()); draw(); controls();};
  $('[data-reset]').onclick = () => {
    state.undo.push(state.image.data.slice());
    if (state.undo.length > 20) state.undo.shift();
    state.image.data.set(state.original); draw(); controls();
  };
  for (const name of ['close','cancel']) $(`[data-${name}]`).onclick = () => dialog.close();
  dialog.addEventListener('close',() => {state=null; stroke=null;});
  $('[data-apply]').onclick = async () => {
    if (busy || !state || state.unavailable) return;
    finishStroke();
    const editing = state;
    let failure = null;
    busy = true; controls();
    try {
      const result = await apply({type:'edit_history',frame:editing.frame,history_revision:editing.revision,pixels:changedPixels(editing.original,editing.image.data)});
      if (state !== editing) return;
      if (result?.error) failure = result.error;
      else dialog.close();
    } catch (error) {
      failure = error.message;
    } finally {
      busy = false;
      if (state === editing) {
        controls();
        if (failure) status.textContent = failure;
      }
    }
  };
  return {
    open(source, frame, slot, revision, unavailable = null) {
      if (busy) return;
      const image = source.getContext('2d').getImageData(0,0,source.width,source.height);
      state = {frame,revision,image,original:image.data.slice(),palette:framePalette(image.data),undo:[],unavailable};
      $('.pixel-editor-unavailable').textContent = unavailable || '';
      $('.pixel-editor-unavailable').hidden = !unavailable;
      selected = state.palette[0]; page = 0; stroke = null;
      canvas.width = image.width; canvas.height = image.height;
      $('h2').textContent = `Edit history frame ${slot+1}`;
      $('[data-palette-note]').textContent = `${state.palette.length} colors from this frame. Painting uses only these colors.`;
      zoom.value = Math.max(2,Math.min(8,Math.floor(Math.min((innerWidth-340)/canvas.width,(innerHeight-240)/canvas.height))));
      palettePage(); setZoom(); draw(); controls(); dialog.showModal();
      $('.pixel-editor-viewport').scrollTo(0,0);
    },
    sync(snapshot) {
      if (state && (snapshot.history_revision !== state.revision || snapshot.loading || snapshot.playing)) dialog.close();
    },
    destroy() { dialog.close(); dialog.remove(); },
  };
}
