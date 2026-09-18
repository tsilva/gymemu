import { createPanel } from './shared.js';
import { createPixelEditor, editUnavailableReason } from './pixel-editor.js';

export function moveHistoryFrame(order, from, to) {
  const next = [...order];
  next.splice(to, 0, ...next.splice(from, 1));
  return next;
}

export function mount({ definition, services }) {
  const element = createPanel({ id: definition.id, label: definition.title, body: '<div class="history-grid"></div><div class="history-note" role="status"></div>' });
  const grid = element.querySelector('.history-grid');
  const note = element.querySelector('.history-note');
  const editor = createPixelEditor({apply:command => services.command(command)});
  let count = 0, current = null, drag = null, pending = false, openAfterPause = null;
  const supported = () => Array.isArray(current?.history_order) && Number.isInteger(current?.history_revision);
  const enabled = () => supported() && current.has_prediction && !current.loading && !current.playing && !pending && count > 1;
  function openEditor(index) {
    const canvas = grid.children[index]?.querySelector('canvas');
    if (!canvas || !current) return;
    try {
      editor.open(canvas,current.history_order?.[index] ?? index,index,current.history_revision,editUnavailableReason(current));
    } catch (error) {
      services.showToast?.(`Could not open the pixel editor: ${error.message}`);
    }
  }
  const indexAt = target => {
    const figure = target?.closest('.history-grid figure');
    return figure && grid.contains(figure) ? Number(figure.dataset.index) : null;
  };
  function clearDrag() {
    drag = null;
    grid.querySelectorAll('figure').forEach(figure => figure.classList.remove('history-dragging', 'history-drop-target'));
  }
  function updateControls() {
    note.textContent = current.mode === 'reconstruction' ? 'Current recorded frame. No temporal history is used.' : !supported() ? 'Restart the player to enable history reordering.' : pending ? 'Predicting reordered history…' : !current.has_prediction ? 'Predict a frame to edit its history.' : current.playing ? 'Pause to edit or reorder history.' : current.history_reordered || current.history_edited_frames?.length ? 'Modified RGB history · Scrubbing or stepping restores the original.' : '';
    note.hidden = !note.textContent;
    grid.querySelectorAll('figure').forEach(figure => {
      figure.tabIndex = enabled() ? 0 : -1;
      figure.dataset.reorderEnabled = String(enabled());
      figure.querySelector('.history-edit').disabled = !current.history_editable || pending || Boolean(current.loading);
      figure.querySelector('.history-revert').disabled = pending || Boolean(current.loading) || current.playing;
    });
  }
  async function move(from, to, revision) {
    if (from === to || !enabled()) return;
    pending = true;
    updateControls();
    try {
      await services.command({type:'reorder_history', order:moveHistoryFrame(current.history_order,from,to), history_revision:revision});
    } finally {
      pending = false;
      updateControls();
    }
  }
  grid.addEventListener('pointerdown', event => {
    if (event.target.closest('button')) return;
    const index = indexAt(event.target);
    if (event.button !== 0 || index === null || !enabled()) return;
    event.preventDefault(); event.stopPropagation();
    grid.children[index].focus();
    drag = {from:index,to:index,x:event.clientX,y:event.clientY,moved:false,revision:current.history_revision};
    grid.setPointerCapture(event.pointerId);
  });
  grid.addEventListener('pointermove', event => {
    if (!drag) return;
    if (Math.hypot(event.clientX-drag.x,event.clientY-drag.y) < 5 && !drag.moved) return;
    drag.moved = true;
    drag.to = indexAt(document.elementFromPoint(event.clientX,event.clientY));
    grid.querySelectorAll('figure').forEach((figure,i) => {
      figure.classList.toggle('history-dragging',i===drag.from);
      figure.classList.toggle('history-drop-target',i===drag.to && i!==drag.from);
    });
  });
  grid.addEventListener('pointerup', event => {
    if (!drag) return;
    const {from,to,moved,revision} = drag;
    clearDrag();
    if (grid.hasPointerCapture(event.pointerId)) grid.releasePointerCapture(event.pointerId);
    if (moved && to !== null) void move(from,to,revision);
  });
  grid.addEventListener('pointercancel',clearDrag);
  grid.addEventListener('lostpointercapture',clearDrag);
  grid.addEventListener('keydown',event => {
    const from = indexAt(event.target);
    if (!event.altKey || from === null || !['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)) return;
    event.preventDefault(); event.stopPropagation();
    const to = Math.max(0,Math.min(count-1,from+(['ArrowLeft','ArrowUp'].includes(event.key)?-1:1)));
    void move(from,to,current.history_revision);
  });
  return { element, destroy:() => editor.destroy(), render(snapshot, view) {
    if (!snapshot) return;
    if (current?.history_revision !== snapshot.history_revision) clearDrag();
    current = snapshot;
    editor.sync(snapshot);
    if (count !== snapshot.history_length) {
      count = snapshot.history_length; grid.replaceChildren();
      grid.style.setProperty('--history-columns', Math.min(4, count));
      grid.style.setProperty('--history-mobile-columns', Math.min(2, count));
      for (let i=0; i<count; i++) {
        const figure = document.createElement('figure'), canvas = document.createElement('canvas'), label = document.createElement('figcaption');
        figure.dataset.index = String(i);
        canvas.setAttribute('aria-label', `History frame ${i+1} of ${count}${i===0 ? ', oldest' : i===count-1 ? ', newest' : ''}`);
        const edit = document.createElement('button');
        edit.type = 'button'; edit.className = 'history-edit icon-button icon-only';
        edit.setAttribute('aria-label',`Edit history frame ${i+1}`);
        edit.title = `Edit history frame ${i+1}`;
        edit.innerHTML = '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m16 3 5 5-12 12-6 1 1-6Z M14 5l5 5"/></svg>';
        for (const name of ['pointerdown','mousedown']) edit.addEventListener(name,event=>event.stopPropagation());
        edit.onclick = async event => {
          event.stopPropagation();
          if (current.playing) {
            openAfterPause = i;
            const result = await services.command({type:'pause'});
            if (result?.error) openAfterPause = null;
          } else openEditor(i);
        };
        const revert = document.createElement('button');
        revert.type = 'button'; revert.className = 'history-revert icon-button icon-only';
        revert.setAttribute('aria-label',`Revert history frame ${i+1}`);
        revert.title = `Revert history frame ${i+1}`;
        revert.hidden = true;
        revert.innerHTML = '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 10h11a7 7 0 0 1 0 14" transform="translate(0 -4)"/><path d="m7 2-4 4 4 4"/></svg>';
        for (const name of ['pointerdown','mousedown']) revert.addEventListener(name,event=>event.stopPropagation());
        revert.onclick = async event => {
          event.stopPropagation();
          if (pending || current.loading || current.playing) return;
          pending = true; updateControls();
          try {
            await services.command({type:'revert_history',frame:current.history_order[i],history_revision:current.history_revision});
          } finally { pending = false; updateControls(); }
        };
        figure.append(canvas,label,edit,revert); grid.append(figure);
      }
    }
    grid.querySelectorAll('figure').forEach((figure,i) => {
      const source = (snapshot.history_order?.[i] ?? i) + 1;
      figure.querySelector('figcaption').textContent = snapshot.history_reordered ? `${i+1}/${count} · frame ${source}` : `${i+1}/${count}`;
      const edited = Boolean(snapshot.history_edited_frames?.includes(source-1));
      if (edited) figure.querySelector('figcaption').textContent += ' · edited';
      figure.querySelector('.history-revert').hidden = !edited;
      figure.setAttribute('aria-label',`History slot ${i+1} of ${count}, original frame ${source}`);
    });
    updateControls();
    const bitmap = view.bitmaps?.history;
    if (bitmap) grid.querySelectorAll('canvas').forEach((canvas, i) => {
      const width = bitmap.width / count; canvas.width = width; canvas.height = bitmap.height;
      canvas.getContext('2d').drawImage(bitmap, i*width,0,width,bitmap.height,0,0,width,bitmap.height);
    });
    if (openAfterPause !== null && !snapshot.playing && !snapshot.loading) {
      const index = openAfterPause; openAfterPause = null;
      openEditor(index);
    }
  }};
}
