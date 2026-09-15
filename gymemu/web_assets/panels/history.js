import { createPanel } from './shared.js';
export function mount({ definition }) {
  const element = createPanel({ id: definition.id, label: definition.title, body: '<div class="history-grid"></div><footer class="frame-foot history-caption"></footer>' });
  const grid = element.querySelector('.history-grid');
  let count = 0;
  return { element, render(snapshot, view) {
    if (!snapshot) return;
    if (count !== snapshot.history_length) {
      count = snapshot.history_length; grid.replaceChildren();
      for (let i=0; i<count; i++) {
        const figure = document.createElement('figure'), canvas = document.createElement('canvas'), label = document.createElement('figcaption');
        canvas.setAttribute('aria-label', `History frame ${i+1}`);
        label.textContent = `${i+1}/${count}${i===0 ? ' · oldest' : i===count-1 ? ' · newest' : ''}`;
        figure.append(canvas,label); grid.append(figure);
      }
    }
    const bitmap = view.bitmaps?.history;
    if (bitmap) grid.querySelectorAll('canvas').forEach((canvas, i) => {
      const width = bitmap.width / count; canvas.width = width; canvas.height = bitmap.height;
      canvas.getContext('2d').drawImage(bitmap, i*width,0,width,bitmap.height,0,0,width,bitmap.height);
    });
    element.querySelector('.history-caption').textContent = `${snapshot.has_prediction ? 'Inputs used for this prediction' : 'Inputs ready for the next step'} · oldest to newest · zero padding retained`;
  }};
}
