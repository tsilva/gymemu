import { createPanel } from './shared.js';
export function mount({ definition, services }) {
  const source = definition.config.source;
  const element = createPanel({ id: definition.id, label: definition.title, body: `
    <div class="frame-viewport"><canvas aria-label="${source} frame"></canvas><p class="frame-empty"></p></div>
    <footer class="frame-foot frame-footer"${source === 'difference' ? '' : ' aria-hidden="true"'}>${source === 'difference' ? '<span class="frame-mse" title="RGB mean squared error, independent of display gain">MSE —</span><span class="difference-legend" title="Gray = 0 · brighter + · darker −">Gray = 0 · brighter + · darker −</span>' : ''}</footer>` });
  const canvas = element.querySelector('canvas'), empty = element.querySelector('.frame-empty');
  if (source === 'difference') {
    const label = document.createElement('label'); label.className = 'gain-control'; label.textContent = 'Gain ';
    const select = document.createElement('select'); select.setAttribute('aria-label', 'Difference gain');
    for (const gain of [1,2,4,8]) { const option = new Option(`${gain}×`, gain); option.selected = gain === (definition.config.gain || 1); select.add(option); }
    select.onchange = () => services.updatePanel(definition.id, { config: { ...definition.config, gain: Number(select.value) } });
    label.append(select); element.querySelector('.frame-foot').prepend(label);
  }
  return { element, render(snapshot, view) {
    if (!snapshot) return;
    if (source === 'difference') element.querySelector('.frame-mse').textContent = `MSE ${snapshot.mse == null ? '—' : snapshot.mse.toFixed(7)}`;
    const bitmap = view.bitmaps?.[source];
    const pending = Boolean(bitmap) && source !== 'original' && !snapshot.has_prediction;
    empty.hidden = Boolean(bitmap) && !pending;
    empty.textContent = pending ? 'Step to generate a prediction' : 'Available in teacher-forced dataset replay';
    canvas.hidden = !bitmap || pending;
    if (bitmap) {
      canvas.width = bitmap.width; canvas.height = bitmap.height;
      const context = canvas.getContext('2d'); context.drawImage(bitmap, 0, 0);
      if (source === 'difference' && definition.config.gain > 1) {
        const pixels = context.getImageData(0,0,canvas.width,canvas.height);
        for (let i=0; i<pixels.data.length; i+=4) for (let c=0; c<3; c++) pixels.data[i+c] = 128 + (pixels.data[i+c]-128)*definition.config.gain;
        context.putImageData(pixels,0,0);
      }
    }
    // Autoregressive starts show the recorded scene before the first prediction.
    if (source === 'prediction' && snapshot.mode === 'autoregressive' && bitmap) { canvas.hidden = false; empty.hidden = true; }
  }};
}
