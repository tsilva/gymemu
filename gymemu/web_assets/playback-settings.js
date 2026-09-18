export function mountPlaybackControls({ services }) {
  const element = document.createElement('div');
  element.className = 'controls-body';
  element.innerHTML = `
    <div class="play-actions"><button data-play class="primary">Play</button><button data-step>Step</button><button data-reset>Reset</button><button data-next>Next</button></div>
    <label class="select-label"><span data-selection-label>Starting scene</span><select data-selection aria-label="Episode or starting scene"></select></label>
    <div class="action-picker"><span class="muted">Step with action</span><div class="action-buttons"></div></div>
    <p class="control-help"></p><div class="context-note"></div>`;
  const send = type => services.command({ type });
  element.querySelector('[data-play]').onclick = () => send(services.getState().playing ? 'pause' : 'play');
  element.querySelector('[data-step]').onclick = () => send('step');
  element.querySelector('[data-reset]').onclick = () => send('reset');
  element.querySelector('[data-next]').onclick = () => send('next');
  const select = element.querySelector('[data-selection]');
  select.onchange = () => services.command({type:'select',value:Number(select.value)});
  let signature = '';
  return { element, render(s) {
    if (!s) return;
    const replay = s.mode !== 'autoregressive';
    const reconstruction = s.mode === 'reconstruction';
    element.querySelector('[data-play]').textContent = s.playing ? 'Pause' : 'Play';
    element.querySelector('[data-play]').disabled = s.finished || Boolean(s.loading);
    element.querySelector('[data-step]').disabled = s.finished || Boolean(s.loading);
    element.querySelector('[data-selection-label]').textContent = replay ? 'Dataset episode' : 'Starting scene';
    const nextSignature = JSON.stringify([s.choices, s.keymap]);
    if (signature !== nextSignature) {
      signature = nextSignature;
      select.replaceChildren(...s.choices.map(item => new Option(item.label,item.value)));
      const target = element.querySelector('.action-buttons'); target.replaceChildren();
      s.action_values.forEach(action => {
        const button = document.createElement('button');
        const keys = Object.entries(s.keymap).filter(([,v]) => v === action).map(([k]) => k);
        button.textContent = keys.length ? `${keys.join('/')} · ${action}` : String(action);
        button.onclick = () => services.command({type:'step',action}); target.append(button);
      });
    }
    select.value = s.selection;
    element.querySelector('.action-picker').hidden = replay;
    element.querySelector('.control-help').textContent = reconstruction ? 'Space steps recorded frames. Tab plays or pauses. R resets; C selects the next episode.' : replay ? 'Space steps the recorded action. Tab plays or pauses. R resets; C selects the next episode.' : 'Action keys step once. Hold keys during play; releasing uses the default action. Tab plays or pauses. R resets; C selects the next scene.';
    element.querySelector('.context-note').textContent = reconstruction ? 'Each recorded frame is encoded and decoded independently. Original and reconstruction show the same timestep.' : replay ? 'Recorded RGB and action history on every step. Predictions never feed back.' : 'Generated frames feed the next prediction. No recorded-frame corrections.';
  }};
}
