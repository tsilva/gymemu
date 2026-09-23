<header class="app-header">
    <a class="app-wordmark" href="/" aria-label="Gymemu player">GYMEMU</a>
    <span class="header-divider" aria-hidden="true"></span>
    <div class="session-heading"><strong id="checkpoint">Connecting to checkpoint…</strong></div>
    <span class="hotreload-warning" role="status" title="Live UI changes reload this player automatically">Hot reload active</span>
    <a id="browse-checkpoints" class="companion-tab" href="/browse" hidden>Checkpoints</a>
    <span id="inspection-position" class="inspection-position" hidden></span>
    <select id="mode" class="mode-badge" aria-label="Playback mode"><option value="autoregressive">Autoregressive</option><option value="teacher-forcing">Teacher forcing</option></select>
    <a id="companion-tab" class="companion-tab" target="gymemu-stats" href="/workspace/stats" title="Open diagnostics tab" aria-label="Open diagnostics tab"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-external-link"></use></svg><span>Diagnostics</span></a>
    <div class="workspace-actions">
      <button id="panels-toggle" class="quiet" type="button" aria-label="Panels" title="Panels"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-layout-grid"></use></svg></button>
      <button id="add-panel" type="button" aria-label="Add widget" title="Add widget"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-plus"></use></svg></button>
      <button id="reset-layout" class="quiet" type="button" aria-label="Reset layout" title="Reset layout"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-restore"></use></svg></button>
    </div>
  </header>
  <aside id="panel-shelf" class="panel-shelf" hidden><span class="eyebrow">HIDDEN PANELS</span><div id="panel-shelf-items"></div></aside>
  <main id="dashboard" class="dashboard grid-stack" aria-label="Playback workspace"></main>
  <footer class="timeline" aria-label="Playback timeline">
    <div class="timeline-labels"><strong id="position">Waiting for playback</strong><button id="timeline-zoom" type="button" aria-label="Reset chart zoom" hidden></button></div>
    <div class="timeline-track">
      <div class="timeline-scrubber-track"><input id="timeline-scrubber" type="range" min="0" max="0" value="0" step="1" aria-label="Recorded frame position" disabled><div id="timeline-zoom-band" hidden><button class="timeline-range-handle" data-edge="first" role="slider" aria-label="Chart range start" aria-orientation="horizontal" aria-valuemin="0" aria-valuemax="0" aria-valuenow="0" title="Drag to adjust chart window start" type="button"></button><button class="timeline-range-handle" data-edge="last" role="slider" aria-label="Chart range end" aria-orientation="horizontal" aria-valuemin="0" aria-valuemax="0" aria-valuenow="0" title="Drag to adjust chart window end" type="button"></button></div></div>
      <div class="timeline-actions">
        <button id="play-toggle" class="primary" type="button" data-action="play" aria-label="Play" title="Play" disabled><svg class="icon" aria-hidden="true"><use id="play-toggle-icon" href="/assets/tabler-icons.svg#ti-player-play"></use></svg></button>
        <button id="reset-playback" class="quiet" type="button" aria-label="Reset playback" title="Reset playback" disabled><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-refresh"></use></svg></button>
        <button id="playback-settings-toggle" class="quiet" type="button" aria-label="Playback settings" title="Playback settings" aria-haspopup="dialog" aria-controls="playback-settings" aria-expanded="false" disabled><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-settings"></use></svg></button>
      </div>
    </div>
  </footer>
  <dialog id="playback-settings" aria-labelledby="playback-settings-heading">
    <div class="playback-settings-heading"><h2 id="playback-settings-heading">Playback settings</h2><button id="playback-settings-close" class="icon-button" type="button" aria-label="Close playback settings" title="Close"><svg class="icon" aria-hidden="true"><use href="/assets/tabler-icons.svg#ti-x"></use></svg></button></div>
    <div id="playback-settings-content"></div>
  </dialog>
  <div id="panel-menu" class="floating-menu" hidden></div>
  <dialog id="panel-editor"><form id="panel-editor-form">
    <span class="eyebrow">WIDGET BUILDER</span><h2 id="editor-heading">Add widget</h2>
    <label>Title<input id="editor-title" maxlength="80" required value="Telemetry"></label>
    <label>Visualization<select id="editor-view"><option value="stats">Current metrics</option><option value="chart">RGB MSE chart and metrics</option></select></label>
    <fieldset id="editor-metrics"><legend>Metrics</legend></fieldset>
    <div class="dialog-actions"><button type="button" id="editor-cancel" class="quiet">Cancel</button><button class="primary" type="submit">Save widget</button></div>
  </form></dialog>
  <div id="toast" role="alert" hidden></div>
