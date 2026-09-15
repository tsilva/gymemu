# Player widgets

This workspace adapts Gradlab's panel registry and module lifecycle. `catalog.js`
defines reusable types and default instances; `workspace.js` validates saved layout
and widget configuration; `runtime.js` mounts modules and dispatches snapshots.
`shared.js` contains Gradlab's panel shell, metric cards, and canvas chart helpers.

A widget exports `mount({ definition, services })` and returns an `element` plus
optional `render(snapshot, view)`, `resize()`, and `destroy()` methods. The definition
contains its stable ID, title, type, enabled state, and config. Register a new type
with a local module path and minimum size, then add its default instance if needed.
Keep model-specific inference out of widgets and the transport.

All images and metadata in a snapshot belong to one playback revision. The browser
decodes the images before committing that revision to all enabled widgets. Render
images from `view.bitmaps`, and metrics from the same supplied snapshot. The input
history atlas is oldest to newest, laid out horizontally at native frame dimensions.
Never compute model inputs from resized or displayed canvas pixels.

The default workspace assigns original, prediction, and signed difference to `main`,
with input history, prediction-error chart, model context, and custom telemetry in
`stats`. Playback controls live in the main tab's gear dialog. Original and difference
are unavailable during autoregressive playback. The signed difference's optional
1×/2×/4×/8× gain changes display contrast only; it does not change raw RGB MSE.

The workspace stores each instance's `type`, `title`, `config`, `enabled`, `builtin`,
and GridStack `placement`. Main comparison panels use an equal-column flex layout
that fills the available viewport, with matched footer heights and no manual resize.
Stats uses GridStack for dragging and resizing. Hiding, disabling, adding, editing,
and duplicating metric widgets persist through the server's local workspace file,
so the layout survives a new automatic port. Browser storage is a secondary copy.
Unknown widget types and module paths are never loaded from saved configuration.
Version 3 migrates earlier single-tab layouts into paired views. The browser uses
BroadcastChannel and storage events to synchronize ordered layout revisions; Python
rejects delayed writes older than its saved revision. Both tabs poll the same
server-owned snapshot stream. Stats sends explicit chart seeks through the same command queue, guarded by episode and history epoch. It sends no keyboard, heartbeat, or blur commands.

The widget builder supports current metric cards and an RGB MSE chart with selected
metric cards. Add a metric descriptor in `telemetry.js` and its allowed key in
`workspace.js`. The chart always plots RGB MSE; unrelated units are not combined.
The chart retains all measured episode transitions across seeks, deduplicated by step.
Reset, episode changes, and mode changes clear the series and increment history_epoch.
Unvisited steps stay unscored. services.inspectStep selects a measured target;
services.setChartRange synchronizes zoom through BroadcastChannel and storage events.
chart-range.js shares Gradlab's drag zoom, double-click reset, and timeline handles.
The hover cursor and selected-step reference use actual timestep coordinates.

`services.command()` sends playback controls to the single inference worker.
`services.updatePanel()` changes a widget's config. Widgets must remove their
listeners and dispose of retained resources in `destroy()` when applicable.

Run `node --test tests/web_player/*.test.mjs` from the repository root. Assets are
served directly by Python; no frontend install or build is needed. GridStack 12.6.0
is the checked-in Gradlab vendor asset, not a runtime CDN dependency. See the
adjacent Gradlab license and third-party notices for copied assets.
