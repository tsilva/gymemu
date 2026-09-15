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

The default workspace has original, prediction, signed difference, input history,
controls, prediction-error chart, and model-context widgets. Original and difference
are unavailable during autoregressive playback. The signed difference's optional
1×/2×/4×/8× gain changes display contrast only; it does not change raw RGB MSE.

The workspace stores each instance's `type`, `title`, `config`, `enabled`, `builtin`,
and GridStack `placement`. Dragging, resizing, hiding, disabling, adding, editing,
and duplicating metric widgets persist through the server's local workspace file,
so the layout survives a new automatic port. Browser storage is a secondary copy.
Unknown widget types and module paths are never loaded from saved configuration.

The widget builder supports current metric cards and an RGB MSE chart with selected
metric cards. Add a metric descriptor in `telemetry.js` and its allowed key in
`workspace.js`. The chart always plots RGB MSE; unrelated units are not combined.
The chart retains at most 300 measured transitions and clears when replay seeks,
resets, changes episode, or changes playback mode.

`services.command()` sends playback controls to the single inference worker.
`services.updatePanel()` changes a widget's config. Widgets must remove their
listeners and dispose of retained resources in `destroy()` when applicable.

Run `node --test tests/web_player/*.test.mjs` from the repository root. Assets are
served directly by Python; no frontend install or build is needed. GridStack 12.6.0
is the checked-in Gradlab vendor asset, not a runtime CDN dependency. See the
adjacent Gradlab license and third-party notices for copied assets.
