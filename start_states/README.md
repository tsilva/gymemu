# Named debugging starts

These small, curated snapshots are intentionally versioned. Each
`<game>/<name>.npz` contains the complete recorded RGB stack, executed native actions
between its frames, and JSON metadata with its name, description, dataset revision,
episode, chronological frame position, and original frame IDs. They contain no
simulator state. Runtime playback continues with predicted frames and user actions.

Breakout snapshots use held-out episodes from dataset revision
`676ff6388f4218d3c3a3ce9f2f33e075fa7314a3`, eight RGB frames and seven previous actions.
The ball coordinates below are pixel centers in the last recorded frame, with y
increasing downward. Each snapshot preserves the full eight-frame motion history.
Brick counts use the original 108-brick wall.

| Name | Episode | Frame position | Situation |
| --- | --- | --- | --- |
| `ball-up` | 5 | 45 | After a paddle bounce, ball moves up toward the bricks; x=53.5, y=149.5 |
| `near-bricks` | 5 | 58 | Ball still moving up, closer to the wall; x=27.5, y=110.5 |
| `paddle-approach` | 5 | 29 | Ball moves down toward paddle contact; x=85.5, y=180.5 |
| `half-cleared` | 50 | 2412 | 58 bricks broken, 50 remain; ball moving down-left at x=28.5, y=141.5 |
| `almost-cleared` | 90 | 2932 | 100 bricks broken, 8 remain; ball moving down-right at x=124.5, y=118.5 |
| `above-bricks` | 5 | 1204 | Ball emerged through the left side and moves up-right above the wall at x=25.5, y=36.5; 86 bricks remain |

Use `play.py CHECKPOINT --start-state ball-up`, or `--list-start-states` to list the
snapshots for that checkpoint's game. Reset restores the selected frame and action
histories. Selection is explicit and never silently falls back to another start.

Import an existing scene with `save_start_state.py CHECKPOINT --name NAME --scene PATH`.
Export a recorded scene with `--episode-id ID --frame-position POSITION` instead of
`--scene`; optionally provide `--split`, `--dataset`, and `--frame-cache`. Snapshots
are saved without overwriting an existing name. Use `--description` to explain the
situation and `--state-dir` for another library, such as ignored `artifacts/start_states`.

Stored RGB pixels and native actions are preserved exactly. Loading validates image
geometry, history length, action vocabulary, and required action history. Matching
game names are required for named lookup; matching dataset revisions are not, allowing
compatible models trained on different revisions to use the same debug inputs.
The original dataset revision remains part of each snapshot's provenance.
