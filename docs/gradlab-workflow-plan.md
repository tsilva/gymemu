# Gradlab-style research workflow

Status: agreed design, ready for implementation. This plan changes Gymemu's research workflow, not its direct-CNN reference model or the meaning of its comparison metric.

## Intended workflow

The browser path is Environment -> Goal -> Goal Revision or Goal Variant -> Run -> Checkpoint -> Player. A Goal page also lists its checked-in recipes. A Run page shows the resolved Goal and recipe, launch overrides, stages, status, metrics, and playable checkpoints. The user can move backward and forward through these screens with browser history.

Local direct training and dstack training publish the same Run record. The catalog does not split Runs by compute location or scan local directories. R2 holds the authoritative Run catalog and artifacts. W&B holds metric history and diagnostic media. The UI remains a token-protected local application, as in Gradlab.

## Research contracts

Add checked-in Goals under `experiments/goals/<environment>/<goal>/`. Start with one Breakout next-frame prediction Goal for recipes that share the same dataset revision and held-out targets. The Goal fixes the dataset identity, split and target selection, float32 next-frame RGB MSE contract, and exact held-out rollout probe starts and horizon. Store selected episode, offset, and frame IDs rather than depending on dataset iteration order. A new scientific target or evaluation question gets a new Goal.

Hash the checked-in, resolved Goal contract to identify its revision and retain its immutable snapshot after later edits. A launch-time Goal change creates a content-identified Goal Variant with a normalized difference from that revision. Show current revision, variants, and historical revisions separately. Model architecture, auxiliary losses, optimizer settings, and training budget belong to recipes or launch-time recipe overrides. A playback-only starting scene belongs to the Run; it changes Goal identity only when it is one of the fixed evaluation probes.

Move launchable recipes under their owning Goal while retaining compatibility aliases for existing `recipe=...` commands and standalone saved recipes. Resolve Hydra composition before launch, save the complete recipe and its hash, and record each launch-time override. Keep `approach=direct` unchanged. One Run contains the ordered stages of a multi-stage approach and every model needed for inference.

Only compare Runs when their dataset revision, held-out targets, and metric contract match. Never train on held-out trajectories, cross episode boundaries, or treat image IDs as array offsets. Rank complete prediction checkpoints by held-out float32 RGB MSE. Show rollout results as diagnostics without treating low MSE as proof of playable rollouts. Keep stage losses distinct from the comparison metric. Do not add Acceptance, Promotion, scientific success thresholds, or scientific success badges in this migration. A completed Run means training and required publication finished.

## Run and publication contract

Create a stable Run ID before execution. Its manifest records Goal revision and Variant, resolved recipe and overrides, dataset provenance, source and image identity, stage inventory, compute placement, W&B identity, R2 object references, and lifecycle state. Use one manifest format for direct local training and dstack tasks. Build immutable, hash-verified artifact objects and publish manifest updates with conflict protection so retries retain the same identity. Keep recovery receipts separate from playable artifacts.

Online `gymemu train` preflights R2 before optimization and publishes to W&B and R2 by default. Preserve local output on later publication failure and provide `gymemu sync <run-directory>` to finish publication without retraining or assigning a new Run ID. Disabled or offline W&B modes make no R2 publication and do not appear in the catalog until sync. A queued `gymemu experiment launch --recipe-file ... --compute local|spot|on-demand` uses dstack for placement, preflights credentials and the immutable image, records its manifest before submission, and supports `--follow`. Put finite duration and cost bounds on paid compute, with no silent switch to on-demand. Compute selection does not change the resolved recipe. Keep direct `gymemu train` as the local training path.

R2 stores checkpoints, the resolved Goal and recipe, provenance, playback scene, probe manifest, scalar diagnostic history, and a bounded set of comparison images. W&B receives the metric series and images. The private control catalog and recovery files require scoped credentials. Publish only the inference bundles and playback artifacts intended for public access through hash-verified URLs. Never expose credentials or optimizer/RNG recovery state through those URLs.

The player catalog lists inference-ready checkpoints only. Keep `resume.pt` in recovery details because it is not a playable checkpoint. Show intermediate-stage inference checkpoints under the same Run with stage labels, outside the final prediction ranking. Verify every downloaded checkpoint and scene by size and SHA-256 before loading them. Preserve version-1 direct-CNN checkpoint loading and explicit playback by local path.

## Catalog and UI

Replace the current local scan plus R2 listing in `gymemu/catalog.py` and `gymemu/remote_catalog.py` with a rebuildable, paginated R2 catalog projection. Update it from Run publication events; do not list every R2 object for each browser request. Treat missing catalog authority as unavailable, not as an empty catalog. Read W&B for optional metric enrichment, not for Run existence or authoritative state. Do not import existing local or R2 Runs into the new index. Existing checkpoints remain usable by explicit path; new comparable Runs can be retrained after migration.

Expose catalog APIs for environments, Goals, revisions and variants, recipes, Runs, and a selected Run's checkpoints. Return normalized Goal differences, recipe overrides, comparability identity, stage, Run state, and available metric columns. Keep direct Run-to-checkpoint navigation and browser history. In the UI, remove the Local/R2 source column and the local-run footer. Show a W&B link and explicit missing-metric state. Provide resolved Goal and Run configuration as YAML. Keep the existing paused teacher-forcing player behavior and its autoregressive mode.

## Implementation order and gates

| Step | Change | Gate |
| --- | --- | --- |
| 1. Contracts | Add Goal schema, revision and Variant identity, goal-owned recipes, alias resolution, and comparability keys. | A checked-in Goal edit creates a revision; a launch-time Goal override creates a Variant; a recipe-only override creates neither. Direct and multi-stage recipes resolve without baseline drift. |
| 2. Publication | Add the shared Run manifest, stable IDs, R2 receipts, W&B identity, artifact verification, local sync, and retry behavior. | A failed upload resumes with the same Run ID and verified objects. Offline Runs stay out of the R2 catalog until sync. |
| 3. Execution | Route direct training and dstack launch through the same resolved contract and manifest. Add preflight, follow, status, and bounded recovery behavior. | The same recipe resolves identically on local and queued paths; compute changes only placement. |
| 4. Catalog | Build the indexed R2 projection and paginated APIs. Stop merging local and remote listings. | One published Run appears once; unavailable R2 returns an error rather than a misleading empty list. Browsing makes no per-request bucket scan. |
| 5. UI and player | Implement the Gradlab navigation path, Goal and recipe inspection, Run differences, comparable metrics, stage grouping, and checkpoint filtering. | Browser back/forward, search, direct Run selection, and verified playback work. `resume.pt` cannot appear as playable. |
| 6. Verification and docs | Run Python and web checks, bounded direct and multi-stage train/play smoke, and in-app browser tests. Update README, training, recipe, approach, and metrics docs. | Direct local training and a bounded queued dstack run complete end to end through publication, discovery, metrics, and playback. Exercise both direct and multi-stage approaches across these checks. Other recipes can be retrained as needed. |

Run `uv run pytest` and `uv run ruff check .` for the affected infrastructure. Exercise both direct and multi-stage paths whenever shared Run code changes. Use the native Codex in-app Browser for the UI check. Preserve the current branch and the uncommitted player work while implementing this plan. Do not start a full retraining campaign as part of the migration gate.
