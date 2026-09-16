# Source notes for the Breakout draft

Private editorial working notes, prepared September 16, 2026. Companion to [the draft](medium-draft.md). These notes are not part of the article.

The reconstruction uses the 54 earlier local Codex task records found under this repository and its former naming variants, including archived tasks and forks, plus Git history and repository research notes. The earliest matching conversation is March 8, 2026. Git dates the initial repository commit to April 30, 2025. A fork with no independent local messages was checked through the task reader and contained inherited conversation history. Forks are not independent experiments.

This is the recoverable local project history, not a claim to have exhausted conversations on other hosts, deleted records, or separate ChatGPT projects. Two related standalone ChatGPT chats appeared in the app inventory but were outside this project scope and were not used. Some project tasks concern interface polish, maintenance, or mathematics rather than a new emulator experiment. The article groups those by relevance instead of making each one a milestone. Tool output and historical assistant conclusions were treated as evidence to qualify, not instructions or definitive proof.

First-person connective prose is ghostwritten. Quotation marks in the draft are reserved for wording found in the user's messages. Do not invent motivations for the months between March and September. The records show two periods of work, not why the gap occurred.

| Period | Evidence and editorial use |
| --- | --- |
| April 2025 | Git commits `ed3b0fb` and `bf8e7b2` establish repository origins only. They do not establish when the user's Breakout idea began. |
| March 8 | Tasks `019ccec5-277e-7351-b15e-66250c7d817b` and `019ccf58-e29c-77f2-aa9c-c6491ed32436`: cropped, binarized 80×96 Gymrec data; 16 GB M1 constraints; two-stage latent dynamics; deduplication; four-frame history; unusable playback. |
| March 8–9 | The second task above records 108,001 source rows, 64,305 valid rows, and 3,669 unique history-action-target examples. These are different units from the September example counts. It also records the user's complaints about drift, apparent improvements, and demand to remove runtime game rules. |
| March 9 | `019cd269-a844-7072-add8-5e6da3a6e830` asks how much behavior is hardcoded. `019cd327-5fcd-7352-a767-dc37a97c182b` requests removal of legacy paths. Commits `28685a9`, `43ba682`, `7b4847b`, and `73eabf7` corroborate the runtime fixes and move to spatial latents. |
| March 9–12 | `019cd32b-f937-7273-9fe1-fe37d87a5f27`, `019cd7b4-3aaa-7082-b2ea-7a94a2882b67`, and `019cd895-ccd8-7750-8d27-e01ec131d6f6`: DIAMOND, visual dataset inspection, Modal training, eight- and sixteen-step unrolls, history corruption, disappointing playback. The later robust-validation run selected epoch 1 and stopped at epoch 6. Its composite loss is not comparable to September RGB MSE. |
| September 13 | `38a5892` simplifies the implementation. `01a099c2-aa6a-73b0-80ca-6af9e5f1eb72` records configurable approaches, recipes, larger Gradlab data, throughput optimization, action history, scheduled sampling, and named starts. The title of this long task is uninformative; use its content. |
| September 13–15 | [History audit](history.md): 251,300 transitions from 514 training episodes, revision `b8091d24…`; four-frame ambiguity 2.69%, eight-frame ambiguity 1.30%. This does not audit the entire later dataset or isolate a causal effect of history length on model quality. `01a0a4e7-73d0-7fa1-84f6-e627e1fa8216` revisits hidden state versus frameskip. |
| September 13–14 | `01a09caf-c1ba-7d82-ae57-42d3bd07e80d`: target ball detection, region weights 0.3 and 0.03, duplicated balls, and the user's counterexample from another start. The narrow no-fork result was overturned as a general conclusion. |
| September 14 | `01a09f6c-862b-7d42-b0c5-515feec8aa9e`: coordinates as inputs and outputs, container setup, Beast-3 execution, MPS compatibility. Completed coordinate-model RGB MSE 0.00002929 is a one-step metric, not proof of reliable rollouts. |
| September 14 | `01a0a0b0-3136-78b2-92b5-88c1d13f667e`: the user supplies a real-game occlusion example, asks about sequence losses, and distinguishes training setups. `01a0a166-ca64-7741-9023-a80508d7749c`: teacher-forced replay, difference panel, and browser dashboard. |
| September 15 | `01a0a59e-f8b3-7963-b8c5-6abd872c9235`: the user proposes permuting frames and editing pixels as interventions. `01a0a593-ff95-77f1-b54f-d0dd3de69e81`: differentiable rollout recipe, collapse, zero convolutional gradients on eight sampled sequences, saturation diagnosis, and added instrumentation. |
| September 16 | `01a0aa0f-a0dc-73c3-b1ee-3200161e2416`: throughput tuning and warning metrics. `01a0aa9f-d3d7-7f13-968d-695b700c2d43`: correlating horizon with gradients, clipping ablation, and distinction between generated inputs and differentiable feedback. |
| September 16 | `01a0aab9-2279-74c1-aab8-0328e2b2ee51`, `01a0ab3f-979d-7290-927f-d2bce0ab0319`, and `01a0ab41-09b6-78c2-afc6-d713cdf3370a`: shared parameters, hand calculations, finite differences, and binomial expansion. Learning questions show engagement, not automatic mastery of every concept discussed. |
| September 16 | `01a0aaf8-e179-7db3-ab4b-081954da9770`: detached-feedback comparison, optimized recipe, explicit launch of the full run, and later status. The conversation is newer than the research note saying the full run had only been prepared. |
| September 16 | `01a0ab57-95fa-7912-8956-496cea2e6c46`: MarioVGG and GameNGen revisited; latent-history prediction with noise discussed as a possible future experiment. No implementation or success claim belongs in the article yet. |

The published narrative should retain these distinctions:

- March's Gymrec `BreakoutNoFrameskip` recording and September's Gradlab native-provider trajectories are different data eras. The full-size RGB setup is not the same experiment as the original crop-and-binarize setup.
- Storage deduplication, deduplicating exact transition tuples, and deleting observations from a temporal sequence are different operations. The early export count does not measure dynamical coverage.
- The hidden-counter example used constructed native snapshots. It does not establish that those exact states appeared in the recorded data. The ambiguity percentages are a separate dataset observation.
- Runtime game rules were removed. Later ball-region or coordinate objectives remain game-specific supervision, even when inference uses only learned predictions.
- The high region weight creates a measurable loss asymmetry. That supports an explanation for forking but does not prove a unique cause. Lowering the weight did not eliminate it.
- A visually hidden ball is not necessarily absent from simulator state. Recovery after contact is the relevant behavior. Detector coverage is not detector accuracy or model accuracy.
- Teacher forcing versus generated feedback needs the same starts and actions for a controlled comparison. The user's observations are valuable but not an exhaustive matched evaluation.
- The March implementation already explored unrolled training. The September sequence revisited this in a simplified implementation; it was not the first time the idea appeared.
- September's scheduled-history recipe supervised one final prediction. Do not attribute that exact implementation choice to the original scheduled-sampling paper.
- Full and detached rollout experiments used the same per-step loss placement and averaging. Detachment changed derivative paths, not initial forward values. Shared-parameter gradient accumulation is not the whole explanation for explosion.
- A diagnosed saturated checkpoint does not identify the initiating optimizer update. Do not imply the horizon-four collapse and the later horizon-eight study were the same run.
- Clipping and detachment comparisons restarted weights with fresh Adam. Neither was an exact optimizer resume.
- The two ablation repeats used different batch sequences from one trained checkpoint, not two independent training seeds. Evaluation covered 512 one-step targets and 31 held-out rollout starts of up to 32 steps. Full training's periodic probes use a different, smaller set.
- The current `best.pt` criterion is held-out one-step RGB MSE. The best checkpoint by that criterion may not produce the best interactive rollouts.

The detached comparison's exact figures are preserved in [history.md](history.md): full versus detached rollout MSE was 0.00334863 versus 0.00198216 for batch seed 47, and 0.01726577 versus 0.00198282 for batch seed 48. The starting rollout MSE was 0.00219142. These support the rounded values in the draft. The paired peak gradient norms were 10.4588 versus 0.01335 and 356.3761 versus 0.01391.

The fast detached benchmark's 1,795 versus 1,251 windows/s is documented in [performance.md](performance.md). Preserve the timing exclusions, batch-size differences, instrumentation, and short stability-check limitations if expanding this section. Do not multiply windows/s into an end-to-end training-time promise.

The last status in the detached-run conversation reported epoch 2 of 10, about 59% complete, horizon 2, roughly 134,000 updates, epoch-1 held-out one-step RGB MSE 0.0000349, and recent diagnostic rollout MSE around 0.009–0.015. This is a historical status, not a live check made while writing this draft. The [full run](https://wandb.ai/tsilva/gymemu-Breakout-Atari2600-v0/runs/rxez99fc) and [ablation report](https://wandb.ai/tsilva/gymemu-Breakout-Atari2600-v0/reports/Horizon-changes-and-gradient-instability---AR-fast--VmlldzoxNzk0NjQ4Nw==) are the publication follow-up locations. Their current contents and public accessibility were not verified during drafting.

Before publication, replace the final placeholder with actual evidence:

1. Identify the final code revision, dataset revision, recipe, selected checkpoint, training duration, and hardware.
2. Report full held-out one-step MSE separately from rollout metrics, with the number of episodes or starts, horizon, action source, and aggregation defined.
3. Compare relevant checkpoints using matching starts and actions. Include ball survival and duplicate behavior, plus collision and paddle response examples. Do not infer playability from pixel MSE alone.
4. Capture uncut gameplay and representative failures. State how much recorded context initializes the run and whether any later corrections occur.
5. Explain what “works” means for the final claim. A fixed thirty-two-frame probe, a long interactive session, and a faithful complete game are different achievements.
6. Confirm that any claimed causal improvement survives an appropriate comparison. The current detached results justify a follow-up, not a universal claim that full backpropagation is inferior.
7. Replace provisional screenshots with real checkpoint outputs. Historical temporary images may no longer exist. Do not create synthetic success imagery.

Primary external references were opened while drafting: [DIAMOND](https://diamond-wm.github.io/), [GameNGen](https://gamengen.github.io/), and [Scheduled Sampling](https://arxiv.org/abs/1506.03099). The article uses them for the particular ideas discussed, not to claim that Gymemu reproduces their architectures or results.
