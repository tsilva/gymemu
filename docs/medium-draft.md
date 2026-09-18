# Teaching a neural network to be Breakout

*Working draft, September 17, 2026. The final gameplay result is still open.*

I wanted to press a key and have a neural network generate the next Breakout frame. That frame would become its next input. Repeat this process, and the model would run the game.

I called the project Gymemu. My main test was simple: could I actually play it?

In March, I started with recorded gameplay, small black-and-white images, and an M1 MacBook with 16 GB of memory. An autoencoder learned to compress and rebuild images. Another model tried to predict what came next.

The first models barely responded to actions. Later versions moved the paddle, but lost the ball or stopped it in place.

Working with a coding assistant, I also took a wrong turn. Some apparent improvements came from hand-written rules in the player. Those rules moved the paddle and handled ball collisions. I asked to remove them. I wanted the model to learn those behaviors from data.

By September, I had a larger color-image dataset and a simpler reference model. Training moved to Beast-3's RTX 4090. Saved recipes made experiments easier to repeat.

The ball remained the main problem. Sometimes it disappeared. Sometimes it became two balls.

Our loss was mean squared error, or MSE: an average of squared pixel differences. The ball occupies very few pixels. A low average error could therefore hide a serious gameplay failure. Giving the ball more weight helped in some tests, but made extra balls relatively cheap.

I added playback with real recorded history at every step. This is called teacher forcing. It helped separate errors in one prediction from errors caused by feeding predictions back repeatedly. A sequence of predictions fed back this way is a rollout.

Even real history could leave information missing. We found identical image and action histories with different next paddle positions. The game's hidden controller state mattered. More history reduced the conflicts we observed, but did not prove exact prediction was always possible.

Training through longer rollouts brought another problem. Gradients, the values used to update model weights, became unstable. One failed model stopped receiving useful gradients through its convolutional layers.

We then detached generated history before each prediction. The model still saw its mistakes, but later losses could no longer flow backward through earlier predictions. This helped short comparisons. It also removed some information that could help the model learn long-term effects.

Here is the experiment sequence recovered from the project records. Rows summarize comparisons, including failed runs and diagnostic tests. March and September used different datasets. Their loss values are not directly comparable. Speed tests measure bounded workloads, not complete training time.

| Order / date | Experiment: what changed | What we observed | Conclusion, if any |
| --- | --- | --- | --- |
| 1 · Mar 8 | Crop, binarize, train an autoencoder and dynamics model | Training ran; validation soon stalled | Training completion did not establish gameplay. |
| 2 · Mar 8 | Four-frame histories; remove repeated transition examples | 3,669 unique examples remained | Smaller data did not prove adequate coverage. |
| 3 · Mar 8–9 | Longer training before early stopping | Outputs barely responded to actions | Extra patience did not solve the failure. |
| 4 · Mar 8–9 | Predict pixels directly | Actions affected output; scenes drifted | Partial improvement. |
| 5 · Mar 8–9 | Four-step pixel rollouts | Drift remained | Short rollouts were insufficient in these tests. |
| 6 · Mar 8–9 | Eight-step pixel rollouts | Less idle drift; movement still unstable | Longer training sequences helped some cases. |
| 7 · Mar 8–9 | Sample training windows more densely | Lower validation loss; worse rollout images | Better loss did not select better playback. |
| 8 · Mar 9 | Add three refinement blocks | Left movement improved slightly | More capacity helped, but remained unplayable. |
| 9 · Mar 9 | Hand-written idle hold, paddle movement, then ball rules | Idle, controls, and collisions improved | Improvements came from code; removed this approach. |
| 10 · Mar 9 | Spatial features, motion losses, sampling changes | Moving ball still froze | No stable learned dynamics demonstrated. |
| 11 · Mar 9 | Warp images using predicted motion | Backward pass fell back to CPU; short rollout drifted | No stable result from this prototype. |
| 12 · Mar 9 | Inspect recorded FIRE examples | No detected launch examples in the valid subset | Data coverage was a concern. |
| 13 · Mar 10 | Train eight-step spatial model on Modal | Completed; playback still drifted | More compute had not solved stability. |
| 14 · Mar 10 | Extend training to sixteen steps | Reported validation loss improved | Stable gameplay remained unverified. |
| 15 · Mar 10 | Add noise and missing pixels to history | Unstable run; incorrect loss reference found | Stopped and fixed the reference. |
| 16 · Mar 10–11 | Retrain with the corrected reference | Loss improved; user still reported poor playback | The fix was insufficient. |
| 17 · Mar 11–12 | Correct dropout, share noise over time, change model selection | Best epoch was first; later performance worsened | This combined experiment failed. |
| 18 · Sep 13 | Audit histories from 1–64 frames | Fewer observed conflicting targets with longer history | No tested length proved complete state recovery. |
| 19 · Sep 13 | Construct equal-looking simulator states | Same action produced different paddle movement | Hidden state can affect future images. |
| 20 · Sep 13 | Cache frames and optimize training execution | 4.57× benchmark speedup | Data loading was a major bottleneck. |
| 21 · Sep 13 | Train direct color-image reference model | Held-out next-frame MSE 0.00003076 | Useful baseline; playability unproven. |
| 22 · Sep 13 | Add previous actions | MSE 0.00002862; paddle looked steadier | Helpful evidence; ball failures remained. |
| 23 · Sep 13 | Gradually mix predictions into history | Slow training; ball still disappeared in previews | Stopped to improve throughput. |
| 24 · Sep 13–14 | Optimize mixed-history training | 1.36–1.68× benchmark speedup | Faster, below the requested 4×. |
| 25 · Sep 14 | Complete optimized mixed-history run | Best MSE 0.00003423, worse than action-history baseline | No one-step improvement; rollout benefit unresolved. |
| 26 · Sep 14 | Predict ball coordinates alongside images | RGB MSE 0.00002929 | Coordinate supervision did not prove stable playback. |
| 27 · Sep 14 | Ball-region loss weight 0.3 | Persistent duplicate balls appeared | Missing and extra balls had unequal penalties. |
| 28 · Sep 14 | Reduce weight to 0.03 | One check improved; another still forked | Lower weight did not eliminate duplicates. |
| 29 · Sep 14 | Combine ball-region loss with mixed history | Ball recovery still failed in inspected collisions | Recovery remained unsolved. |
| 30 · Sep 14 | Compare recorded-history playback and real collisions | Ball survived tested replay; real frames sometimes hid it | Check reappearance, using matched starts and actions. |
| 31 · Sep 15–16 | Train through 1, 2, 4, then 8 predicted steps | Model collapsed during four-step phase | Longer differentiable rollouts could destabilize training. |
| 32 · Sep 16 | Inspect collapsed model and soften its output function | Convolution gradients were zero; softening restored them | Output saturation blocked learning; initial trigger unknown. |
| 33 · Sep 16 | Optimize differentiable rollout training | 2.32× speedup; larger batch changed update count | Faster execution was not a quality guarantee. |
| 34 · Sep 16 | Restart full-gradient training with diagnostics | Saturation and prediction errors rose again | Instrumentation exposed warning signs. |
| 35 · Sep 16 | Clip gradients at two thresholds | Mixed outcomes; all six models worsened from their start | Clipping alone was unreliable here. |
| 36 · Sep 16 | Compare full and detached feedback before updates | Equal predictions; full-feedback gradients could be much larger | Feedback paths amplified gradients on tested batches. |
| 37 · Sep 16 | Train matched detached-feedback comparisons | Rollout MSE fell 40.8% and 88.5% against controls | Promising short tests from one starting checkpoint. |
| 38 · Sep 16 | Optimize detached training and check short continuations | 1,795 versus 1,251 windows/s; small mixed error changes | Faster; long-run equivalence unproven. |
| 39 · Sep 16 | Start full eight-frame detached run | Improved MSE; stopped during epoch 6 | Incomplete; old checkpoint lacked optimizer state. |
| 40 · Sep 16–17 | Fit linear ball-position predictors to frozen features | Position errors around 3–4 native units | Features contained useful, imperfect ball information. |
| 41 · Sep 17 | Benchmark four-frame/four-step training and resume | About 3,566 windows/s; resume check passed | Less prediction work was faster. |
| 42 · Sep 17 | Complete four-frame/four-step training | Best MSE 0.00003126, about 8% worse than eight-frame run | Both history and rollout changed; cause not isolated. |
| 43 · Sep 17 | Reproduce paddle blur; change actions and device | Action response remained; CPU reproduced the blur | Ignored actions or Apple playback were weak explanations. |
| 44 · Sep 17 | Search identical four-frame/four-action inputs | 3,797 groups had different next paddle positions | Conflicting targets explained some softened edges. |
| 45 · Sep 17 | Extend image and action histories together | No observed paddle conflicts at 64 frames/actions | Shortest tested conflict-free length, not a universal minimum. |
| 46 · Sep 17 | Benchmark single-frame reconstruction | Selected setup reached about 21,094 frames/s | Fast enough to test appearance separately. |
| 47 · Sep 17 | Train only image reconstruction | Best held-out MSE 1.77 × 10⁻⁸ | Strong reconstruction score; dynamics not tested. |
| 48 · Sep 17 | Prototype a fixed brick detector | Held-out counts matched; startup caused mismatches | Useful labeling method with explicit limits. |
| 49 · Sep 17 | Train paddle predictors on extracted image features | One frame/four actions met the chosen accuracy target | Practical prediction, not exact state recovery or full-image learning. |
| 50 · Sep 17 | Validate brick labels across the dataset | Non-startup counts matched; startup states flagged | Count checks passed; individual cells lack independent ground truth. |
| 51 · Sep 17 | Request paddle prediction from full color images | Follow-up begun; no result in reviewed record | Conclusion pending. |

The latest work separates appearance from motion. The reconstruction model rebuilds the frame it receives. It does not predict the next frame. Its tiny error must not be compared directly with next-frame prediction error.

The paddle study also needs a clear limit. Its first models received positions extracted from images, not full images. They cannot establish how much history an image model needs.

I now have narrower questions and tests for them. The final test is still sustained gameplay: a responsive paddle, correct collisions, and a ball that keeps moving.

[Finish after final evaluation: name the selected model, report matched rollout results, show uncut gameplay from several starts, and describe remaining failures.]
