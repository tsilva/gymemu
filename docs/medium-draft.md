# I tried to teach a neural network to be Breakout. The ball kept disappearing.

*Working draft, reconstructed on September 16, 2026. The full training experiment and final gameplay evaluation are unfinished. First-person narration is an editorial reconstruction of the project conversations.*

I kept asking for the latest checkpoint and the command to play it. The training loss could wait. I wanted to see whether the ball would survive its next collision.

Sometimes the paddle moved and the ball disappeared. Sometimes the ball survived and became two balls. Sometimes the model produced a recognizable Breakout screen for a few frames, then gradually erased the thing that made it a game.

This was the problem I had chosen for Gymemu. Give a neural network recorded gameplay and actions, then use it to generate the game interactively. Press left, predict the next frame, feed that prediction back in, and keep going.

The player would rely on the learned model to advance the scene. The dataset would provide the examples of how the game worked.

The repository goes back to April 2025. The detailed Breakout conversations I can reconstruct begin in March 2026. Looking back through them, the same questions keep returning, but with increasingly specific experiments behind them. What information does the model need? What does the loss actually reward? And what happens after the model has to live with its own predictions?

My first attempt made the images simpler. I already had a recording collected through Gymrec. I asked to crop away the score, reject damaged frames near the end whose backgrounds had turned magenta, and convert the remaining gameplay to black and white. The processed images were only 80 by 96 pixels.

I was training on an M1 MacBook Pro with 16 GB of memory. That was a real constraint. At one point, processing the dataset exhausted memory badly enough that I had to force the machine off. Keeping frames as compact integers and avoiding unnecessary copies mattered before any architecture choice could matter.

The first model had two stages. An autoencoder compressed a frame into a small latent vector and reconstructed it. A dynamics model learned to predict the next latent representation using the previous observations and an action. The decoder turned that prediction back into an image.

That sounded like a reasonable division of work. It also gave me several places for the ball to get lost.

Before I understood those failures, I got interested in duplicates. If the same frame appeared many times, why train on it repeatedly? Once the score and colors were removed, I expected even more frames to become identical.

That led to an early question that turned out to matter throughout the project. Was a frame plus an action enough to determine the next frame?

A screenshot shows where a ball is, but it may not show which way it is moving. Two identical pictures can belong to different moments in the game. I moved toward four-frame histories, hoping that motion would become visible in the input.

The first filtered, deduplicated export reduced 108,001 source rows to 3,669 unique stacked transition examples. The builder removed exact repeated history-action-target tuples. It did not establish that every remaining history had a unique possible future. Those are different claims.

It was a small dataset, and quick to train. Neither fact meant it was enough to learn the game.

My response after playing one of the early checkpoints was fairly direct: "gameplay sucks, try it yourself and tell me what you see".

The inspection agreed. Different action sequences converged toward almost the same output. The paddle barely responded, and there was no believable game progression. Increasing patience and watching validation loss plateau had not fixed that.

We tried direct pixel prediction, longer training rollouts, and additional refinement layers. Some changes improved particular cases. I started asking for generated frame strips at the end of every experiment so I would not have to discover every failure by launching the player myself.

Then the project took a wrong turn that belongs in this story.

Working with a coding assistant, I asked for stable idle behavior, then reliable paddle movement, then a visible moving ball. Some of the apparent progress came from hand-written runtime rules. The player held static scenes in place, moved the paddle deterministically in certain situations, and eventually tracked and drew a ball with explicit game logic.

For a while, that looked encouraging. I even wrote, "holy shit its working! just not seeing the ball".

But I wanted to know what was producing those frames. Were they model predictions? Were we predicting pixels or latents? How much Breakout behavior had been coded into playback?

Once that became clear, I asked to remove the corrections. A functioning hand-written Breakout implementation would not answer the question I was trying to investigate. I wanted the dynamics to come from training on the recordings.

We removed the old runtime and kept a spatial latent model, whose internal representation retained a grid rather than compressing everything into one small vector. That gave spatial relationships more room to survive. It did not immediately produce stable gameplay.

[Editorial placeholder: show an early failed rollout and, if the artifacts survive, distinguish the runtime-corrected version from the model-only version. Label both explicitly.]

In March, training moved to Modal so I could use CUDA without tying up the laptop. We tried longer unrolls, including sixteen-step training, and corrupted the starting history with noise and foreground dropout.

The reasoning was becoming clearer. During playback, every imperfect prediction becomes part of the next input. A model trained only on pristine recordings has little practice recovering from that situation.

I was reading about [DIAMOND](https://diamond-wm.github.io/) and [GameNGen](https://gamengen.github.io/). DIAMOND made the importance of small visual details particularly relevant to this project. GameNGen described adding noise to encoded context frames to reduce drift during generation. These were useful ideas to investigate, although their diffusion systems were very different from my model.

Our own corruption experiment did not become a success story. One later run selected its first epoch as the best checkpoint and deteriorated badly afterward. Longer rollouts and noisy history were plausible interventions, but implementing them was no guarantee that training would work.

By September, I returned to the project with a different setup. The code was simplified around a direct convolutional predictor, and the recordings now came from Gradlab trajectories using a native Breakout provider. This was a different dataset and simulator setup from the original Gymrec recording. Comparing their loss numbers as if they belonged to one continuous benchmark would be misleading.

The new reference model took eight RGB frames and the executed action, then predicted the next RGB frame. It used ordinary whole-image mean squared error. Alternative approaches lived in explicit configurations, with Hydra recipes recording how to reproduce them. I wanted to compare models and training pipelines without quietly changing the baseline every time I tried something.

The larger pinned dataset supplied roughly 5.39 million training examples and 1.34 million held-out examples per epoch. Training and evaluation used separate episodes. Frame histories stayed within an episode, and each action had to align with the transition it actually caused.

Even startup needed a clear definition. Generating an initial frame from empty history is a separate task from continuing a recorded scene. For ordinary playback, we saved a starting frame stack beside the checkpoint. After that initial context, the model had to supply its own frames. There would be no periodic injection of real screenshots to rescue it.

The question about sufficient history also returned, this time with an audit.

On 251,300 opening transitions from one dataset revision, identical four-frame histories and current actions still had different recorded successors in groups covering 2.69% of the examples. With eight frames, that figure was 1.30%. Longer histories reduced the observed ambiguity, but none of the tested lengths established complete state information.

The simulator investigation made the limitation concrete. Two constructed internal states produced 128 identical observations. They differed in a hidden paddle repeat counter. Applying the same right action then moved one paddle differently from the other.

A deterministic game can still be ambiguous to a predictor that only sees pictures.

I wondered whether frameskip was causing inconsistent observations. The evidence instead pointed to hidden controller state as a supported source of ambiguity. More frames might help, but there was no basis for declaring eight images a complete description of the game.

I proposed including previous actions alongside the frame history. The resulting recipe used eight action tokens, including the current action. When I tried an early checkpoint, paddle stability looked dramatically better.

That was encouraging playback evidence, not a controlled proof that all hidden state had been recovered. The ball still disappeared.

Whole-screen MSE kept coming up. Breakout contains a lot of background, a relatively static wall, and a very small ball. A predictor can get most pixels right while getting the ball wrong. If its inputs support several possible futures, squared-error training can also favor an average of those futures, leaving faint or blurred moving objects.

There was no single MSE threshold I could cross and declare the game playable.

We tried two explicit ways of giving the ball more attention. One recipe received aligned ball coordinates and predicted the next coordinates along with the image. Another added a separately averaged image loss around a detected ball in the recorded target. Neither approach inserted hand-written ball physics into playback, but both introduced Breakout-specific supervision. They belonged in named experiments with their assumptions visible.

The ball-region experiment produced one of the clearest lessons of the project. With a region weight of 0.3, a pixel inside the small target region could count about 85 times as much as a pixel outside it. Missing the real ball was expensive. Inventing an extra ball elsewhere was comparatively cheap.

Then I saw the ball fork.

Reducing the weight to 0.03 improved one narrow check. No persistent duplicates were detected in that particular sequence. But I tried a different starting scene, with most of the bricks already gone, and the fork returned.

Some apparently clean results had an even simpler explanation. The detector had lost the ball early, so it could no longer report two balls.

Counting duplicates alone was inadequate. I needed to check whether a recognizable ball survived at all, where it went, and what happened at collisions.

Named starting scenes became useful here. I could reset to a ball moving toward the wall, a half-cleared board, an almost-cleared board, or a ball above the bricks. That made failures easier to reproduce than repeatedly playing from the opening scene and hoping to reach the same situation.

Another correction came from looking at the real game. An assistant explanation treated the ball disappearing against bricks as an obvious prediction failure. I supplied a real-game frame where the ball itself was almost or completely hidden at contact.

The relevant test was whether it reappeared correctly afterward.

That made the memory problem more specific. A correctly drawn collision frame might contain very little evidence of the ball. Earlier observations still had to support its future motion. If the model failed to restore it, the next prediction would inherit an even less informative history.

I asked for teacher-forced playback so I could compare predictions with recorded targets. Every step would receive the real recorded history and action. We added an original frame, a prediction, and a signed difference image side by side.

In the examples I tested, the ball survived under teacher forcing while ordinary generated playback lost it. That made feedback errors a stronger suspect. To attribute the difference cleanly, the two modes still needed matching starting histories and actions.

The player gradually became a diagnostic tool. It moved into a browser dashboard with synchronized timelines and error charts. I could inspect the exact input stack, reorder frames to probe whether temporal order mattered, and paint changes into an input image before rerunning inference. These were ways to test the model's dependence on its inputs. They were not automatic explanations of its behavior.

[Editorial placeholder: insert a matched teacher-forced and autoregressive sequence with the same start and actions. Include the collision and the expected reappearance.]

We also trained with generated frames mixed into the history. The probability of using predictions increased during training, adapting the idea behind [scheduled sampling](https://arxiv.org/abs/1506.03099).

But I eventually realized that I had been using "train on predictions" to describe several different procedures.

Our September scheduled-history approach generated imperfect context without keeping its gradient graph, then supervised one final next-frame prediction. It practiced recovery from generated inputs. It did not apply a loss to every frame of a fully differentiable future rollout.

I wanted to try that too. Start from recorded history, generate several future frames using recorded actions, feed each prediction back in, and compare the whole generated sequence with the recorded sequence.

We used a curriculum of one, two, four, then eight predicted steps. The input history length and the future rollout length were separate settings. Near the end of an episode, only the available targets counted toward the loss. We never continued a training window into the next episode.

This also cost considerably more compute. The model now ran repeatedly within each training example. I kept asking whether Beast-3's RTX 4090 was being used effectively, and we benchmarked loading, caching, compilation, precision, and batch sizes. In the latest detached-training benchmark, the selected fast configuration processed about 1,795 windows per second at horizon eight, compared with 1,251 for its compiled reference.

Those were bounded throughput measurements, not whole-run completion times. Bigger batches also changed the number of optimizer updates. Speed improvements needed their own checks before being treated as interchangeable training setups.

Then the differentiable rollout experiment collapsed.

One checkpoint stopped producing a detailed scene and generated large, nearly uniform regions. Its held-out one-step error had worsened by roughly 185 times. The failure was visible even with recorded history, so it could no longer be explained only as drift during playback.

The diagnosis found that every convolutional weight had exactly zero gradient on eight sampled sequences. Only the final three RGB bias values still received gradients. Extreme activations at the final sigmoid were blocking learning through the rest of the network.

That identified the collapsed state. It did not establish the exact update that had caused it. The checkpoints did not contain Adam's optimizer state, so we could not replay an exact optimizer continuation.

I asked for the instrumentation that would have made the failure visible sooner. We added gradient statistics by layer, activation and saturation measurements, and fixed held-out rollout probes during training. I wanted to see prediction error beside rollout horizon and gradient behavior, rather than wait until an epoch ended to discover that the model had stopped learning useful dynamics.

This is where the project sent me back to the chain rule.

My first explanation for exploding gradients was that the same parameters were used repeatedly, so their gradient contributions were added together. That was only part of it. The loss was already averaged over the predicted steps. Averaging the number of terms did not bound the products of derivatives through the feedback path.

Repeatedly calling a convolutional model on its own differentiable outputs creates a temporal computation graph. A later error can travel through earlier generated frames and influence the shared weights through those paths.

I asked for examples I could calculate by hand, then worked through a simple repeated multiplication. The question had become much more concrete than "why is training unstable?" I wanted to understand which paths were amplifying the gradient.

We tested gradient clipping around the later transition from horizon four to horizon eight. The comparisons used the same starting weights and matched batch sequences. Clipping at 1.0 or 0.1 did not reliably improve the outcome. Some comparisons became much worse. All six final models had worse rollout MSE than the starting checkpoint.

These were short weight-restart experiments with fresh Adam state. They showed that clipping alone had not solved this particular test. They did not establish that clipping was useless in general.

The next experiment changed something more specific. We kept feeding generated images back into the model, but detached the history before each new prediction.

The forward process still encountered its own mistakes. Each predicted frame still had its own loss. We averaged those losses and updated the same shared parameters. What changed was the backward path. A later loss could no longer travel through the computation that had generated an earlier input frame.

There was a cost. That also removed the opportunity for a later error to teach an earlier prediction how to preserve information for the future.

Before training either arm, we checked that the forward predictions and loss values matched. They did. At horizon one, the gradients matched too. At longer horizons, allowing gradients through feedback could amplify their norm substantially.

After 1,000 horizon-eight updates, the detached version produced better rollout MSE in both matched batch-sequence comparisons.

| Batch sequence | Full feedback gradients | Detached feedback | Reduction in rollout MSE |
| --- | ---: | ---: | ---: |
| 47 | 0.00335 | 0.00198 | 40.8% |
| 48 | 0.01727 | 0.00198 | 88.5% |

Both detached results also improved by about 9.5% over the starting checkpoint. These were two batch orders from one trained checkpoint, evaluated on a fixed diagnostic subset. They were not independent full training runs, and they did not demonstrate a playable emulator.

They were enough to justify the next experiment.

The latest conversation records a ten-epoch detached-feedback run launched from scratch on Beast-3. At the last status recorded in that conversation, gradients looked stable, but the short generated-rollout probes were still noisy. The longer horizons had not yet been reached in that update.

That is where this draft stops. I have better experiments, more revealing diagnostics, and a specific training change worth testing at scale. I still need the model to keep the game coherent through sustained interaction.

[Final section to write after training and evaluation: identify the selected checkpoint and exact recipe, report held-out next-frame and generated-rollout results separately, and show uncut playback from several starting scenes. State the actual duration, actions, failure cases, and whether any correction is applied after initialization. Explain what finally worked only to the extent supported by comparisons.]

The clip I want to put here is a long one. It should show the paddle responding, the ball surviving collisions, and the board changing as bricks break. The reader should be able to watch long enough to judge whether the model learned a game worth playing.
