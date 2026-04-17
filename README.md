# GB Electricity Synthetic Data Pipeline

An end-to-end research pipeline for generating synthetic GB electricity price paths and using them in downstream gas storage stochastic control and deep hedging. The project starts from free public UK electricity-market data, builds a shared econometric and event-risk backbone, and then swaps different **Layer 3** generative models on top of the same market structure.

This repository is a research build, a learning log, and a portfolio project. The point is not to hide the failed ideas. The point is to keep the failed ideas runnable, diagnose why they failed, and then make the next architecture answer a concrete measured weakness rather than just chasing a new model because it sounds fancier.


---
## About this project

This is not a production system and it is not meant to read like one. It is a personal research project built partly to produce a working simulator for GB electricity prices, and partly as a way of learning the mathematics and modelling ideas I most wanted to go deeper on. I would rather keep the real sequence of ideas, mistakes, dead ends, and rebuilds visible than pretend the final structure appeared fully formed.

The project started from a fairly concrete place. I was applying for energy and data roles, looking at publicly available GB electricity market data, and realised that this market is unusually well suited to serious experimentation: the APIs are free, the price process is structurally awkward, negative prices are real, wind matters, transmission constraints matter, and the data forces you to confront regime changes, volatility clustering, and event-driven behaviour rather than hiding them.

That is exactly why I kept going. GB electricity is not a convenient toy dataset. It is difficult in the right way. The project became a way to study how generative models behave when the underlying market genuinely has jumps, state changes, weather dependence, and policy-driven distortions.

I have left the failed and superseded stages visible on purpose. Stream 1 is still in the repo because the point is not to hide what did not work. The point is to show what each architecture could and could not do, what was learned from that, and why the later streams exist.

---

## Background and Motivation

This project sits at the intersection of several things I have been building toward for a while.

My undergraduate dissertation was on a stochastic exchange model of wealth inequality, so discrete random systems, Monte Carlo simulation, ergodicity, and state transitions were already natural objects for me. My postgraduate work then moved much further into mathematical finance: stochastic calculus, derivative pricing, econometrics, numerical methods, and neural-network-based option pricing. Rough paths, signatures, and generative models arrived later, but once they did, they felt like the right language for sequential market data. They were things I read about and grew seriously interested in.

That is part of why this project exists in this form. It is not just “an energy market repo”. It is where several mathematical threads and interests of mine meet:

-stochastic processes and state dynamics
-econometrics and volatility modelling
-path signatures and rough-path ideas
-generative modelling for sequential data
-stochastic control and deep hedging

The project is therefore doing two jobs at once: building a simulator for a difficult real market, and serving as a serious learning vehicle for the mathematics I want to keep developing.

---

## Current status

**As of 17 April 2026:**

- **Streams 1 and 2 are complete and frozen** as the baseline and the econometrically repaired CVAE track.
- **Layer 2 is complete and settled** as the shipped spike/negative hazard process on the native half-hour grid.
- **Stream 3 is complete enough to freeze as the current SigCWGAN candidate.** It materially improved on the CVAE baseline and gave the project a reusable multistream validation workflow.
- **Stream 4 has already produced useful research findings.** The Neural SDE can train stably and use conditioning, but its continuous Brownian paths still look too smooth for GB electricity spike-revert behaviour.
- **Stream 5 is the next active build.** It will be a jump-aware conditional diffusion on return paths.
- **Project B** is the informed rebuild for later: a full jump-driven architecture in which jump timing and jump size are modelled directly rather than treated as a side effect of a continuous process.

The repo is now best understood as a **shared three-layer system** with multiple competing Layer 3 generators rather than a loose sequence of unrelated notebooks.

---

## What this project is actually trying to do

The practical goal is to build a realistic simulator for GB electricity prices that is good enough to use in downstream stochastic control and deep hedging experiments.

The research goal is slightly broader: to work out which model classes can and cannot reproduce the market object I care about.

GB electricity is a very good stress test for this because the data is structurally awkward:

- negative prices are real rather than pathological,
- wind generation is a first-order market driver,
- volatility clusters in concentrated episodes rather than smoothly,
- spike behaviour is economically meaningful,
- transmission constraints and pumped storage matter,
- and a downstream hedging model has to care about path shape rather than only point forecasts.

That is why the project moved away from a single-generator story and toward a sequence of streams. Each stream exists because the previous one revealed a structural miss.

---

## The project frame that is now fixed

The current authoritative framing is a three-layer structural system:

```text
Layer 1 — Regime process      r_t ~ P(r_t | r_{t-1}, X_t)
Layer 2 — Event risk          λ_t = f(history, X_t, r_t)
Layer 3 — Price dynamics      P_t = μ(X_t, r_t) + D_t
                              dD_t = −κ_r D_t dt + σ_r dW_t + J dN_t
```

The important point is that **Layers 1 and 2 are now shared fixtures** and the streams differ mainly in **Layer 3**.

That matters because it turns the project into a real model comparison instead of a moving-target exercise. Streams 3, 4, and 5 are not allowed to quietly change the whole data-generating story underneath themselves.

### Shared fixture contract

The current downstream contract is:

- `segments_s2.parquet`
- `hazard_features.parquet`
- `hazard_features_seg.parquet`
- `feature_analysis_s2.json`
- blocked chronological split with embargo
- a **19-dimensional conditioning vector** made up of:
  - **7 econometric scalars**
  - **9 refined regime one-hot features**
  - **3 hazard segment features**

This fixed contract is one of the most important achievements of the project so far. It means the later generators are being judged on the same market information rather than on a convenient private version of the truth.

---

## Data

`master.parquet` covers 2017–2026 and is built from free UK market data sources.

| Source | What it provides |
|---|---|
| Elexon Insights API | Half-hourly GB system price, imbalance, generation by fuel including wind |
| Carbon Intensity API | Regional generation mix and carbon intensity |
| National Grid ESO / NESO | Wind and solar forecasts, demand, network constraints |
| NESO Historic Demand Data | National demand, embedded wind/solar generation, signed pumped-storage behaviour |
| LCCC CfD data | Strike prices, generation volumes and difference payments |

One of the first genuinely useful findings in the project came from a silent data bug: the Elexon pumped-storage column was not the signed storage behaviour I thought it was. Fixing that by joining the NESO demand feed added proper signed pumped storage, embedded renewables, and a more realistic full wind signal. That change mattered both econometrically and generatively.

The big lesson there was simple and worth keeping in the README: **never trust a market-data column name without checking the empirical distribution.**

---

## What has been built so far

## Stream 1 — Intuition-driven CVAE (complete, frozen baseline)

The first version of the project was built from a reasonable but ultimately naive idea: represent short path segments with signatures, condition a CVAE on a block of market features, generate synthetic segments, and validate them with stylised-fact checks plus signature diagnostics.

This stream was not a waste. It did two important things:

1. it proved the repo could become a full pipeline rather than just a notebook graveyard;
2. it revealed the central failure that shaped the rest of the project.

That failure was **inter-segment volatility clustering**. The CVAE could pass enough marginal and low-order checks to be useful as a baseline, but it could not reproduce the temporal persistence in volatility. That turned out not to be a tuning issue. It was architectural. Independent latent draws per segment sever the temporal chain the market actually has.

That is a theme that now runs through the whole project: when a model fails in a way that lines up with its mathematical design, I treat that as a structural lesson, not as an invitation to do endless local tuning.

## Stream 2 — Econometrically conditioned CVAE (complete, frozen)

Stream 2 was the repair job on Stream 1.

Instead of choosing conditioning variables by intuition, the pipeline was rebuilt around an econometric investigation of the data. The conditioning vector became something earned rather than assumed.

### What Stream 2 achieved

- reduced the candidate feature set to a lean conditioning bundle;
- enforced explicit no-look-ahead rules;
- moved away from price-defined regime labels;
- identified a stable regime backbone;
- introduced a defensible segmenting and split contract;
- improved the CVAE materially while also making its remaining weakness clearer.

The most important conceptual upgrade was that the market was treated as a **system** rather than as a price series with accessories.

### Econometric findings that mattered

Among the useful conclusions:

- lagged wind generation does matter for later price behaviour;
- mean reversion is present on roughly a 30-hour horizon;
- four core market regimes are visible in the econometric backbone;
- volatility memory is real and strong enough that it should be treated as load-bearing, not decorative.

The winning downstream label scheme became **`regime_refined`**, and the saved project features established the backbone later streams now reuse.

## Layer 2 — Shipped spike/negative hazard process (complete)

This is now a settled part of the repo and one of the most important updates relative to older README versions.

The project **does not use Hawkes as the shipped Layer 2 event engine**.

A classical 2D Hawkes attempt was explored Hawkes route was explored but the package/library path never became reliable enough to use. The project now uses **two discrete-time LogisticHazard models on the native half-hour grid**:

- one for `spike_start`
- one for `neg_start`

Artifacts:

- `hazard_features.parquet`
- `hazard_features_seg.parquet`
- `hazard_fit.json`

This was a good decision for the repo. It preserved the event-risk role that later generators need, avoided forcing the whole project through an awkward library dependency, and gave the downstream streams a cleaner fixed contract.

## Stream 3 — SigCWGAN (complete enough to freeze as current candidate)

Stream 3 matters because it is where the project stopped being “CVAE plus more conditioning” and became a genuinely different model class.

The SigCWGAN route was attractive for two reasons:

- it restored an autoregressive path structure that the CVAE never had,
- and it moved the training objective closer to path-space geometry through signatures.

### What Stream 3 achieved

- built the `_s3` preparation pipeline and conditioning bundle cleanly;
- produced the expected `logsigs_s3.parquet`, `norm_stats_s3.json`, and `conditioning_stats_s3.json` artifacts;
- upgraded notebook 05 into a reusable multistream validator;
- exposed a bad saved split label bug and forced validation to become defensive;
- materially beat the older CVAE baseline in substance.

A representative held-out validation picture for the current frozen SigCWGAN candidate is roughly:

- `MMD ≈ 0.0044`
- `std_gap ≈ 0.488`
- `mean_gap ≈ 0.349`
- `return ACF gap ≈ 0.097`
- event mismatch relatively small

The exact numbers matter less than the pattern: **Stream 3 is clearly stronger than the CVAE, but its remaining weakness is still dispersion and Lévy-type geometry rather than basic convergence or conditioning failure.**

That is a useful place for the project to be. It means the remaining work is more specific and more honest.

## Stream 4 — Neural SDE (research finding already obtained)

Stream 4 was the natural next move because it generates **raw paths directly**, which makes it much more useful for downstream deep hedging than a logsignature-space generator.

The important thing Stream 4 taught is that a model can be mathematically cleaner and still be structurally wrong for the market object.

### What Stream 4 appears to have achieved

- stable training;
- clear use of conditioning;
- direct raw-path generation;
- a more natural bridge into notebook 07d and later hedging work.

### What Stream 4 appears to have taught

The main weakness is not instability. It is **smoothness**.

The Neural SDE seems able to learn a reasonable continuous-time conditional diffusion, but GB electricity prices do not behave like a purely Brownian continuous path when the important market episodes happen. The generated paths are still too smooth in local variance and too compressed in Lévy-type path geometry.

That is a structural finding, not a small hyperparameter complaint. If the market’s important episodes are concentrated jump-like spike-revert events, then a continuous Brownian Neural SDE is trying to explain the wrong object.

---

## What the project has achieved so far

A README should not only list notebooks. It should say what the project has actually earned.

### 1. A stable structural frame

The project is no longer “a few generative models tried on electricity data”. It now has a stable three-layer interpretation with shared fixtures and an honest cross-stream comparison plan.

### 2. A defensible conditioning contract

The project now has a fixed 19-dimensional downstream conditioning bundle rather than a changing pile of intuitively plausible inputs.

### 3. A working event layer

The shipped hazard process is done and reusable. That is a real project asset, not just a side result.

### 4. A reusable multistream validator

Notebook 05 has become the authoritative judge across streams. That is a major improvement in research hygiene because it reduces the risk of selecting a winner based on a training proxy that cannot even see the failure mode.

### 5. A stronger baseline than the original CVAE story

SigCWGAN is a real improvement on the older CVAE path. The project is not stuck at the point of proving only that the first model was wrong.

### 6. A clearer research direction

The failures are now much sharper. The recurring miss is not “these models do not train”. It is “these models struggle to reproduce concentrated volatility episodes, dispersion, and higher-order path geometry”.

That kind of clarity is progress.

---

## What the project has learned

This section matters because the project is now as much about diagnosed structure as it is about finished code.

### 1. The CVAE was the wrong final model class

It was a good first build and a useful baseline, but it never had a clean route to the inter-segment temporal structure the market needs.

### 2. Better conditioning helps, but it does not solve an architectural mismatch

Stream 2 was much better than Stream 1, but the ACF failure stayed. That is exactly what should happen when the conditioning gets better but the generator class still severs temporal dependence.

### 3. Stream 3 proved the project should prefer structural fixes over endless local tuning

SigCWGAN improved meaningfully, but it also showed that a model can get the easier parts right while still missing variance and path geometry. That is why I no longer treat “another tuning round” as the default answer.

### 4. Stream 4 taught that raw-path generation is valuable but continuity is a real constraint

The Neural SDE is useful because it produces directly hedgeable paths and fits the downstream story better. But if the market’s key episodes are jumpy, a continuous diffusion is still structurally limited.

### 5. The recurring failure looks structural, not cosmetic

Across Streams 3 and 4 the same broad weakness remains:

- local variance is too compressed,
- tails are too smooth,
- Lévy-type geometry is not strong enough,
- and spike-revert structure is not being captured as directly as it should be.

That is exactly why Stream 5 exists, and why Project B now makes sense.

---

## Stream 5 — what it is and what I hope it achieves

**Stream 5 is the next active Layer 3 build.**

It is not a full rebuild and it is not a plain Gaussian diffusion. The current plan is a **jump-aware conditional diffusion on 72-step return segments**.

The idea is to let the denoiser model a return path as a combination of:

- a background continuous component,
- a jump gate or jump probability process,
- and a jump-size head.

In other words, Stream 5 is trying to make jumps explicit without yet forcing the entire repo through a full Project B redesign.

### What Stream 5 is trying to prove

The research question is simple:

> Can a jump-aware diffusion recover spike tails, local variance, and Lévy-type path roughness better than the Brownian Neural SDE, while keeping the rest of the project contract fixed?

### What I hope Stream 5 achieves

Ideally Stream 5 should:

- improve **variance matching** relative to Stream 4,
- improve **tail behaviour** and spike realism,
- improve **lead-lag / Lévy-type path geometry**,
- remain competitive on the broader distributional checks,
- and give the project a cleaner bridge from the current architecture to a fuller jump model later.

Even a partial success would be valuable. If Stream 5 clearly helps with tails and variance but not enough with geometry, that still tells me explicit jumps matter and that a later full jump-diffusion is justified.

---

## Project B — the informed rebuild

Project B is my long term planned (aspirational) rebuild that starts from what the earlier streams have actually taught rather than from what I originally hoped would work.

The core idea is straightforward:

**jumps should be first-class model objects, not a nuisance term left for a smooth generator to approximate.**

The planned structure is:

- **Layer 1:** regime process
- **Layer 2:** neural jump-diffusion temporal point process for event timing
- **Layer 3:** neural Merton-style jump diffusion for price dynamics
- optional **Lévy-aware auxiliary module** only after the jump mechanism itself is credible

This is not there because “bigger architecture = better”. It is there because the existing streams increasingly suggest that the hard part of the market is the concentrated jumpy behaviour, not a slightly more complicated continuous diffusion.

Project B therefore exists as the architecture most informed by the current evidence:

- CVAE was too disconnected across segments,
- SigCWGAN was useful but still limited in geometry,
- the Brownian Neural SDE was cleaner but too smooth,
- so the next serious rebuild should model event timing and jump response directly.

---

## Downstream use: gas storage and deep hedging

The generative side of the project is not the whole point. The downstream notebooks are why path realism matters.

### 07a — Gas storage

This notebook uses simulated price behaviour inside a stochastic control setup. That is one of the reasons the project cares about realistic path structure instead of only good one-step predictive metrics.

### 07d — Deep hedging

This notebook currently has a known limitation: the existing baseline uses Schwartz OU paths, which are useful but too simple. They are continuous, Gaussian, and wind-blind compared with the market behaviour the rest of the project is studying.

The whole reason later streams matter is that they should eventually provide **more realistic training paths for the hedging agent**.

That is why raw-path generation matters so much. A model can look elegant in logsignature space and still be awkward for downstream hedging.

---

## Evaluation philosophy

The repo now has a much better decision rule than it did earlier on.

The winner is **not** chosen by whichever internal training loss looks prettiest.

The winner is chosen by notebook 05 style **path-law validation**, including:

- distributional fidelity,
- temporal dependence,
- event realism,
- variance matching,
- tail behaviour,
- and Lévy-type geometry.

That is deliberate. I do not want to pick a generator using a score that is blind to the very failure modes the project cares about.

---

## Why the repo keeps the failed paths

One thing I want the repo to communicate clearly is that failed streams are still worth keeping.

Stream 1 stays because it is the honest baseline.

Stream 2 stays because it shows what better econometrics can and cannot fix.

Stream 3 stays because it is the first strong non-CVAE candidate and because it forced the validation layer to mature.

Stream 4 stays because it taught a real structural lesson about smoothness versus jumpy market geometry.

That sequence is the project. The learning is not separate from the code history.

---

## Repository direction from here

The current order of work is:

1. keep Stream 3 frozen as the live SigCWGAN candidate;
2. keep Stream 4 as a documented structural finding rather than over-tune it;
3. build and judge Stream 5 under the same shared contract;
4. run the cross-stream comparison honestly;
5. then decide whether the right future move is a Stream 5 winner, a later raw-path SigCWGAN revisit, or the fuller Project B jump architecture.

That is a much cleaner position than the project was in originally. The failures are narrower, the comparisons are fairer, and the next step is motivated by what the data and validators actually said.


---
## Things I had hoped to do

The main research bottleneck is still the generator itself: variance matching, jump behaviour, and higher-order path geometry are not yet fully settled across Streams 3–5. Until that is resolved, there is limited value in pushing further downstream notebooks too hard. In practical terms, there is no point polishing portfolio, filtering, or market-making extensions on top of a simulator that is still being improved. I had/have plans for the following as I am interested in them:

- 07b_sql_analytics
  This notebook is built and still useful. Its role is mainly to query, inspect, and summarise generated scenarios once they have been written to the database. That makes it good supporting infrastructure, but it is not where the current modelling risk sits. For now it stays in maintenance mode while the main effort stays on the core generator comparison.

- 07c_portfolio
  This notebook is on hold until the cross-stream comparison is complete and a stronger generator has been chosen. Portfolio CVaR analysis is downstream of the simulation problem. The intention is to reopen this once the comparison has produced a clearer winner.

- 07e_kernel_filter
  This notebook remains on hold as a mathematically interesting extension rather than a near-term priority. Signature-kernel filtering and state-estimation ideas are still very much part of the broader scope of the project, which is realistic path generation for valuation and deep hedging. For that reason it is being deferred - I am still really interesting in learning this tool though.

- 07f_avellaneda
  This notebook is also on hold. It sits furthest from the current core pipeline and depends even more heavily on having a simulator that can already be trusted. The market-making angle is still interesting, and it does connect naturally to the wider themes behind the project and my prior academic mathematical interests, but here it is clearly a later-stage add-on rather than part of the mainline build. It does 

--- 
## Notes on style and process

A few things are deliberate throughout the notebooks and write-up.

The notebooks are print-heavy. That is intentional. Part of the work here has been learning what to inspect, what to plot, and which diagnostics actually tell you something.
Diagnosed failures are kept in view. When a model class fails for structural reasons, that is part of the project result.
The narrative is honest about what is finished and what is not. Some notebooks are stable, some are exploratory, and some are parked on purpose. I would rather flag that openly than make the repo look tidier than it really is.

The project is ambition-bounded by being one person on finite compute. Colab Pro, A100 sessions, and practical library constraints have shaped some decisions. 

---

## Bottom line

This project started as a synthetic-data pipeline for GB electricity prices.

It has now become something more useful than that: a measured comparison between model classes on a genuinely awkward market object, with a shared econometric and event-risk backbone, a reusable path-law validator, and a much clearer view of what the market seems to require.

The main lesson so far is not “deep generative models are hard”. It is more specific:

> for GB electricity, the recurring miss looks structural. The market’s hard part is concentrated volatility, event clustering, and jumpy path geometry. That means later success is more likely to come from explicitly modelling jumps and event timing than from endlessly retuning smooth generators.

That is why Stream 5 is jump-aware, and why Project B exists.
