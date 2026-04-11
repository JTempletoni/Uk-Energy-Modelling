# GB Electricity Synthetic Data Pipeline

A personal research project building an end-to-end pipeline for generating synthetic GB electricity price paths, with downstream applications in gas storage stochastic control and deep hedging. Built on real market data from public APIs, path signatures, conditional generative models (CVAE, SigCWGAN, neural SDE), and deep hedging networks.

This is a personal learning project, not a production system. The code is (I think) honest about what works, what failed, and why. I have included a diagnosis of specific failure modes in the first generative model and have been working on a redesigned empirical track to fix them. The first pipeline and its notebooks will remain unchanged and they taught me a great deal in a short period. Upon reflection I built it almost all from intuition and using papers/knowledge I had accumulated studying other things and developed an interest in. My original plan was overly ambitious and totally intuition driven, so I learnt as I went and somethings were cut or modified. My ambition is also limited by the limits of Colab pro.

I have now tried to engage with the current literature on the UK and European power markets. A lot of my original thinking came from applying for a data job at a scottish energy firm, having a look at the energy data available and living in the north of scotland - generation is built faster than infrastructure so I know there are bottlenecks. 


The redesigned pipeline will begin by studying the "system" that is the energy market first, discovering its signals, features, regimes, etc independent of price data. I also intend to be much more statistically rigorous - I made silly  mistakes sampling. The proper way to begin is to identify the regimes/signals independently of price data. My current regime labels are threshold rules based on prices. I need to study the dynamics of the system first. Then I can identify exegonous information among other things that can be used as conditional data - to do that I think I will make use of GARCH, ARIMA etc.

---

## Background and motivation

This project connects multiple stages of my prior work:

- **Undergrad dissertation** — Markov chain Monte Carlo agent games of wealth inequality (Yardsale model): bilateral exchange, ergodic theory, stationary distributions of interacting particle systems
- **Masters dissertation** — Neural network option pricing in the Hull-White volatility model: FFT pricing scheme, KDE-based synthetic training data, universal approximation theorem
- **This project** — both threads meet. Markov regime switching for non-stationarity. Path signatures as the rigorous feature map for sequential data (grounded in the same UAT as the masters dissertation). VAE/GAN generative models. Applied to real UK energy market data with downstream applications in stochastic control and derivatives hedging.

The choice of GB electricity is deliberate: the market has free public APIs (Elexon Insights, Carbon Intensity), is structurally interesting (negative prices, wind cannibalisation, CfD policy), and the data is genuinely hard to model — which makes it a better test of the methods than equity indices.

---

## Data

`master.parquet` covers 2017–2026, 157,776 half-hourly observations, 86 columns. All sources are free.

| Source | What it provides |
|---|---|
| [Elexon Insights API](https://developer.data.elexon.co.uk) | Half-hourly GB system price, imbalance, generation by fuel including wind |
| [Carbon Intensity API](https://api.carbonintensity.org.uk) | Regional generation mix, half-hourly from 2018 |
| [National Grid ESO](https://data.nationalgrideso.com) | Wind/solar forecasts, demand, B6 boundary constraints |
| [NESO Historic Demand Data](https://www.neso.energy/data-portal) | National demand, embedded wind/solar generation, **signed pumped storage** (pumping vs generating) |
| [LCCC CfD data](https://lowcarboncontracts.uk) | Strike prices, CfD generation volumes, difference payments (quarterly) |

`total_wind_mw` from Elexon (full 2017–2026 coverage) is preferred over `pct_wind` from Carbon Intensity (NaN gaps before late 2018 create artefacts in PCA space).

### The demand-data fix (a small story about a silent data bug)

The Elexon FUELHH `ps` column reports pumped-storage **generation only**: it is non-negative, with min=0 and 0% of half-hours showing pumping. I assumed for an embarrassingly long time that it was signed. The transition-predictor diagnostic in notebook 02 caught it: `ps_prev_sign` had AUROC = 0.5000 (i.e. constant feature, no information). That's what made me actually look at the column rather than trust the column name.

The fix was to join the NESO Historic Demand Data feed (`PUMP_STORAGE_PUMPING`, `EMBEDDED_WIND_GENERATION`, `EMBEDDED_SOLAR_GENERATION`, `ND` national demand) onto `master.parquet`. This adds:

- `ps_net_mw` — signed pumped storage (positive = generating, negative = pumping). Now actually carries information about whether the previous segment was cheap enough to charge the reservoirs.
- `embedded_wind_mw`, `embedded_solar_mw` — distribution-connected wind and solar that don't show up in the transmission-level Elexon feed. Embedded wind alone is now ~5–7 GW in GB; ignoring it understates the renewable share by a meaningful margin.
- `total_wind_full_mw` = transmission wind + embedded wind. This is now the canonical wind column from stream 2 onwards.
- `national_demand_mw` — proper demand, not the residual implied by the fuel mix.

The lesson, for the writeup as much as for me: do not trust column names without checking the empirical distribution. A column called `ps` that is identically non-negative is not what you think it is. A regime-conditioning vector built on it will silently throw away one of its features.

---

## What has been built — original track (stream 1)

The first pass through the pipeline was intuition-driven: features chosen on domain reasoning, architecture adapted from Bühler et al. (2020), conditioned on what seemed plausible. It worked well enough to generate synthetic paths that pass the primary statistical tests. It also produced clear, diagnosable failures that motivate the second track. That is an acceptable outcome for me on a first attempt on a small and structurally unusual dataset. Some of the errors were hard to identify as I created them jumping between notebooks to alter things and didn't realise the downstream consequences.

| Notebook | What it does | Status |
|---|---|---|
| `01_data_acquisition` | Elexon Insights + Carbon Intensity + NESO Historic Demand → parquet | Done |
| `02_eda_stylized_facts` | Regime labelling, price/wind distributions, autocorrelation | Done |
| `03_signature_features` | 4D lead-lag path, depth-4 log-signatures, segment generation | Done |
| `04a_cvae_generator` | CVAE trained on 3,285 daily segments, ELBO + sig-MMD loss | Done |
| `05_validation` | Sig-MMD, KS, Lévy area, large move rate, volatility clustering ACF | Done |
| `06_scenario_database` | 10k scenarios → DuckDB | Done |
| `07a_stoch_control` | 30-day gas storage LSMC valuation, regime-stratified bootstrap | Done |
| `07b_sql_analytics` | DuckDB scenario analytics | Done |
| `07c_portfolio` | CVaR portfolio optimisation | planned |
| `07d_deep_hedging` | Deep hedging network on Schwartz OU paths | Done (to be revised) |
| `07e_kernel_filter` | Sig-kernel GP regression, Kalman filter |  planned |
| `07f_avellaneda` | Avellaneda-Stoikov market making | planned |

### Validation results (notebook 05)

| Test | Result |
|---|---|
| Sig-MMD | **PASS** p = 0.84, MMD stat = −0.000631 |
| KS marginals | 18 / 31 informative dimensions pass (58%) |
| std of net price move | PASS |
| Lévy area (∝ realised variance) | PASS |
| Large move rate | PASS |
| Skewness | FAIL |
| Kurtosis | FAIL |
| **Overall** | **4 / 6 stylized checks pass** |

A note on the KS dimension count: the depth-4 log-signature of a 4D lead-lag path has 90 dimensions, but 59 are structural zeros — identically zero by the antisymmetry properties of the lead-lag construction, carrying no distributional information. Notebook 04a drops them before training. The KS test therefore runs on the 31 informative dimensions. This is the correct behaviour; it is documented here because the silent reduction from 90 to 31 is not obvious from the notebook output alone.The reason ther are lots of print statements is so I can determine what is going on accurately - learning what to print and when is also something I am learning along the way.

The Sig-MMD result (p = 0.84) is the primary result: the joint distribution of generated paths is statistically indistinguishable from real paths under the signature kernel. The KS failures and heavy-tail misses are expected — VAEs compress variance because the KL term pushes the posterior toward the uninformative prior, smoothing away extreme events. These are known VAE limitations, not bugs.

**My key finding — volatility clustering:**

The ACF of |Lévy area| (∝ |realised variance|) shows real data with clear positive autocorrelation persisting out to 20 weeks. The generated series is near zero at all lags. This is not a tuning failure — it is architectural. A VAE samples z ~ N(0,I) independently per segment. The ELBO has no term penalising inter-segment temporal correlations. Volatility clustering is an inter-segment property. No amount of feature engineering fixes this. It requires a different model class.

**A note on circular modelling and model collapse:**

Training on model-generated outputs, or augmenting with synthetic data without validation, causes progressive degradation of tail distributions and diversity (Shumailov et al. 2023). The KDE augmentation in 04a was disabled (N_AUGMENT=0) deliberately. Any augmentation in the enhanced track is validated against real data before inclusion.

---

## Markov structure — four roles

The project uses Markov and Bayesian methods in four distinct roles. These were planned from the outset; they are implemented to varying degrees.

**1. Regime switching HMM — partially implemented.** The four regimes (calm / volatile / spike / negative) are currently identified by rule-based thresholds: an adaptive 5-MAD threshold with a hard £300/MWh floor for spikes, and a −£20/MWh floor for negative prices. Segment regime is assigned by a priority rule (any spike half-hour → spike segment; any negative → negative; any volatile → volatile; else calm) rather than majority vote, which always returns calm because 88.6% of half-hours are calm. The empirical Markov transition matrix between consecutive segment regimes is computed in notebook 05 as a validation diagnostic. What is not yet implemented is replacing the rule-based labels with posterior probabilities from the forward algorithm — so the CVAE conditioning is probabilistic rather than hard-labelled. This is a planned improvement for the enhanced track.

**2. MCMC calibration — not yet implemented.** The intended role is Bayesian posterior estimation over model parameters via HMC/NUTS (PyMC or numpyro), propagating parameter uncertainty into the generated path distributions. This connects directly to the undergrad dissertation (MCMC as the inference engine) and to the masters dissertation (uncertainty quantification in option pricing). Currently all model parameters are point estimates.

**3. Sequential Monte Carlo — not yet implemented.** A particle filter for online hidden state estimation as new data arrives, with MCMC moves at the resample step to prevent particle degeneracy. This would allow regime conditioning to update in real time rather than being fixed at training — relevant for deployment rather than backtesting. (Stochastic filtering always intrigued me and I am determined to come to grips with it.)

**4. Yardsale → limit order book — not yet implemented.** The Yardsale bilateral exchange Markov chain from the undergrad dissertation has a Boltzmann-Gibbs stationary distribution over wealth. The mapping to queue-level dynamics in a limit order book connects to the Avellaneda-Stoikov framework in the planned notebook 07f. This is the most speculative of the four roles and the furthest from the core pipeline. At the time the Fokker-Planck derivation and Boltzmann modelling was beyond me (I was warned off wasting time on something beyond the scope of the project by my supervisor), but it continues to annoy me that I haven't tackled it and that is why this notebook will exist.

---

## The four streams — why each one exists

The original pipeline is one CVAE. The full project is four generative streams, each fixing a specific failure of the previous one. They share `master.parquet`, `02_eda_stylized_facts` (re-run with the proper econometric investigation), and the same downstream notebooks (06, 07a, 07b, 07d). Only the generator changes.

The point of doing this as four streams rather than picking one and committing isn't completionism — it's that I want concrete, measured statements about what each model class buys you and what it costs, rather than asserting one is best on theoretical grounds. Each stream addresses a known weakness of its predecessor, and the comparison notebook `08_stream_comparison` is the final deliverable that ties them together.

```
stream 1 — CVAE (intuition-driven)               ← done, frozen baseline
stream 2 — CVAE (econometrically conditioned)    ← done, frozen
stream 3 — SigCWGAN (autoregressive)             ← in progress
stream 4 — Neural SDE (continuous time)          ← planned
```

None of streams 2–4 has been pushed to GitHub yet. The original track remains untouched and runnable.

---

### Stream 2 — CVAE done properly

**The diagnosis.** Stream 1 conditioned on whatever scalars seemed sensible to me at the time. I never tested whether they actually predicted anything, I never checked whether they were endogenous (jointly determined with price by the same market clearing), and I labelled regimes from the price itself — which is the modelling equivalent of marking your own homework. The 31-out-of-31 KS failures and the kurtosis miss are the predictable consequences.

**The fix.** Build the conditioning vector empirically. Notebook 02 is rewritten as a structured econometric investigation rather than descriptive EDA, and segment generation moves into it so the chain from raw data to conditioning vector is visible in one place. The output is `feature_analysis_s2.json` — a JSON spec that drives the architecture of stream 2's CVAE rather than the architecture driving the feature choices.

The screen for any candidate variable is four conditions, all of which must hold:

1. expressible as a scalar at segment level
2. no look-ahead (i.e. exogenous to the segment we are predicting)
3. not redundant with another candidate (correlation < 0.7 or it gets dropped)
4. predictive of `log(segment RV)` in regression at the 5% level

The balancing-mechanism columns (accepted bid/offer volumes, net imbalance volume, system buy/sell prices) all fail condition 2 — they are determined simultaneously with price in the same auction. Using them as conditioning variables is look-ahead bias and circular by construction. Stream 1 used some of them. Stream 2 doesn't.

**Wind, properly understood.** I had been thinking of `total_wind_full_mw` as a proxy for wind speed and considering whether to go and download ERA5 reanalysis. I now think this is wrong-headed. The grid operator has already aggregated output across hundreds of geographically distributed wind farms spanning the full GB landmass and expressed the result in MW of actual generation — a capacity-weighted, economically normalised measure that encodes the nonlinear relationship between wind speed and power output via the fleet's power curves. It is the same meteorological information as gridded reanalysis, in units that are directly relevant to price formation, accessible via one API call. There is no need to go digging in weather data.

Two layers of meteorological structure are extractable from this signal. The mean dynamics — frontal passage timescales (1–2.5 days), synoptic cycles (3–7 days), seasonal patterns — can be captured by an ARMA(p,q) on wind output levels. This is why the ACF of real |Lévy area| shows ~2–3 week periodicity: it is the autocorrelation of meteorological regimes encoded in the generation signal. The variance dynamics — how erratically wind changes — correspond to unstable weather fronts and are the meteorological analogue of volatility clustering, which is why a GARCH-X variant with wind in the variance equation is the right thing to fit.

**The chosen volatility model.** A grid search across GARCH(1,1), EGARCH, GJR-GARCH, with normal / Student-t / skew-t innovations, decided by AIC/BIC and an ARCH-LM residual test, with sanity filtering for physically pathological fits (skew-t variants kept blowing up). The winner is **GJR-GARCH(1,1)-t**, persistence ≈ 0.98, which is loud-and-clear evidence of long-memory volatility. The Hurst exponent on `|returns|` is ≈ 0.7–0.8, which agrees with the GARCH persistence. Both numbers say the same thing: volatility clustering is strong and long-range.

**The conditioning vector that came out of all this** (11 dims, intentionally lean):

```
nuclear_prev_mean        — slow-changing, outage proxy
wind_prev_mean           — merit-order driver (uses total_wind_full_mw)
RV_prev                  — volatility memory
ccgt_fraction_prev_mean  — marginal fuel indicator
ocgt_active_prev_mean    — peaker activation (binary-like)
wind_var_prev_mean       — σ²(wind) from the EGARCH wind fit; weather-front proxy
har_rv_prev_mean         — HAR-RV multi-horizon vol summary
regime_onehot            — GMM 4-state on supply-side features (4 dims)
```

Crucially, the regime label is no longer a function of price. A GMM is fit on the supply-side state vector (wind, nuclear, pumped storage, carbon intensity, hydro), giving four interpretable states with sensible sojourn times — S0 (38%), S1 (26%), S2 (15%, scarcity, ~55h dwell), S3 (21%). This is a physical-state regime, not a price-threshold regime, and the BIC elbow is unambiguously at 4.

**Stream 2 results.** Best epoch 979, val loss 0.8024 on the rerun. Posterior collapse falls from 75% (stream 1) to **38%** — a much sharper improvement than I expected from just tightening the conditioning vector. The lean, empirically-grounded 11-dim conditioning frees up latent dimensions that were previously redundant in stream 1's bloated 38-dim version, and 62% of the latent capacity is now actually being used. Train and val curves track each other cleanly with a small positive val–train gap settling around 0.03 (no overfitting), KL plateaus at ≈ 0.39 well above the `min_kl=0.3` floor, and the per-regime val reconstruction is well-separated and stable across calm / volatile / spike / negative — i.e. the regime conditioning is doing real work rather than collapsing into one mean. Marginal Wasserstein-1 is reasonable for the low-order log-signature dimensions. Stylised facts: 3–4 of 5 pass.

**The ACF still fails.** Synthetic |ls_7| ACF is ~0 at lag 1; real data is ≈ 0.55. This is the central failure that stream 3 has to fix, and it is exactly the failure I predicted at the end of stream 1: it is architectural, not a tuning problem. A better-conditioned CVAE is still a CVAE. The ELBO objective samples `z ~ N(0, I)` independently per segment, which severs the inter-segment temporal chain. No conditioning vector engineering can put that chain back. You need a different model.

---

### Stream 3 — SigCWGAN

**Why this one.** SigCWGAN (Ni et al. 2021) is the smallest sensible architectural change that fixes the inter-segment ACF problem. Stream 3 attacks the failure on two fronts simultaneously: an autoregressive generator (so the output of segment $t$ becomes part of the input for segment $t+1$, restoring the temporal chain that the CVAE's per-segment $z \sim \mathcal{N}(0, I)$ severs), and a Hawkes-process conditioning signal that explicitly tells the generator how recently spike and negative-price events have been firing. The loss is computed on path signatures, which encode temporal cross-terms by construction, so unlike the ELBO it directly sees inter-segment structure rather than just marginal-distribution mismatch.

**How it actually works.** The model is, despite the "GAN" in the name, not adversarial in the usual sense. There is no discriminator network. The training procedure is:

1. For a batch of real (past, future) segment pairs, compute the path signature of each past window and each future window: $S(\text{past})$ and $S(\text{future})$.
2. Fit a linear regression $S(\text{future}) \approx W \cdot S(\text{past})$ on the batch. The fitted $W$ is a closed-form, batch-local estimator of the conditional expected signature $\mathbb{E}[S(X^{\text{fut}}) \mid X^{\text{past}}]$. (Ridge-regularised, $\lambda \approx 10^{-4}$, to deal with rank deficiency.)
3. The generator $G$ is an AR-FNN — a feedforward network that takes the past signatures, a noise $z \sim \mathcal{N}(0, I_{d_z})$ and the conditioning vector $c$, and outputs the next path increment.
4. The loss is the squared distance, in signature space, between the generator's signature and the closed-form regression target:

$$
\mathcal{L}_{\text{Sig-W1}} \;=\; \big\| S\big(G(\text{past}, z, c)\big) \;-\; W\,S(\text{past}) \big\|_2^2
$$

There is no min-max game. The "target" is a closed-form linear projection rather than an adversary that fights back, which means SigCWGAN doesn't suffer from the standard GAN failure modes (discriminator overpowering, mode collapse from adversarial dynamics). For a small, structurally awkward dataset like 6,530 GB electricity segments this matters a lot — I would not trust a vanilla WGAN-GP to converge sensibly here.

**The signature bit.** The Sig-Wasserstein-1 distance between two path-distributions $\mu, \nu$ is

$$
\text{Sig-W}_1(\mu, \nu) \;=\; \big\| \mathbb{E}_{X \sim \mu}[S_M(X)] - \mathbb{E}_{Y \sim \nu}[S_M(Y)] \big\|_2
$$

where $S_M$ is the signature truncated at depth $M$. The reason this is a meaningful distance on path-space is the same reason path signatures appeared in stream 1 in the first place — Chevyrev & Oberhauser show the expected signature characterises the law of a stochastic process. So matching expected signatures is matching distributions on path-space, and any continuous functional on path-space (option payoff, hedging error, you name it) is well-approximated by a linear functional of the signature. This is the same UAT-style result as my masters dissertation, applied to sequential rather than scalar inputs.

**Hawkes intensities as a first-class conditioning input.** The autoregressive feedback alone is not the only thing stream 3 brings — the conditioning vector also gains two columns that explicitly encode self-exciting event clustering. Spike events and negative-price events in GB electricity are not Poisson: a spike makes another spike in the next few hours much more likely (gas price moves, sustained low wind, nuclear outages are all multi-hour phenomena), and negative-price episodes cluster overnight when wind is high and demand is low. The natural model for this is a multivariate Hawkes process with conditional intensity

$$
\lambda_i(t) \;=\; \mu_i \;+\; \sum_{j} \int_{-\infty}^{t} \phi_{ij}(t - s)\, dN_j(s)
$$

and exponential kernels $\phi_{ij}(u) = \alpha_{ij}\, e^{-\beta_{ij} u}$. I fit a 2D Hawkes (spike-onset, negative-onset) on the full half-hourly history with `tick.hawkes.HawkesExpKern`, deduplicating multi-period episodes to single events at the leading edge — a 6-hour spike is one event, not twelve, otherwise the branching ratio explodes towards 1 and the fit is meaningless. Diagnostics: branching ratio $n^*$ = spectral radius of the adjacency matrix (must be < 1 for stability), log-likelihood, and the time-rescaled residuals' KS statistic against $\text{Exp}(1)$. The fitted intensities $\lambda_{\text{spike}}(t)$ and $\lambda_{\text{neg}}(t)$ become half-hourly conditioning columns, aggregated to segment-mean as `lambda_spike_prev_mean` and `lambda_neg_prev_mean`.

This is **not** a separate prerequisite notebook — the Hawkes fit and intensity computation are folded directly into stream 3's data preparation, and `segments_s3.parquet` is built as `segments_s2.parquet` plus the two intensity columns. Stream 3's CVAE-equivalent conditioning vector is therefore 13 dims (the 11 from stream 2, plus the two Hawkes intensities). The hypothesis is that recent self-excitation tells the AR-FNN "you are in a regime where another spike is likely soon", which is exactly the inter-segment structure CVAE could not see — and pairing Hawkes conditioning with the autoregressive feedback should be a stronger fix than either on its own.

**Honesty about what I don't know yet.** I have not finished writing the SigCWGAN training loop, and I don't yet know whether it will deliver. The success criterion I've set myself is concrete: synthetic ACF for $|ls_7|$ at lag 1 must exceed 0.1 (real is ≈ 0.55, stream 2 is ≈ 0). If stream 3 hits 0.1 it is a non-trivial improvement; if it doesn't, something is wrong with the loss or the autoregressive feedback. Either outcome is informative.

---

### Stream 4 — Neural SDE

**Why this one.** The neural SDE is the mathematical closure of the project. Stream 1, stream 2 and stream 3 are all discrete-segment generators: they emit one segment at a time and you stitch them together. The deep hedging notebook 07d, on the other hand, currently trains its hedging agent on **continuous-time** Schwartz one-factor OU paths

$$
dS_t \;=\; \kappa\,(\theta - S_t)\,dt \;+\; \sigma\,dW_t
$$

with calibrated $\kappa$, $\theta$, $\sigma$. That's the simplest possible continuous-time model: linear mean-reversion, constant vol, Gaussian innovations, no wind, no spikes, no state-dependent volatility. It's a useful baseline because it has a closed-form Greek for the short put I am hedging, but a hedging agent trained on it cannot possibly learn regime-dependent or spike-aware strategies — those structures are not in the training paths.

The neural SDE generalises this. Replace the constant drift and constant diffusion with neural networks of the state and an exogenous conditioning vector:

$$
dY_t \;=\; f_\theta(t, Y_t, c)\, dt \;+\; g_\phi(t, Y_t, c)\, dW_t
$$

where $Y_t = (\text{price}_t, \text{wind}_t)^\top$ is a 2D state, $c$ is the same 13-dim conditioning vector as stream 3 (regime one-hot, wind/nuclear/RV/wind-var means, Hawkes intensities), and $f_\theta$, $g_\phi$ are tanh-MLPs with $g_\phi$ passed through softplus to keep diffusion positive. Schwartz OU is the special case where $f_\theta(t, Y, c) = \kappa(\theta - Y_1)$ and $g_\phi \equiv \sigma$. So the neural SDE is strictly more expressive than the current hedging baseline, and if it trains successfully it should reduce to something like Schwartz OU on the calm regime and extend to something more nonlinear on the spike and negative regimes.

**Training.** Use the adjoint sensitivity method from Li, Wong, Chen, Duvenaud (2020) so that backprop through `torchsde.sdeint` is memory-tractable. The loss is signature-kernel MMD,

$$
\text{MMD}^2(\mu, \nu) \;=\; \mathbb{E}_{X,X'}\big[k\big(S(X), S(X')\big)\big] \;+\; \mathbb{E}_{Y,Y'}\big[k\big(S(Y), S(Y')\big)\big] \;-\; 2\, \mathbb{E}_{X,Y}\big[k\big(S(X), S(Y)\big)\big]
$$

with an RBF kernel on truncated signatures (depth 3 — the MMD estimator gets noisy at higher depth). The drift output layer is initialised to zero so the model starts from approximately zero drift and has to *learn* mean reversion rather than presupposing it. Training uses Euler-Maruyama with $dt = 1/72$, batch size 128, gradient clipping at 1.0, normalised inputs (the network expects $[-3, 3]$, not $[0, 2000]$ MW or $[-50, 500]$ £/MWh).

**Interpretability.** This is the bit I am most curious about. After training I can plot the learned drift surface $f_\theta(t, Y, c)$ as a function of price at fixed wind and conditioning, and ask: does it look like linear mean-reversion? If yes, the neural SDE has empirically rediscovered Schwartz OU, which is itself a result. If no, what nonlinearities does it introduce — a kink at zero (negative-price boundary)? A super-linear repulsion above £200/MWh (spike-revert)? An asymmetric well? Any of these would be a finding worth writing up. Same for the diffusion: I expect $g_\phi$ to be $U$-shaped in price (high vol at extremes, low vol in calm) and increasing in `wind_var_prev_mean`, but I want to see it rather than assume it.

I can also back out an effective $\kappa$ by linearising the learned drift around the long-run mean and compare it to the calibrated Schwartz $\kappa$ from notebook 07a. If the half-hourly mean-reversion rate (annualised RV drops 8× from 30 minutes to 24 hours, which is loud) shows up in the learned drift, the neural SDE is consistent with the empirical RV signature plot. This is the kind of cross-check I wish I'd built into stream 1 from the start.

**Why this matters for hedging.** Stream 4 has a unique advantage downstream: because it produces raw paths in the original (price, wind) state space rather than log-signatures, it feeds notebook 07d **directly**. No log-sig → path detour. This is closer to how Bühler et al. (2019) originally formulated deep hedging — train the hedger on simulated paths from a flexible generative model, not on log-sigs that have to be inverted somehow. And because the generated paths reproduce the wind state explicitly, the hedger can in principle learn a wind-conditional strategy, which is exactly what Biegler-König & Oeltz (2025) argue is necessary for hedging renewable-linked exposures. The CfD reference-price hedge, in particular, is mis-specified if you ignore the wind-price correlation, because curtailment events break the wind→price relationship in a way that a wind-blind hedger cannot anticipate.

**Honesty about cost.** Neural SDEs are slow to train (the SDE solve is the inner loop of the gradient computation) and fiddly (drift / diffusion norms can blow up if the LR is wrong, and the MMD estimator is high-variance at small batch sizes). I have budgeted a notebook for this and I am explicitly not committing to a perfect result. The "minimum acceptable" outcome is that generated paths visually exhibit mean-reversion plus occasional spikes, MMD on signature features beats stream 2, and feeding them into 07d gives lower CVaR than the Schwartz OU baseline.

---

### The cross-stream comparison (notebook 08)

The four streams converge in `08_stream_comparison.py`, which does no new modelling — it just loads `validation_*.json` from each stream and renders a comparison table plus three bar charts: ACF reproduction error (the headline), Lévy area Wasserstein-1, and hedging CVaR from notebook 07d under each model's paths. The narrative I want this notebook to support is: *I tried CVAE; understood why it fails on electricity data; built two increasingly sophisticated alternatives; measured the improvement quantitatively at each step; and was honest about the failure modes of each.* That is the actual learning, and it is the reason for going through the four streams in sequence rather than committing to one architecture from the start.

---

### 07d deep hedging — known limitation and planned fix

The current 07d trains a hedging agent on Schwartz one-factor OU paths: Gaussian innovations only, constant volatility, no wind coupling. An agent trained on these paths cannot learn regime-dependent or spike-aware strategies. The write-up states this explicitly.

Once SigCWGAN paths pass validation, the plan is drift removal (Bühler, Murray, Pakkanen & Wood 2021): learning a near-martingale measure on SigCWGAN output that removes exploitable statistical drift while preserving spike clustering and wind coupling. Those paths then replace the OU simulation in 07d. Once the neural SDE is trained, its raw paths become a third candidate for the hedging training set, and the 07d notebook is re-run under each, with the resulting CVaR numbers feeding into the comparison table in notebook 08.

---

## Core mathematics

### Path signatures

For a continuous path $X : [0,T] \to \mathbb{R}^d$, the signature is the sequence of iterated integrals:

$$S(X)^{(n)} = \int_{0 < t_1 < \cdots < t_n < T} dX_{t_1} \otimes \cdots \otimes dX_{t_n}$$

Truncation error decays as $O(1/N!)$ — depth 4 is sufficient in practice. The expected signature characterises the law of a stochastic process (path-space analogue of the MGF). Any continuous functional on path space is approximated by a linear functional of the signature — the same UAT as the masters dissertation, applied to sequential data.

The 4D path uses lead-lag embedding on price so that level-2 cross-terms capture realised variance via the Lévy area (index 7), plus `total_wind_full_mw` so that cross-terms capture dynamic price-wind coupling nonlinearly. Wind appears both in the path and in the conditioning vector — different purposes, different information.

### Sig-Wasserstein loss (SigCWGAN)

$$\text{Sig-W}_1(\mu, \nu) = \|\mathbb{E}_{X \sim \mu}[S_M(X)] - \mathbb{E}_{Y \sim \nu}[S_M(Y)]\|_2$$

Replaces the adversarial discriminator with a theoretically grounded distance on path space. Directly penalises differences in expected signature including inter-segment temporal structure, rather than optimising a surrogate minimax objective that is unstable with small datasets.

### Neural SDE (stream 4)

$$dY_t = f_\theta(t, Y_t, c)\,dt + g_\phi(t, Y_t, c)\,dW_t, \qquad Y_t = (\text{price}_t, \text{wind}_t)^\top$$

with $f_\theta$, $g_\phi$ tanh-MLPs ($g_\phi$ softplus-positive), trained to minimise signature-kernel MMD between real and generated paths. Reduces to Schwartz OU when $f_\theta$ is linear in $Y_1$ and $g_\phi$ is constant — so the calibrated Schwartz parameters from 07a are the natural sanity check on the learned drift.

---

## UK market context

- **Half-hourly settlement**: GB electricity clears in 48 half-hour periods per day. The system imbalance price is the marginal cost of real-time balancing — highly volatile with high wind penetration.
- **Negative prices**: ~2.6% of half-hours. Caused by must-run generation (nuclear, subsidised wind under old ROC regime) and transmission constraints. Log returns are undefined — arithmetic returns throughout.
- **CfD mechanism**: Generators receive `strike_price − reference_price` from LCCC. The reference price going negative is the key policy exposure, addressed in 07d.
- **B6 constraint**: Scotland-England transmission boundary. When Scottish wind exceeds capacity, generators are constrained off — creating implicit locational price separation not visible in the aggregate system price.
- **Pumped storage**: Cruachan (660 MW) and Dinorwig (1.8 GW) arbitrage price spikes. Sets a practical floor on negative price duration and ceiling on spike duration — both regime properties the model needs to reproduce. The signed `ps_net_mw` from the NESO Historic Demand join is what makes this visible to the conditioning vector.
- **Embedded generation**: Distribution-connected wind and solar (~5–7 GW of embedded wind, growing solar fleet) do not appear in the Elexon transmission-level fuel mix and have to be added in from the NESO feed, otherwise the renewable share is materially understated.

---

## Key references

- Bühler, Horvath, Lyons et al. — *A Data-Driven Market Simulator for Small Data Environments* (2020)
- Ni, Szpruch, Wiese et al. — *Sig-Wasserstein GANs for Conditional Time Series Generation* (2021)
- Bühler, Murray, Pakkanen, Wood — *Deep Hedging with Market Impact* (2021) — drift removal
- Bühler, Gonon, Teichmann, Wood — *Deep Hedging* (2019)
- Kidger, Foster, Li, Lyons — *Neural SDEs as Infinite-Dimensional GANs* (2021)
- Li, Wong, Chen, Duvenaud — *Scalable Gradients for Stochastic Differential Equations* (2020) — adjoint method
- Biegler-König & Oeltz — *Deep Hedging of Green PPAs* (2025)
- Boogert & de Jong — *Gas Storage Valuation Using a Monte Carlo Method* (2008)
- Chevyrev & Oberhauser — *Signature Moments to Characterise Laws of Stochastic Processes* (2022)
- Shumailov et al. — *The Curse of Recursion: Training on Generated Data Makes Models Forget* (2023)
- Weron — *Electricity Price Forecasting: A Review* (2014)
- Ketterer — *The Impact of Wind Power Generation on the Electricity Price in Germany* (2014)
- Cacciarelli et al. — *Causal DML wind-price UK* (2025)
- Fu et al. — *A CGAN Framework for Financial Time Series Generation* (2019)
- Corsi — *A Simple Approximate Long-Memory Model of Realized Volatility* (2009) — HAR-RV
- Lyons — *Differential Equations Driven by Rough Signals* (1998)
- Avellaneda & Stoikov — *High-Frequency Trading in a Limit Order Book* (2008)
- Hawkes — *Spectra of some self-exciting and mutually exciting point processes* (1971)
