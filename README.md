# GB Electricity Synthetic Data Pipeline

A personal research project building an end-to-end pipeline for generating synthetic GB electricity price paths, with downstream applications in gas storage stochastic control and deep hedging. Built on real market data from public APIs, path signatures, conditional variational autoencoders, and deep hedging networks.

This is a learning project and CV piece, not a production system. The code is (I think) honest about what works, what failed, and why. I have included a diagnosis of specific failure modes in the first generative model and have been working on a redesigned empirical track to fix them. The first pipeline and its notebooks will remain unchanged and they taught me a great deal in a short period. Upon reflection I built it almost all from intuition and using papers/knowledge I had accumulated studying other things and developed an interest in. My original plan was overly ambitious and totally intuition driven, so I learnt as I went and somethings were cut or modified.

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
| [LCCC CfD data](https://lowcarboncontracts.uk) | Strike prices, CfD generation volumes, difference payments (quarterly) |

`total_wind_mw` from Elexon (full 2017–2026 coverage) is preferred over `pct_wind` from Carbon Intensity (NaN gaps before late 2018 create artefacts in PCA space).

---

## What has been built — original track

The first pass through the pipeline was intuition-driven: features chosen on domain reasoning, architecture adapted from Bühler et al. (2020), conditioned on what seemed plausible. It worked well enough to generate synthetic paths that pass the primary statistical tests. It also produced clear, diagnosable failures that motivate the second track. That is an acceptable outcome for me on a first attempt on a small and structurally unusual dataset. Some of the errors were hard to identify as I created them jumping between notebooks to alter things and didn't realise the downstream consequences.

| Notebook | What it does | Status |
|---|---|---|
| `01_data_acquisition` | Elexon Insights + Carbon Intensity APIs → parquet | Done |
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

## What is being built next — A more empirical and cautious attack

The original track remains untouched and runnable. The enhanced track runs in parallel with `_wind` suffixed outputs. **None of it has been pushed yet.** After completing the first pass, I stepped back to identify what the failures actually meant before writing more code. This plan is subject to change and will evolve as I learn more.

The design principle for the second track is empirical rather than intuitive: every modelling choice — which variables to condition on, what segment length to use, how to represent wind, which volatility model to use — is derived from data using classical econometric tests, not asserted or intuited on domain grounds. The output of that investigation is `feature_analysis.json`, which drives the architecture of 04b, rather than the architecture driving the feature choices.

```
02  EDA + econometrics + feature selection + segment generation
        ↓  segments.parquet, feature_analysis.json
03b wind signatures + joint KDE augmentation
        ↓  logsigs_wind.parquet, norm_stats_wind.json
04b wind-conditioned CVAE (empirically derived conditioning)
        ↓  cvae_wind_model.pt, cvae_generated_logsigs_wind.parquet
04c SigCWGAN — autoregressive generator, Sig-W1 loss
        ↓  sigcwgan.pt
05b three-pronged validation + ablation across 04a / 04b / 04c
        ↓  validation_report_wind.json
```

### Notebook 02 — what the econometric investigation covers

The new notebook 02 replaces descriptive EDA with a structured investigation whose primary output is `feature_analysis.json` — the empirically derived conditioning vector for 04b. Segment generation also moves here from notebook 03, so the reasoning from data to modelling decision is visible in one place.

**Exogenous vs endogenous.** The 86-column dataset contains balancing mechanism outcomes (accepted bid/offer volumes, adjustment data, net imbalance volume) determined simultaneously with price by the same market clearing. Using them as conditioning variables is look-ahead bias and circular by construction. Every column will be screened against four criteria: (1) expressible as a scalar at segment level, (2) no look-ahead, (3) not redundant with another candidate, (4) predictive of the log-signature in regression. Variables must pass all four.

**Pumped storage as a market signal.** Pumped storage (Cruachan 660 MW, Dinorwig 1.8 GW) pumps when prices are low or negative and generates when prices are high. The raw contemporaneous signal is endogenous. The lagged sign — what the plant did in the previous segment — is exogenous and carries information about whether the previous period was cheap enough to charge: a proxy for the demand-side conditions that produce negative prices, without look-ahead.

**Wind as the principal exogenous driver.** `total_wind_mw` from the Elexon Insights API is not merely a proxy for wind speed. The GB grid operator has already aggregated output across hundreds of geographically distributed wind farms spanning the full GB landmass, and expressed the result in MW of actual generation — a capacity-weighted, economically normalised measure accounting for the nonlinear relationship between wind speed and power output via the installed fleet's power curves. The result encodes the same meteorological information as gridded weather reanalysis data (ERA5, Met Office UKV), accessible via a single API call, in units directly relevant to price formation. So I don't think there is any need to go digging in weather data.

Two layers of meteorological structure are extractable from this signal. The mean dynamics — frontal passage timescales (1–2.5 days), synoptic cycles (3–7 days), seasonal patterns I think can be captured by fitting ARMA(p,q) on wind output levels. This is why the ACF of real |Lévy area| shows ~2–3 week periodicity: it is the autocorrelation of meteorological regimes encoded in the generation signal. The variance dynamics — how erratically wind output changes — correspond to unstable weather fronts and are the meteorological analogue of volatility clustering. A GARCH-X variant with wind output in the variance equation would extract this. Both conditioning signals are tested as candidates for 04b.

**Volatility model selection.** Does GB half-hourly electricity follow GARCH, EGARCH, or GJR-GARCH? Decided by AIC/BIC and ARCH-LM test, not assumption.

**Does the signature add anything beyond scalars?** A Granger causality test on ls_7 (Lévy area) against scalar returns and wind: if the signature does not Granger-cause future volatility beyond what scalars already contain, the signature approach is not empirically justified.

### Why SigCWGAN fixes the volatility clustering failure

SigCWGAN (Ni et al. 2021) uses an autoregressive generator: the hidden state at the end of segment t is passed to the generator for segment t+1. The Sig-Wasserstein loss directly penalises differences in expected signature between real and generated sequences — including inter-segment cross-terms that encode temporal clustering. This targets a structurally different objective from the ELBO and is the principled fix for the failure identified in notebook 05.

### 07d deep hedging — known limitation and planned fix

The current 07d trains a hedging agent on Schwartz one-factor OU paths: Gaussian innovations only, constant volatility, no wind coupling. An agent trained on these paths cannot learn regime-dependent or spike-aware strategies. The write-up states this explicitly.

Once SigCWGAN paths pass validation, the plan is drift removal (Bühler, Murray, Pakkanen & Wood 2021): learning a near-martingale measure on SigCWGAN output that removes exploitable statistical drift while preserving spike clustering and wind coupling. Those paths then replace the OU simulation in 07d. Biegler-König & Oeltz (2025) confirms that the wind-price correlation is critical for hedging renewable-linked exposures — the CfD hedge computed under OU assumptions is systematically mis-specified.

---

## Core mathematics

### Path signatures

For a continuous path $X : [0,T] \to \mathbb{R}^d$, the signature is the sequence of iterated integrals:

$$S(X)^{(n)} = \int_{0 < t_1 < \cdots < t_n < T} dX_{t_1} \otimes \cdots \otimes dX_{t_n}$$

Truncation error decays as $O(1/N!)$ — depth 4 is sufficient in practice. The expected signature characterises the law of a stochastic process (path-space analogue of the MGF). Any continuous functional on path space is approximated by a linear functional of the signature — the same UAT as the masters dissertation, applied to sequential data.

The 4D path uses lead-lag embedding on price so that level-2 cross-terms capture realised variance via the Lévy area (index 7), plus `total_wind_mw` so that cross-terms capture dynamic price-wind coupling nonlinearly. Wind appears both in the path and in the conditioning vector — different purposes, different information.

### Sig-Wasserstein loss (SigCWGAN)

$$\text{Sig-W}_1(\mu, \nu) = \|\mathbb{E}_{X \sim \mu}[S_M(X)] - \mathbb{E}_{Y \sim \nu}[S_M(Y)]\|_2$$

Replaces the adversarial discriminator with a theoretically grounded distance on path space. Directly penalises differences in expected signature including inter-segment temporal structure, rather than optimising a surrogate minimax objective that is unstable with small datasets.

---

## UK market context

- **Half-hourly settlement**: GB electricity clears in 48 half-hour periods per day. The system imbalance price is the marginal cost of real-time balancing — highly volatile with high wind penetration.
- **Negative prices**: ~2.6% of half-hours. Caused by must-run generation (nuclear, subsidised wind under old ROC regime) and transmission constraints. Log returns are undefined — arithmetic returns throughout.
- **CfD mechanism**: Generators receive `strike_price − reference_price` from LCCC. The reference price going negative is the key policy exposure, addressed in 07d.
- **B6 constraint**: Scotland-England transmission boundary. When Scottish wind exceeds capacity, generators are constrained off — creating implicit locational price separation not visible in the aggregate system price.
- **Pumped storage**: Cruachan (660 MW) and Dinorwig (1.8 GW) arbitrage price spikes. Sets a practical floor on negative price duration and ceiling on spike duration — both regime properties the model needs to reproduce.

---

## Key references

- Bühler, Horvath, Lyons et al. — *A Data-Driven Market Simulator for Small Data Environments* (2020)
- Ni, Szpruch, Wiese et al. — *Sig-Wasserstein GANs for Conditional Time Series Generation* (2021)
- Bühler, Murray, Pakkanen, Wood — *Deep Hedging with Market Impact* (2021) — drift removal
- Bühler, Gonon, Teichmann, Wood — *Deep Hedging* (2019)
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
