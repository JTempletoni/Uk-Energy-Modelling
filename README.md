# Energy Synthetic Data Pipeline

I am trying to make a mathematically grounded pipeline for generating synthetic energy commodity market data, built on path signatures, variational autoencoders, and the UK electricity market.(all of this is subject to change as I learn by doing!) I also want to try and learn new tools that I investigated while studying but didn't have time to learn. Hopefully the synthetic data pipeline can used with different data, e.g. equities, bonds or even sports? 

## My historic thread

This project connects three stages of prior work:

- **Undergrad dissertation**: Markov chain Monte Carlo agent games of wealth inequality (Yardsale model) — bilateral exchange, ergodic theory, stationary distributions of interacting particle systems
- **Masters dissertation**: Neural network option pricing in the Hull-White volatility model — FFT pricing scheme, universal approximation theorem, KDE-based synthetic data for training
- **This project**: Combines both threads — Markov regime switching for non-stationarity, path signatures as the rigorous feature map for sequential data (grounded in the same UAT as the masters dissertation), VAE/GAN generative models, applied to real UK energy market data

## Data sources (all free)

| Source | What it provides | URL |
|--------|-----------------|-----|
| Elexon Insights API | Half-hourly GB electricity: system price, imbalance, generation by fuel | developer.data.elexon.co.uk |
| Carbon Intensity API | Regional generation mix including Scotland, half-hourly from 2018 | api.carbonintensity.org.uk |
| National Grid ESO | Wind/solar forecasts, demand, B6 boundary constraints | data.nationalgrideso.com |
| LCCC CfD data | Strike prices, CfD generation volumes, difference payments (quarterly) | lowcarboncontracts.uk |

## UK market structure (key concepts)

- **Half-hourly settlement**: GB electricity clears in 48 half-hour periods per day. The System Buy/Sell Price (imbalance price) is the marginal cost of balancing in real time — highly volatile, especially with high wind penetration
- **CfD (Contract for Difference)**: Primary renewable support mechanism. Generator receives `strike_price - market_reference_price` from LCCC (or pays back if market > strike). Payoff is a financial forward, with basis risk from curtailment. So prices go negative and there is a policy to deal with that.

These two affect supply:
- **B6 constraint**: The Scotland-England transmission boundary. When Scottish wind exceeds transfer capacity, generators are constrained off and paid constraint payments. Creates implicit locational price separation
- **Pumped storage**: Cruachan (660 MW, Argyll) and Dinorwig (1.8 GW, Wales) act as grid batteries — pump when prices low/negative, generate when prices high. Sets a practical floor on negative price duration and ceiling contribution to spikes

## Core mathematics

### Path signatures

For a continuous path $X : [0,T] \to \mathbb{R}^d$, the signature is the sequence of iterated integrals:

$$S(X) = \left(1, \mathbf{X}^{(1)}, \mathbf{X}^{(2)}, \ldots \right)$$

where $\mathbf{X}^{(n)} = \int_{0 < t_1 < \cdots < t_n < T} dX_{t_1} \otimes \cdots \otimes dX_{t_n}$

Key properties:
- Uniquely determines the path (up to tree-like equivalence)
- Truncation error decays as $O(1/N!)$ — depth 4-5 sufficient in practice
- Universal approximation: any continuous functional on path space is approximated by a linear functional of the signature
- Expected signature characterises the law of a stochastic process (path-space analogue of the MGF)

### Signature kernel

$$k(X, Y) = \langle S(X), S(Y) \rangle_{T((E))}$$

Positive definite kernel on path space. Used for the MMD two-sample test to validate generated paths against real paths.

### Sig-Wasserstein loss (SigCWGAN)

$$\text{Sig-W}_1(\mu, \nu) = \|\mathbb{E}_{X \sim \mu}[S_M(X)] - \mathbb{E}_{Y \sim \nu}[S_M(Y)]\|_2$$

Replaces the adversarial discriminator with a stable, theoretically grounded distance on path space.

## Markov structure (four roles)

1. **Regime switching HMM**: Hidden states (calm / volatile / spike / negative) with learned transition matrix $\Pi$. CVAE conditioned on posterior regime probabilities from forward algorithm
2. **MCMC calibration**: Bayesian posterior over model parameters via HMC/NUTS (PyMC or numpyro). Propagates uncertainty into generated path distributions
3. **Sequential Monte Carlo**: Particle filter for online hidden state estimation as new data arrives. MCMC moves at resample step prevent degeneracy
4. **Yardsale → LOB**: Bilateral exchange Markov chain → stationary Boltzmann-Gibbs distribution → queue level dynamics in limit order book. Connects undergrad dissertation directly to Avellaneda-Stoikov market making

## Notebook structure

```
00_market_structure.ipynb       — UK market mechanics, CfD payoff, B6 constraints
01_data_acquisition.ipynb       — Elexon + Carbon Intensity APIs → parquet
02_eda_stylized_facts.ipynb     — Spikes, seasonality, ACF, heavy tails
03_signature_features.ipynb     — Lead-lag, log-signatures, financial interpretation
04a_cvae_generator.ipynb        — Encoder/decoder, ELBO + sig-MMD loss
04b_sigcwgan.ipynb              — Conditional expected signature + AR-FNN
05_validation.ipynb             — Sig-MMD test, KS, ACF, arbitrage checks
06_generate_dataset.ipynb       — 10k scenarios → DuckDB
07a_stoch_control.ipynb         — Gas storage / swing options (HJB)
07b_sql_analytics.ipynb         — DuckDB scenario analytics
07c_portfolio.ipynb             — CVaR, factor risk, MCMC uncertainty
07d_option_pricing.ipynb        — CfD pricing, deep hedging
07e_kernel_filter.ipynb         — Sig-kernel, GP regression, Kalman
07f_avellaneda.ipynb            — Avellaneda-Stoikov market making
```

## Key references

- Bühl er, Horvath, Lyons et al. — *A Data-Driven Market Simulator for Small Data Environments* (2020)
- Ni, Szpruch, Wiese et al. — *Sig-Wasserstein GANs for Conditional Time Series Generation* (2021)
- Bühl er, Gonon, Teichmann, Wood — *Deep Hedging* (2019)
- Chevyrev & Oberhauser — *Signature moments to characterise laws of stochastic processes* (2022)
- Cont — *Empirical properties of asset returns: stylized facts and statistical issues* (2001)
- Avellaneda & Stoikov — *High-frequency trading in a limit order book* (2008)
