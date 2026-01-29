# Research References

This repository implements methods from the following papers and resources, spanning classical quantitative finance, stochastic volatility modeling, and modern machine learning approaches to derivatives pricing and hedging.

## Deep Hedging & Reinforcement Learning
- [Deep Hedging: Hedging Derivatives Under Multiple Constraints](https://arxiv.org/abs/1802.03042) (Buehler et al., 2019)
  *Applies supervised learning to minimize hedging error with transaction costs and market frictions*

## Option Pricing & Volatility Models
- [The Volatility Surface: A Practitioner's Guide](https://www.wiley.com/en-us/The+Volatility+Surface%3A+A+Practitioner%27s+Guide-p-9780471792512) (Gatheral, 2006)
- [A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options](https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/Heston.pdf) (Heston, 1993)
  *Original stochastic volatility model with semi-analytic pricing formulas*
- [Stochastic Volatility Modelling](https://www.lorenzobergomi.com) (Bergomi, 2016)
  *Comprehensive treatment of stochastic volatility modeling and hedging implications*

## Fourier Methods for Option Pricing
- [A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions](http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/COS.pdf) (Fang & Oosterlee, 2008)
  *Introduces the COS method for fast and accurate option pricing using Fourier-cosine expansions*
- [Precise option pricing by the COS method--How to choose the truncation range](https://arxiv.org/abs/2109.01030) (Junike & Pankrashkin, 2022)
  *Another way to choose a truncation interval in the COS method*

## Jump-Diffusion Models
- [Option Pricing when Underlying Stock Returns are Discontinuous](https://www.sciencedirect.com/science/article/pii/0304405X76900222) (Merton, 1976)
  *Foundational paper on jump-diffusion models for option pricing*
- [The Variance Gamma Process and Option Pricing](https://www.researchgate.net/publication/2331688_The_Variance_Gamma_Process_and_Option_Pricing) (Madan et al., 1998)
  *Introduction of the Variance Gamma model for capturing heavy tails and skewness*

## Numerical Methods & Calibration
- [Monte Carlo Methods in Financial Engineering](https://www.bauer.uh.edu/spirrong/Monte_Carlo_Methods_In_Financial_Enginee.pdf) (Glasserman, 2003)
  *Comprehensive reference for Monte Carlo methods in finance*

---

## Implementation Status
- [x] COS Method (Fang & Oosterlee, 2008) - Implemented in `pricing.heston.cos`
- [x] Heston Model (Heston, 1993) - Implemented in `models.heston`
- [x] Deep Hedging (Buehler et al., 2019) - Implemented in `ml.models.hedge_net`
- [ ] Jump-Diffusion Models (Merton, 1976) - Planned for future implementation
- [ ] Diffusion Models for Path Generation - Planned for integration with `generative-models-journey`
