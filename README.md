# quantlab

A personal research repository exploring **quantitative finance**, **model risk**, and **machine learning for derivatives pricing and hedging**.

> 🔍 **Note**: This project is *unrelated* to [QuantLib.org](https://www.quantlib.org/). It is a private learning and experimentation space by [AlmostAnna](https://github.com/AlmostAnna).

---

## What’s Inside

This repo contains:
- **Classical models**: Black-Scholes, Heston
- **Hedging analysis**: Error decomposition, discrete trading
- **Machine learning**: Buehler-style deep hedging with transaction costs
- **Stress testing**: Sensitivity to volatility misspecification, rebalancing frequency

All code is organized to support **reuse**, **clarity**, and **diagnostics**—not just one-off experiments.

---

## Structure
```
quantlab/ 
├── notebooks/  
│ ├── models/ # Stochastic volatility, etc. 
│ ├── model_risk/ # Hedging errors, replication failure, Greeks 
│ └── ml/ # Deep hedging, training diagnostics 
├── src/ # Reusable quant primitives (installable as 'quantlab')
|   ├── quantlab/
│      ├── calibration/ 
|      ├── data/ # Synthetic data generation
│      ├── hedging/ # Greeks, naive strategies
|      ├── instruments/ 
│      ├── market_data/ 
|      ├── ml/ # Models, metrics
|      ├── models/
|      ├── pricing/
|      ├── sim/ # MC simulations
│      └── utils/  
├── ml/ # ML-specific training and evaluation
├── tests/ # Tests
├── pyproject.toml # For editable install 
└── environment.yml
```

---

## Getting Started

1. Clone and install:
   ```bash
   git clone https://github.com/AlmostAnna/quantlab.git
   cd quantlab
   pip install -e .[dev]
   ```

---
## Philosophy

- Clarity over cleverness: Code should speak for itself.
- Model risk matters: Every assumption is surfaced and tested.
- ML as a tool, not a black box: Diagnostics, baselines, and stress tests are first-class citizens.

© 2025 - 2026 AlmostAnna — For learning, reflection, and professional growth.



