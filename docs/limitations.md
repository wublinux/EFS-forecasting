# Limitations

- The thesis dataset contains only 75 dates per year and eight categories; conclusions should
  not be generalized to large or multi-echelon supply chains.
- Publication rights for the source sales data are unconfirmed, so reproducible public examples
  use synthetic data rather than the original observations.
- Weather can improve or degrade forecasts depending on category. The ablation is empirical and
  cannot establish a causal effect.
- Rule count grows exponentially with input count. The six-input weather model permits up to 64
  rules even with only two membership functions per input.
- EFS training depends on licensed MATLAB toolboxes and is substantially slower than simple
  statistical baselines.
- LSTM estimates are unstable on very small datasets; multiple seeds and uncertainty summaries
  are required.
- The current system retrains in batches. It does not yet evolve online, consume live streams,
  optimize inventory decisions, or forecast multiple supply-chain echelons.
- The Streamlit application is a local research interface, not a hardened multi-user production
  service.
