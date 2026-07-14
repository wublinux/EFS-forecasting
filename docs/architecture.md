# System architecture

```mermaid
flowchart LR
    A["Canonical or legacy CSV"] --> B["Python validation and conversion"]
    B --> C["2016-only imputation and scaling"]
    C --> D["Shared 2017 evaluation dates"]
    D --> E["Seasonal naive"]
    D --> F["Rolling ARIMA"]
    D --> G["32-unit LSTM"]
    D --> H["Versioned JSON/CSV job"]
    H --> I["MATLAB R2024b interval type-2 EFS"]
    I --> J["Predictions, model, rules, activations"]
    E --> K["Audited run artifact"]
    F --> K
    G --> K
    J --> K
    K --> L["CLI and Streamlit"]
    K --> M["Static GitHub Pages evidence"]
```

## Boundary between Python and MATLAB

Python owns the data contract, leakage controls, evaluation dates, baseline models, metric
aggregation, and artifact manifest. MATLAB owns interval type-2 FIS initialization, three-stage
optimization, inference, and rule explanation. The boundary is a versioned JSON job plus CSV
inputs and outputs; no MATLAB Engine binding is required.

For a normal workstation, Python launches `matlab -batch`. Public GitHub runners use the same
job files but execute them through the official MathWorks Action because its temporary public
project license is scoped to that action. Cached job output is accepted only after input hashes,
features, seed, and EFS configuration match.

GitHub Pages is deliberately static. It presents checked-in evidence and never executes MATLAB
or accepts private datasets.
