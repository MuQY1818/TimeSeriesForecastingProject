# T-LAFS: Time-series Language-augmented Feature Search

This project introduces **T-LAFS (Time-series Language-augmented Feature Search)**, an advanced, AI-driven framework for automated time-series feature engineering and forecasting.

The core of this project is the `tlafs_core.py` script, a highly configurable entry point that demonstrates the full capabilities of the T-LAFS framework on various time-series datasets.

## Core Idea: Probe-and-Validate Strategy

T-LAFS utilizes a novel **"Probe-and-Validate"** strategy to achieve state-of-the-art forecasting performance.

1.  **Probe Phase**: An AI Strategist, powered by a Large Language Model (LLM), collaborates with a specialized **"Probe" model** (e.g., `DualStreamAttentionProbe`, `QuantumDualStreamProbe`). The LLM generates feature engineering plans (e.g., creating lags, interactions, rolling statistics) which are then executed. The selected Probe model evaluates the effectiveness of these new features. This iterative process allows the system to automatically search the vast feature space and discover an optimal set of features. The AI can also decide to prune less useful features, making the search more efficient.

2.  **Validate Phase**: Once the optimal feature set is discovered by the probe, it is handed off to a suite of diverse, powerful models (including LightGBM, RandomForest, XGBoost, and various Neural Networks like Transformers). These models are trained on the AI-generated features, and the best-performing model is selected as the final champion.

This strategy ensures that the feature engineering is not overfitted to a single model architecture and that the discovered features are robust and widely effective.

## Key Features

- **Automated Feature Engineering**: Leverages an LLM (e.g., GPT-4o) to intelligently add, create, and remove features.
- **Modular Probe Architecture**: Easily switch between different probe models (e.g., standard `DualStream`, `QuantumDualStream`) to evaluate different feature-sensing strategies.
- **Flexible Dataset Handling**: Switch between different datasets by changing a single variable in the main script. The framework adapts automatically.
- **Robust Validation**: The final feature set is validated against a wide range of standard and advanced models to ensure generalization.
- **Data-Leakage Safe**: The entire framework is built with strict adherence to time-series principles, ensuring no future information is leaked during feature creation or model training.
- **Result Persistence**: Automatically saves comprehensive experiment results, including the best feature plan and all model scores, to a JSON file in the `results/` directory.

## Technology Stack

- **Core**: Python 3.x
- **AI/ML**:
    - PyTorch
    - Scikit-learn
    - LightGBM
    - XGBoost
- **Data Handling**: Pandas, NumPy
- **LLM Integration**: OpenAI API
- **Plotting**: Matplotlib

## How to Run the Experiment

1.  **Setup Environment**:
    - Make sure you have Python installed.
    - Install the required packages:
      ```bash
      pip install pandas numpy torch scikit-learn lightgbm xgboost openai matplotlib
      ```
2.  **Set API Keys**:
    - For the LLM strategist to work, you must set your OpenAI API key and base URL as environment variables.
      ```bash
      export OPENAI_API_KEY="your_api_key"
      export OPENAI_BASE_URL="your_base_url" # Optional, if using a proxy
      ```
      (On Windows, use `set` instead of `export`).
      If you don't set these, the script will fall back to a pre-defined, non-AI plan.

3.  **Configure the Experiment**:
    - Open the `tlafs_core.py` file.
    - Inside the `main()` function, you can easily configure the experiment:
      - **`DATASET_TYPE`**: Choose the dataset to use.
        - `'min_daily_temps'`: The original daily minimum temperatures dataset.
        - `'total_cleaned'`: A dataset of daily transaction counts.
      - **`PROBE_NAME`**: Select the probe model to guide the feature search.
        - `'dual_stream'`: The standard Dual-Stream Attention Probe.
        - `'quantum_dual_stream'`: A probe inspired by quantum mechanics principles.
        - `'bayesian_quantum'`: A probe incorporating Bayesian principles with quantum concepts.

    Example configuration:
    ```python
    def main():
        # ===== Configure your experiment here =====
        DATASET_TYPE = 'total_cleaned'  # Switch to the sales dataset
        PROBE_NAME = 'quantum_dual_stream' # Use the quantum probe
        # ... rest of the settings ...
    ```

4.  **Run the script**:
    ```bash
    python tlafs_core.py
    ```

The script will execute the full Probe-and-Validate pipeline using your configuration. Final results, including plots and a JSON summary, will be saved to the `plots/` and `results/` directories respectively.

## Example Result

On the `min_daily_temps.csv` dataset, a previous run of the T-LAFS framework automatically discovered a feature set that allowed a **Transformer** model to achieve a final **R² score of 0.9977**, showcasing the power of AI-driven feature discovery. By changing the configuration, you can now apply this powerful framework to other datasets like `total_cleaned.csv`.