# T-LAFS: Time-series Language-augmented Feature Search

This project introduces **T-LAFS (Time-series Language-augmented Feature Search)**, an advanced, AI-driven framework for automated time-series feature engineering and forecasting.

The core of this project is the `clp_probe_experiment.py` script, which demonstrates the full capabilities of the T-LAFS framework on a given time-series dataset.

## Core Idea: Probe-and-Validate Strategy

T-LAFS utilizes a novel **"Probe-and-Validate"** strategy to achieve state-of-the-art forecasting performance.

1.  **Probe Phase**: An AI Strategist, powered by a Large Language Model (LLM), collaborates with a specialized "Probe" model (`ChronoLanguageProbe`). The LLM generates feature engineering plans (e.g., creating lags, interactions, rolling statistics) which are then executed. The `ChronoLanguageProbe` model, uniquely capable of understanding both quantitative data and qualitative "linguistic" patterns of change, evaluates the effectiveness of these new features. This iterative process allows the system to automatically search the vast feature space and discover an optimal set of features. The AI can also decide to prune less useful features, making the search more efficient.

2.  **Validate Phase**: Once the optimal feature set is discovered by the probe, it is handed off to a suite of diverse, powerful models (including LightGBM, RandomForest, XGBoost, and various Neural Networks like Transformers). These models are trained on the AI-generated features, and the best-performing model is selected as the final champion.

This strategy ensures that the feature engineering is not overfitted to a single model architecture and that the discovered features are robust and widely effective.

## Key Features

- **Automated Feature Engineering**: Leverages an LLM (e.g., GPT-4o) to intelligently add, create, and remove features.
- **Dual-Stream Probe Model**: The custom `ChronoLanguageProbe` (PyTorch-based) model analyzes both numerical data and "linguistic" patterns of change (e.g., "sharp increase," "stable").
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

3.  **Run the script**:
    ```bash
    python clp_probe_experiment.py
    ```

The script will execute the full Probe-and-Validate pipeline. Final results, including plots and a JSON summary, will be saved to the `plots/` and `results/` directories respectively.

## Example Result

On the `min_daily_temps.csv` dataset, the T-LAFS framework was able to automatically discover a feature set that allowed a **Transformer** model to achieve a final **R² score of 0.9977**, showcasing the power of AI-driven feature discovery.