import streamlit as st
import os
import joblib
from src.utilities import prepare_plotting_config, load_data, return_figures, prepare_X_y
from src.constraints import constraint_gross_exposure, constraint_var_historical, constraint_turnover_band
from src.evaluation import display_backtest_result
from src.tuning import tune_all_models
import pandas as pd
from src.training import models_config
from src.evaluation import display_backtest_result, plot_detailed_results, prepare_plotting_config, plot_series_st, plot_acf_pacf_st, plot_rolling_stats_st, plot_correlation_matrix_st

import sys
import os

print(f"Current working directory: {os.getcwd()}")

st.title("🏦 ReplicApp")
st.set_page_config(page_title="Hedge Fund Replication", layout="wide")
theme = st.sidebar.selectbox("Plot Theme", ["White", "Dark"])

st.sidebar.title("Navigation")

page = st.sidebar.radio("Choose Section", [
    "🧑‍💼 Investor Profile",
    "📈 Evaluation",
])

df_cleaned_imp_LLL1, futures_cleaned_imp_LLL1, indices, tickers_name_dict = load_data() 

folder_path = './'

# Define paths
data_raw_path = folder_path+"data/raw/"
data_interim_path = folder_path+"data/interim/"
data_processed_path = folder_path+"data/processed/"

#define directory for models
results_dir = folder_path+"results/"

model_names = [
    'linear',
    'ridge',
    'lasso',
    'elasticnet',
    'kalman'
]

def load_results():
    res = []
    for m in model_names:
        path = os.path.join(results_dir, m, f"{m}_best_model.joblib")
        if not os.path.exists(path):
            st.warning(f"Result file not found for model: {m}. Please run tuning.")
            continue
        try:
            res.append(joblib.load(path))
        except Exception as e:
            st.error(f"Could not load result for {m}: {e}")
    return res

def show_characterisation():
    st.header("🧑‍💼 Investor Profile")
    st.markdown("""
    Answer a few questions to help us understand your investment style and build a custom index to replicate.
    """)

    index_options = {
            'HFRXGL': "HFRX Global Hedge Fund Index",
            'LEGATRUU': "Bloomberg Global Aggregate Bond",
            'MXWO': "MSCI World",
            'MXWD': "MSCI ACWI"
        }

    # --- Preset values for weights ---
    default_weights = {
            "Conservative": {'HFRXGL': 0.15, 'LEGATRUU': 0.7, 'MXWO': 0.1, 'MXWD': 0.05},
            "Balanced": {'HFRXGL': 0.2, 'LEGATRUU': 0.3, 'MXWO': 0.35, 'MXWD': 0.15},
            "Growth Seeking": {'HFRXGL': 0.2, 'LEGATRUU': 0.1, 'MXWO': 0.5, 'MXWD': 0.2},
            "Risk Maximiser": {'HFRXGL': 0.1, 'LEGATRUU': 0., 'MXWO': 0.65, 'MXWD': 0.25}
        }

    # --- Preset values for constraints ---
    preset_values = {
        "Conservative": {"max_gross": 1.2, "max_var": 0.05, "min_turnover":0.05, "max_turnover": 0.08},
        "Balanced":     {"max_gross": 1.5, "max_var": 0.08, "min_turnover": 0.02, "max_turnover": 0.10},
        "Growth Seeking":    {"max_gross": 1.7, "max_var": 0.12,"min_turnover": 0.01, "max_turnover": 0.15},
        "Risk Maximiser":        {"max_gross": 2.0, "max_var": 0.2, "min_turnover": 0.01, "max_turnover": 0.25},
        }
    
    # --- Risk Profile Quiz ---
    q1 = st.radio(
        "How much risk are you comfortable taking with your investments?",
        [
            ("🟢 I prefer to avoid risk and preserve my capital.", 0),
            ("🟡 I’m comfortable with some risk for moderate returns.", 1),
            ("🟠 I want to grow my investments and can accept higher ups and downs.", 2),
            ("🔴 I want to maximize returns and am comfortable with large swings in value.", 3)
        ],
        format_func=lambda x: x[0],
        key="q1"
    )
    q2 = st.radio(
        "How would you feel if your portfolio dropped 10% in a month?",
        [
            ("😱 Very uncomfortable, I want to minimize losses.", 0),
            ("😬 Somewhat uncomfortable, but I can tolerate it.", 1),
            ("🙂 Not too worried, I expect some volatility.", 2),
            ("😎 I see it as an opportunity to gain more in the long run.", 3)
        ],
        format_func=lambda x: x[0],
        key="q2"
    )
    q3 = st.radio(
        "What is your main investment goal?",
        [
            ("🛡️ Preserve my wealth with minimal risk.", 0),
            ("⚖️ Balance between growth and safety.", 1),
            ("🚀 Grow my wealth, accepting some risk.", 2),
            ("🎢 Maximize returns, even if it means big ups and downs.", 3)
        ],
        format_func=lambda x: x[0],
        key="q3"
    )
    q4 = st.radio(
        "How long do you plan to keep your money invested?",
        [
            ("⏳ Less than 3 years.", 0),
            ("🕰️ 3–5 years.", 1),
            ("🕰️ 5–10 years.", 2),
            ("🕰️ 10+ years.", 3)
        ],
        format_func=lambda x: x[0],
        key="q4"
    )

    # Only set preset if all questions are answered (not None)
    if all(x is not None for x in [q1, q2, q3, q4]):
        score = q1[1] + q2[1] + q3[1] + q4[1]
        if score <= 2:
            preset = "Conservative"
        elif score <= 5:
            preset = "Balanced"
        elif score <= 8:
            preset = "Growth Seeking"
        else:
            preset = "Risk Maximiser"
        st.session_state.preset = preset

        # Reset dependent state variables when preset changes
        for k, v in default_weights[preset].items():
            st.session_state[f"weight_{k}"] = v

        # Reset constraint sliders to reflect new preset
        st.session_state["max_gross"] = preset_values[preset]["max_gross"]
        st.session_state["max_var"] = preset_values[preset]["max_var"]
        st.session_state["min_turnover"] = preset_values[preset]["min_turnover"]
        st.session_state["max_turnover"] = preset_values[preset]["max_turnover"]
    else:
        preset = st.session_state.get("preset", "Conservative")  # fallback


    st.success(f"**Your suggested risk profile is: {preset}**")

    profile_descriptions = {
        "Conservative": "🟢 You prioritise capital preservation and minimal risk. Designed for investors like retirees or very cautious savers. This portfolio protects principal, avoids large drawdowns, and generates steady bond income with low volatility.",
        "Balanced": "🟡 You aim for a mix of stability and growth. A true 60/40-style strategy with a modern twist. Suitable for mid-career professionals or risk-aware investors aiming for reliable long-term growth with fewer shocks.",
        "Growth Seeking": "🟠 You focus on long-term capital growth. Best suited for long-term investors who want growth but value smart risk management. This portfolio combines global equity exposure with tactical alternatives and modest stability.",
        "Risk Maximiser": "🔴 You’re comfortable with high risk and volatility. This is for fearless, return-hungry investors who can handle market swings. No bonds means no drag — and no cushion. Suitable only for those with long horizons and strong nerves."
    }
    st.info(profile_descriptions.get(preset, ""))

    with st.expander("Index split (advanced users only)", expanded=False):
        st.markdown("---")
        st.markdown("### Choose Your Target Index Blend")
        st.markdown("Select the hedge fund indices and weights you want to replicate:")
        
        weights = {}
        st.markdown("Adjust the sliders to set your preferred blend (must sum to 1.0):")
        total = 0
        for idx, label in index_options.items():
            # Use session_state if exists, else default
            slider_key = f"weight_{idx}"
            default_val = st.session_state.get(idx, float(default_weights[preset].get(idx, 0.0)))
            w = st.slider(
                f"{label} ({idx})",
                0.0, 1.0, default_val, 0.05,
                key=slider_key
            )
            weights[idx] = w
            total += w
        if abs(total - 1.0) > 0.01:
            st.warning("Weights should sum to 1.0 for a valid index blend.")

        st.session_state.index_weights = weights

    

    # Use session_state for constraints if available, else preset
    init_gross = st.session_state.get("max_gross", preset_values[preset]["max_gross"])
    init_max_var = st.session_state.get("max_var", preset_values[preset]["max_var"])
    init_max_turnover = st.session_state.get("max_turnover", preset_values[preset]["max_turnover"])
    init_min_turnover = st.session_state.get("min_turnover", preset_values[preset]["min_turnover"])

    constraint_funcs = []

    with st.expander("Constraint Options (advanced users only)", expanded=False):
        st.markdown("Modify constraint parameters below if you want fine control.")
        gross_exposure = st.checkbox("Gross Exposure Constraint", value=True, key="gross_exposure")
        var_constraint = st.checkbox("VaR Constraint", value=True, key="var_constraint")
        turnover_constraint = st.checkbox("Turnover Band Constraint", value=True, key="turnover_constraint")

        if gross_exposure:
            constraint_funcs.append(constraint_gross_exposure)
        if var_constraint:
            constraint_funcs.append(constraint_var_historical)
        if turnover_constraint:
            constraint_funcs.append(constraint_turnover_band)

        # Save constraint values to session_state
        if gross_exposure:
            max_gross = st.slider("Max Gross Exposure", 1.0, 5.0, init_gross, step=0.1, key="max_gross")
            
        if var_constraint:
            max_var = st.slider("Max VaR (e.g. 0.08 = 8%)", 0.01, 0.20, init_max_var, step=0.01, key="max_var")

        if turnover_constraint:
            min_turnover = st.slider("Min Turnover (No-Trade Band)", 0.0, 0.10, init_min_turnover, step=0.01, key="min_turnover")
            max_turnover = st.slider("Max Turnover Cap", 0.05, 0.50, init_max_turnover, step=0.01, key="max_turnover")

    X, y = prepare_X_y(df_cleaned_imp_LLL1, indices, st.session_state.index_weights)
    st.session_state.X = X #for plotting later
    if st.button("Tune Models"):
        with st.spinner("Tuning models, please wait..."):
            models_tuned_results, best_configs = tune_all_models(
                models_config=models_config,
                X=X,
                y=y,
                constraint_funcs=constraint_funcs,
                constraint_params={
                    'max_gross': st.session_state.get('max_gross') if gross_exposure else None,
                    'max_var': st.session_state.get('max_var') if var_constraint else None,
                    'var_confidence': 0.01,
                    'var_horizon': 4,
                    'min_turnover': st.session_state.get('min_turnover') if turnover_constraint else None,
                    'max_turnover': st.session_state.get('max_turnover') if turnover_constraint else None
                },
                step=1,
                metric_key='information_ratio',
                save_dir=results_dir
            )
        st.success("Models tuned successfully! Head to Evaluation to see the results")

def show_evaluation():
    st.header("📈 Portfolio Evaluation")
    st.markdown("""
    In this section, we evaluate all tuned models on their ability to replicate the target index.
    Each model's key metrics are shown below. The best model (by correlation to the target) is highlighted and analyzed in detail.
    """)
    cached_results = load_results()
    # Load cached results (list of dicts with 'best_result' and 'best_config')
    if not cached_results:
        st.error("No cached results found. Please run the tuning section first.")
        return

    st.subheader("Model Performance Summary")
    st.markdown("Each column below summarises a model's out-of-sample performance:")

    cols = st.columns(len(cached_results))
    summaries = []
    for col, r in zip(cols, cached_results):
        result_obj = r['best_result']
        with col:
            st.markdown(f"**{result_obj.model_name.capitalize()}**")
            st.metric("Sharpe", f"{result_obj.aggregate_metrics['replica_sharpe']:.3f}")
            st.metric("IR", f"{result_obj.aggregate_metrics['information_ratio']:.3f}")
            st.metric("TE", f"{result_obj.aggregate_metrics['tracking_error']:.3f}")
            st.metric("Corr", f"{result_obj.aggregate_metrics['correlation']:.3f}")
            st.metric("TC", f"{result_obj.aggregate_metrics['transaction_costs_sum']:.5f}")
            st.caption(f"Params: `{result_obj.model_params}`")
        # Collect for best model selection
        summary = result_obj.summary()
        summary['model_name'] = result_obj.model_name
        summary['model_params'] = result_obj.model_params
        summaries.append(summary)

    # Find the best model by correlation
    df_summaries = pd.DataFrame(summaries)
    best_model_idx = df_summaries['IR'].abs().idxmin()
    best_result_obj = cached_results[best_model_idx]['best_result']
    best_config_params = cached_results[best_model_idx]['best_config']

    st.markdown("---")
    st.subheader("🏆 Best Model Analysis")
    st.markdown(f"""
    The best model is **{best_result_obj.model_name.capitalize()}**  
    - **Correlation to target:** `{df_summaries.loc[best_model_idx, 'corr']:.3f}`
    - **Sharpe Ratio:** `{df_summaries.loc[best_model_idx, 'sharpe']:.3f}`
    - **Information Ratio:** `{df_summaries.loc[best_model_idx, 'IR']:.3f}`
    - **Tracking Error:** `{df_summaries.loc[best_model_idx, 'TE']:.3f}`
    - **Transaction Costs:** `{df_summaries.loc[best_model_idx, 'transaction_costs']:.6f}`
    - **Parameters:** `{best_result_obj.model_params}`
    """)

    st.markdown("#### Cumulative Returns and Key Metrics")
    display_backtest_result(best_result_obj, theme=theme)

    st.markdown("#### Detailed Diagnostics")
    st.info("Below are detailed plots and diagnostics for the best model, including drawdowns, exposures, VaR, and more.")
    best_config = prepare_plotting_config(best_result_obj, best_config_params)
    st.info(st.session_state.index_weights)
    plot_detailed_results(best_config, st.session_state.X, save_dir= results_dir ,max_var_threshold=0.08, theme=theme)

if page == "🧑‍💼 Investor Profile":
    show_characterisation()

elif page == "📈 Evaluation":
    show_evaluation()