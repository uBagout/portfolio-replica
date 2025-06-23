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

st.title("🏦 Hedge Fund Replication")
st.set_page_config(page_title="Hedge Fund Replication", layout="wide")
theme = st.sidebar.selectbox("Plot Theme", ["White", "Dark"])

st.sidebar.title("Navigation")

page = st.sidebar.radio("Choose Section", [
    "📊 Data Analysis",
    "🧠 Tuning",
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

def show_eda():
    figs = return_figures()

    st.header("Exploratory Data Analysis")
    
    st.markdown("### Raw Data")
        
        #
    st.dataframe(df_cleaned_imp_LLL1.head(3))
    #st.image(figs[1], caption="Correlation Matrix")  # implement from your notebook

    fig0 = plot_correlation_matrix_st(df_cleaned_imp_LLL1, title="Correlation Matrix of LLL1 Imputed Data", theme=theme)
    st.pyplot(fig0)

    st.markdown("### LLL1 Imputed Data Overview")
    st.markdown("LLL1 had some missing data, we relied on pulling values using TvDataFeed module")
    
    if theme=="Light":
        st.image(figs[0], caption="LLL1 Imputed Data Overview")
    else:
        st.image(figs[1], caption="LLL1 Imputed Data Overview")
    st.markdown("### Select an index to explore:")
    selected_index = st.selectbox("Choose an index", indices.columns.tolist())

    if st.button("PLOT"):
        st.subheader(f"Time Series: {selected_index}")
        fig1 = plot_series_st(df_cleaned_imp_LLL1, selected_index, title=f"{selected_index} Over Time", theme=theme)
        st.pyplot(fig1)

        st.subheader(f"ACF & PACF: {selected_index}")
        fig2 = plot_acf_pacf_st(df_cleaned_imp_LLL1[selected_index], lags=20, title_prefix=selected_index, theme=theme)
        st.pyplot(fig2)

        st.subheader(f"Rolling Mean & Std: {selected_index}")
        fig3 = plot_rolling_stats_st(df_cleaned_imp_LLL1[selected_index], window=20, title=f"{selected_index} Rolling Stats", theme=theme)
        st.pyplot(fig3)

# Prepare data
index_weights = {
    'HFRXGL': 0.50,
    'LEGATRUU': 0.25,
    'MXWO':   0.25,
    'MXWD':   0.
}
X, y = prepare_X_y(indices, futures_cleaned_imp_LLL1, index_weights)

def show_tuning():
    st.header("🧠 Tuning")

   # --- Risk Profile Quiz ---
    st.markdown("#### Let's find your risk profile!")

    q1 = st.radio(
        "1. How much risk are you comfortable taking with your investments?",
        [
            ("🟢 I prefer to avoid risk and preserve my capital.", 0),
            ("🟡 I’m comfortable with some risk for moderate returns.", 1),
            ("🟠 I want to grow my investments and can accept higher ups and downs.", 2),
            ("🔴 I want to maximize returns and am comfortable with large swings in value.", 3)
        ],
        format_func=lambda x: x[0]
    )
    q2 = st.radio(
        "2. How would you feel if your portfolio dropped 10% in a month?",
        [
            ("😱 Very uncomfortable, I want to minimize losses.", 0),
            ("😬 Somewhat uncomfortable, but I can tolerate it.", 1),
            ("🙂 Not too worried, I expect some volatility.", 2),
            ("😎 I see it as an opportunity to gain more in the long run.", 3)
        ],
        format_func=lambda x: x[0]
    )
    q3 = st.radio(
        "3. What is your main investment goal?",
        [
            ("🛡️ Preserve my wealth with minimal risk.", 0),
            ("⚖️ Balance between growth and safety.", 1),
            ("🚀 Grow my wealth, accepting some risk.", 2),
            ("🎢 Maximize returns, even if it means big ups and downs.", 3)
        ],
        format_func=lambda x: x[0]
    )
    q4 = st.radio(
        "4. How long do you plan to keep your money invested?",
        [
            ("⏳ Less than 3 years.", 0),
            ("🕰️ 3–5 years.", 1),
            ("🕰️ 5–10 years.", 2),
            ("🕰️ 10+ years.", 3)
        ],
        format_func=lambda x: x[0]
    )

    score = q1[1] + q2[1] + q3[1] + q4[1]

    if score <= 2:
        st.session_state.preset = "Conservative"
    elif score <= 5:
        st.session_state.preset = "Balanced"
    elif score <= 8:
        st.session_state.preset = "Growth Seeking"
    else:
        st.session_state.preset = "Risk Maximiser"

    st.success(f"**Your suggested risk profile is: {st.session_state.preset}**")

    profile_descriptions = {
    "Conservative": "🟢 You prioritise capital preservation and minimal risk. Ideal for short-term goals or low tolerance for loss.",
    "Balanced": "🟡 You aim for a mix of stability and growth. Moderate tolerance for risk and medium-term investment goals.",
    "Growth Seeking": "🟠 You focus on long-term capital growth and accept significant short-term fluctuations.",
    "Risk Maximiser": "🔴 You’re comfortable with high risk and volatility in exchange for the potential of high returns."
    }

    st.info(profile_descriptions.get(st.session_state.preset, ""))

    # --- Preset values for constraints ---
    preset_values = {
        "Conservative": {"max_gross": 1.2, "max_var": 0.03, "var_confidence": 0.01, "min_turnover":0.05, "max_turnover": 0.08},
        "Balanced":     {"max_gross": 1.5, "max_var": 0.06,"var_confidence": 0.01, "min_turnover": 0.02, "max_turnover": 0.10},
        "Growth Seeking":    {"max_gross": 2.0, "max_var": 0.08, "var_confidence": 0.05,"min_turnover": 0.01, "max_turnover": 0.15},
        "Risk Maximiser":        {"max_gross": 3.0, "max_var": 0.12, "var_confidence": 0.05, "min_turnover": 0.01, "max_turnover": 0.25},
    }

    preset = st.session_state.get("preset", "Balanced")
    max_gross = preset_values[preset]["max_gross"]
    max_var = preset_values[preset]["max_var"]
    var_confidence = preset_values[preset]["var_confidence"] 
    max_turnover = preset_values[preset]["max_turnover"]
    min_turnover = preset_values[preset]["min_turnover"]

    constraint_funcs = []

    # --- Advanced Options ---
    with st.expander("Advanced Options (for experts only)", expanded=False):
        st.markdown("Modify constraint parameters below if you want fine control.")
        gross_exposure = st.checkbox("Gross Exposure Constraint", value=True)
        var_constraint = st.checkbox("VaR Constraint", value=True)
        turnover_constraint = st.checkbox("Turnover Band Constraint", value=True)

        
        if gross_exposure:
            constraint_funcs.append(constraint_gross_exposure)
        if var_constraint:
            constraint_funcs.append(constraint_var_historical)
        if turnover_constraint:
            constraint_funcs.append(constraint_turnover_band)

        if gross_exposure:
            max_gross = st.slider("Max Gross Exposure", 1.0, 5.0, max_gross, step=0.1)
        if var_constraint:
            max_var = st.slider("Max VaR (e.g. 0.08 = 8%)", 0.01, 0.20, max_var, step=0.01)
            var_confidence = st.slider("VaR Confidence Level", 0.01, 0.10, var_confidence, step=0.01)
            var_horizon = st.slider("VaR Horizon (weeks)", 1, 12, 4, step=1)
        else:
            var_horizon = 4
        if turnover_constraint:
            min_turnover = st.slider("Min Turnover (No-Trade Band)", 0.0, 0.10, min_turnover, step=0.01)
            max_turnover = st.slider("Max Turnover Cap", 0.05, 0.50, max_turnover, step=0.01)

    if st.button("Tune Models"):
        with st.spinner("Tuning models, please wait..."):
            models_tuned_results, best_configs = tune_all_models(
                models_config=models_config,
                X=X,
                y=y,
                constraint_funcs=constraint_funcs,
                constraint_params={
                    'max_gross': max_gross,
                    'max_var': max_var,
                    'var_confidence': var_confidence,
                    'var_horizon': var_horizon,
                    'min_turnover': min_turnover,
                    'max_turnover': max_turnover
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
            st.metric(" ", f"{result_obj.aggregate_metrics['information_ratio']:.3f}")
            st.metric("TE", f"{result_obj.aggregate_metrics['tracking_error']:.3f}")
            st.metric("Corr", f"{result_obj.aggregate_metrics['correlation']:.3f}")
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
    - **Parameters:** `{best_result_obj.model_params}`
    """)

    st.markdown("#### Cumulative Returns and Key Metrics")
    display_backtest_result(best_result_obj, theme=theme)

    st.markdown("#### Detailed Diagnostics")
    st.info("Below are detailed plots and diagnostics for the best model, including drawdowns, exposures, VaR, and more.")
    best_config = prepare_plotting_config(best_result_obj, best_config_params)
    plot_detailed_results(best_config, X, max_var_threshold=0.08, theme=theme)

if page == "📊 Data Analysis":
    show_eda()

elif page == "🧠 Tuning":
    show_tuning()

elif page == "📈 Evaluation":
    show_evaluation()