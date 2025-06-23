import streamlit as st
import os
import joblib
from src.utilities import prepare_plotting_config, load_data, return_figures, prepare_X_y
from src.constraints import constraint_gross_exposure, constraint_var_historical, constraint_turnover_band
from src.evaluation import display_backtest_result
from src.tuning import tune_all_models
import pandas as pd
from src.training import models_config
from src.evaluation import display_backtest_result, plot_detailed_results, prepare_plotting_config

print(f"Current working directory: {os.getcwd()}")

st.set_page_config(page_title="Hedge Fund Replication", layout="wide")
st.title("🏦 Hedge Fund Replication")
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
    st.image(figs[1], caption="Correlation Matrix")  # implement from your notebook

    st.markdown("### LLL1 Imputed Data Overview")
    st.markdown("LLL1 had some missing data, we relied on pulling values using TvDataFeed module")
    st.image(figs[0], caption="LLL1 Imputed Data Overview")

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

    st.markdown("#### Constraints")

    gross_exposure = st.checkbox("Gross Exposure Constraint", value=True)
    var_constraint = st.checkbox("VaR Constraint", value=True)
    turnover_constraint = st.checkbox("Turnover Band Constraint", value=True)

    constraint_funcs = []
    if gross_exposure:
        constraint_funcs.append(constraint_gross_exposure)
        max_gross = st.slider("Max Gross Exposure", 1.0, 5.0, 2.0, step=0.1)
    if var_constraint:
        constraint_funcs.append(constraint_var_historical)
        max_var = st.slider("Max VaR (e.g. 0.08 = 8%)", 0.01, 0.20, 0.08, step=0.01)
        var_confidence = st.slider("VaR Confidence Level", 0.01, 0.10, 0.01, step=0.01)
        var_horizon = st.slider("VaR Horizon (weeks)", 1, 12, 4, step=1)

    if turnover_constraint:
        constraint_funcs.append(constraint_turnover_band)
        min_turnover = st.slider("Min Turnover (No-Trade Band)", 0.0, 0.10, 0.02, step=0.01)
        max_turnover = st.slider("Max Turnover Cap", 0.05, 0.50, 0.10, step=0.01)

    if st.button("Tune Models"):
        with st.spinner("Tuning models, please wait..."):

            models_tuned_results, best_configs = tune_all_models(
                models_config=models_config,
                X=X,
                y=y,
                constraint_funcs=[constraint_gross_exposure, constraint_var_historical, constraint_turnover_band],
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
        st.success("Models tuned successfully!")
        st.markdown("### Head to Evaluation to see the best results")

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
            st.caption(f"Params: `{result_obj.model_params}`")
        # Collect for best model selection
        summary = result_obj.summary()
        summary['model_name'] = result_obj.model_name
        summary['model_params'] = result_obj.model_params
        summaries.append(summary)

    # Find the best model by correlation
    df_summaries = pd.DataFrame(summaries)
    best_model_idx = df_summaries['corr'].idxmax()
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
    display_backtest_result(best_result_obj)

    st.markdown("#### Detailed Diagnostics")
    st.info("Below are detailed plots and diagnostics for the best model, including drawdowns, exposures, VaR, and more.")
    best_config = prepare_plotting_config(best_result_obj, best_config_params)
    plot_detailed_results(best_config, X, max_var_threshold=0.08)

if page == "📊 Data Analysis":
    show_eda()

elif page == "🧠 Tuning":
    show_tuning()

elif page == "📈 Evaluation":
    show_evaluation()