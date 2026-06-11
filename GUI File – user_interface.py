# dashboard.py
# ============================================================
# Futures ML Pipeline - Streamlit Dashboard
# ============================================================
# HOW TO RUN:
#   1. pip install streamlit scikit-learn pandas numpy matplotlib
#   2. streamlit run dashboard.py
#   3. Browser opens automatically at http://localhost:8501
#
# In IntelliJ IDEA:
#   - Open Terminal at the bottom
#   - cd into the futures-ml-pipeline folder
#   - Run: streamlit run dashboard.py
# ============================================================

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st

# Add pipeline root to path so modules can be imported
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="Futures ML Pipeline",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -- Custom CSS ----------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-box {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 8px;
        padding: 14px 18px;
        text-align: center;
    }
    .metric-label {
        font-size: 11px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        font-family: monospace;
    }
    .metric-good  { color: #00d97e; }
    .metric-bad   { color: #ff4757; }
    .metric-plain { color: #a0aec0; }
    .log-box {
        background: #0a0c10;
        border: 1px solid #2d3250;
        border-radius: 6px;
        padding: 12px 14px;
        font-family: monospace;
        font-size: 12px;
        height: 200px;
        overflow-y: auto;
        color: #a0aec0;
    }
    .section-title {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #555;
        margin-bottom: 10px;
    }
    div[data-testid="stSidebar"] {
        background-color: #111827;
    }
</style>
""", unsafe_allow_html=True)


# -- Import pipeline modules ---------------------------------------------------
@st.cache_resource(show_spinner=False)
def import_modules():
    from module1_dataprep.dataprep     import load_data, build_feature_matrix, time_split, make_rolling_windows, make_subsets
    from module2_training.training     import train_model
    from module3_testing.testing       import run_testing
    from module4_statistics.statistics import compute_metrics
    return load_data, build_feature_matrix, time_split, make_rolling_windows, make_subsets, train_model, run_testing, compute_metrics

try:
    load_data, build_feature_matrix, time_split, make_rolling_windows, make_subsets, train_model, run_testing, compute_metrics = import_modules()
    MODULES_OK = True
except Exception as e:
    MODULES_OK = False
    MODULE_ERROR = str(e)


# -- Helper: matplotlib figure to Streamlit -----------------------------------
def make_aum_figure(aum_history):
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    profit = aum_history[-1] >= aum_history[0]
    color  = "#00d97e" if profit else "#ff4757"
    ax.plot(aum_history, color=color, linewidth=2)
    ax.axhline(aum_history[0], color="#444", linestyle="--", linewidth=0.8)
    ax.fill_between(range(len(aum_history)), aum_history[0], aum_history,
                    where=[v >= aum_history[0] for v in aum_history],
                    alpha=0.15, color=color)
    ax.set_xlabel("Time step", color="#666", fontsize=9)
    ax.set_ylabel("Capital ($)", color="#666", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: "$" + f"{x:,.0f}"))
    ax.tick_params(colors="#555", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#2d3250")
    fig.tight_layout()
    return fig


def make_roc_figure(fpr, tpr, roc_auc):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.plot(fpr, tpr, color="#63b3ed", linewidth=2.5,
            label="AUC = {:.3f}".format(roc_auc))
    ax.fill_between(fpr, tpr, alpha=0.08, color="#63b3ed")
    ax.plot([0, 1], [0, 1], "--", color="#444", linewidth=1, label="Random (0.500)")
    ax.set_xlabel("False Positive Rate", color="#666", fontsize=9)
    ax.set_ylabel("True Positive Rate", color="#666", fontsize=9)
    ax.tick_params(colors="#555", labelsize=8)
    ax.legend(fontsize=8, frameon=False, labelcolor="#888")
    for sp in ax.spines.values():
        sp.set_color("#2d3250")
    fig.tight_layout()
    return fig


def make_accuracy_figure(metrics):
    labels = ["Overall\nAcc", "+ Precision\n(Longs)", "+ Recall", "- Precision\n(Shorts)", "Top-50%\nTrades"]
    values = [
        metrics["overall_accuracy"],
        metrics["precision_positive"],
        metrics["recall_positive"],
        metrics["precision_negative"],
        metrics["accuracy_threshold"],
    ]
    colors = ["#00d97e" if v >= 55 else "#ffd666" if v >= 45 else "#ff4757" for v in values]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    bars = ax.bar(labels, values, color=colors, edgecolor="#2d3250", linewidth=0.8, width=0.55)
    ax.axhline(50, color="#ff4757", linestyle="--", linewidth=1.2, label="50% random baseline", alpha=0.6)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy %", color="#666", fontsize=9)
    ax.tick_params(colors="#555", labelsize=8)
    ax.legend(fontsize=8, frameon=False, labelcolor="#888")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                "{:.1f}%".format(val),
                ha="center", va="bottom", fontsize=8, color="#ccc")
    for sp in ax.spines.values():
        sp.set_color("#2d3250")
    fig.tight_layout()
    return fig


def make_stability_figure(stab_data):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    x     = np.arange(len(stab_data))
    accs  = [d["acc"] for d in stab_data]
    pps   = [d["pp"]  for d in stab_data]
    lbls  = [d["label"][:16] for d in stab_data]
    ax.plot(x, accs, "o-", color="#a78bfa", linewidth=2, markersize=5, label="Overall Acc")
    ax.plot(x, pps,  "s-", color="#00d97e", linewidth=2, markersize=5, label="+ Precision")
    ax.axhline(50, color="#ff4757", linestyle="--", linewidth=1, alpha=0.6, label="50% baseline")
    ax.axhline(np.mean(accs), color="#a78bfa", linestyle=":", linewidth=1, alpha=0.5)
    ax.axhline(np.mean(pps),  color="#00d97e",  linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(lbls, rotation=30, ha="right", fontsize=8, color="#666")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy %", color="#666", fontsize=9)
    ax.tick_params(colors="#555", labelsize=8)
    ax.legend(fontsize=8, frameon=False, labelcolor="#888", loc="upper right")
    for sp in ax.spines.values():
        sp.set_color("#2d3250")
    fig.tight_layout()
    return fig


# -- Metric card HTML ----------------------------------------------------------
def metric_card(label, value, good=None):
    if good is True:
        cls = "metric-good"
    elif good is False:
        cls = "metric-bad"
    else:
        cls = "metric-plain"
    return """
    <div class="metric-box">
        <div class="metric-label">{label}</div>
        <div class="metric-value {cls}">{value}</div>
    </div>
    """.format(label=label, cls=cls, value=value)


# -- Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Futures ML Pipeline")
    st.markdown("---")

    st.markdown("**Data**")
    csv_path = st.text_input("CSV file path", value="TY.csv")
    max_rows  = st.number_input("Max rows", min_value=100, max_value=50000,
                                 value=3000, step=500)
    lookback  = st.number_input("Lookback (minutes)", min_value=1, max_value=20,
                                 value=4, step=1)
    t2_date   = st.text_input("T2 split date (YYYY-MM-DD)", value="2023-10-01")

    st.markdown("---")
    st.markdown("**Model**")
    model_type = st.selectbox("Algorithm", [
        "svm",
        "svm_tuned",
        "gradient_boost",
        "henry_sevan",
    ], format_func=lambda x: {
        "svm":            "SVM (RBF, balanced)",
        "svm_tuned":      "SVM Tuned (C=5, high weight)",
        "gradient_boost": "Gradient Boosting",
        "henry_sevan":    "Henry/Sevan benchmark",
    }[x])

    feature_mode = st.selectbox("Feature mode", [
        "engineered",
        "both",
        "raw",
        "relative",
    ], format_func=lambda x: {
        "engineered": "Engineered (tick_imb + vwap)",
        "both":       "Raw + Relative",
        "raw":        "Raw only (baseline)",
        "relative":   "Relative only",
    }[x])

    st.markdown("---")
    st.markdown("**Options**")
    train_as_test = st.checkbox("Train = Test (overfit check)", value=False)

    if model_type == "henry_sevan":
        st.warning("Henry/Sevan: linear SVM, no balancing. Predicts ~0 long positions.")

    st.markdown("---")
    csv_full = os.path.join(HERE, csv_path)
    if os.path.exists(csv_full):
        st.success("TY.csv found")
    else:
        st.error("TY.csv not found at: " + csv_full)

    if not MODULES_OK:
        st.error("Module import failed: " + MODULE_ERROR)


# -- Main area tabs ------------------------------------------------------------
tab_run, tab_bench, tab_stability, tab_about = st.tabs([
    "  Run Pipeline  ",
    "  Benchmark  ",
    "  Stability  ",
    "  About  ",
])


# ===================================================================
# TAB 1 -- RUN PIPELINE
# ===================================================================
with tab_run:
    st.markdown("### Run Pipeline")
    st.caption("Configure settings in the sidebar, then click Run.")

    run_btn = st.button("RUN PIPELINE", type="primary", use_container_width=True,
                        disabled=(not MODULES_OK))

    # Step progress bar
    step_cols = st.columns(4)
    step_names = [
        ("1", "Data Prep",   "rolling window features"),
        ("2", "Training",    "fit ML model"),
        ("3", "Testing",     "predictions"),
        ("4", "Statistics",  "metrics & charts"),
    ]
    step_placeholders = []
    for i, (n, name, sub) in enumerate(step_names):
        with step_cols[i]:
            ph = st.empty()
            step_placeholders.append(ph)

    def render_steps(active):
        for i, (n, name, sub) in enumerate(step_names):
            if i < active:
                step_placeholders[i].success("Module {} - {} - DONE".format(n, name))
            elif i == active:
                step_placeholders[i].info("Module {} - {} - Running...".format(n, name))
            else:
                step_placeholders[i].empty()
                step_placeholders[i].caption("[ {} ] {}  \n{}".format(n, name, sub))

    render_steps(-1)

    log_area   = st.empty()
    results_ph = st.empty()

    if run_btn:
        if not os.path.exists(csv_full):
            st.error("CSV file not found: " + csv_full)
            st.stop()

        logs = []
        def add_log(msg):
            logs.append(msg)
            log_area.code("\n".join(logs[-20:]), language="bash")

        # -- MODULE 1 ----------------------------------------------
        render_steps(0)
        add_log("[MODULE 1] Data preparation...")
        t0 = time.time()

        df = load_data(csv_full, max_rows=int(max_rows))
        add_log("  Loaded {:,} rows ({} to {})".format(
            len(df),
            df["Timestamp"].iloc[0].date(),
            df["Timestamp"].iloc[-1].date()))

        X, y, feature_names, timestamps = build_feature_matrix(
            df, lookback=int(lookback), feature_mode=feature_mode)
        add_log("  Features: {} | mode={}".format(X.shape, feature_mode))
        add_log("  Target: % price change | up={} down={}".format(
            int((y > 0).sum()), int((y <= 0).sum())))

        X_train, y_train, X_test, y_test, _, _ = time_split(
            X, y, timestamps, t2_date)

        if train_as_test:
            X_test, y_test = X_train.copy(), y_train.copy()
            add_log("  [!] train_as_test ON - test = training data")

        add_log("  Train: {:,} | Test: {:,}".format(len(X_train), len(X_test)))
        add_log("  Time: {:.1f}s".format(time.time() - t0))

        # -- MODULE 2 ----------------------------------------------
        render_steps(1)
        add_log("[MODULE 2] Training {}...".format(model_type.upper()))
        t1 = time.time()
        model, scaler = train_model(X_train, y_train, model_type=model_type)
        add_log("  Model: {}".format(type(model).__name__))
        add_log("  Time: {:.1f}s".format(time.time() - t1))

        # -- MODULE 3 ----------------------------------------------
        render_steps(2)
        add_log("[MODULE 3] Generating predictions...")
        t2 = time.time()
        y_true, y_pred, y_proba = run_testing(model, scaler, X_test, y_test)
        add_log("  Predictions: {:,} samples".format(len(y_pred)))
        add_log("  Time: {:.1f}s".format(time.time() - t2))

        # -- MODULE 4 ----------------------------------------------
        render_steps(3)
        add_log("[MODULE 4] Computing statistics...")
        t3 = time.time()
        metrics, aum, fpr, tpr, roc_auc = compute_metrics(
            y_true, y_pred, y_proba, y_continuous=y_test)
        add_log("  Time: {:.1f}s".format(time.time() - t3))
        add_log("[COMPLETE] Total time: {:.1f}s".format(time.time() - t0))

        # Render steps all done
        for i in range(4):
            step_placeholders[i].success("Module {} - {} - DONE".format(
                step_names[i][0], step_names[i][1]))

        # -- METRICS -----------------------------------------------
        st.markdown("---")
        st.markdown("#### Results")

        m1, m2, m3, m4, m5 = st.columns(5)
        acc = metrics["overall_accuracy"]
        pp  = metrics["precision_positive"]
        np_ = metrics["precision_negative"]
        auc = metrics["roc_auc"]
        pnl = metrics["pnl"]

        m1.markdown(metric_card("Overall Accuracy",
                                "{:.1f}%".format(acc), acc >= 55),
                    unsafe_allow_html=True)
        m2.markdown(metric_card("+ Precision (Longs)",
                                "{:.1f}%".format(pp), pp >= 50),
                    unsafe_allow_html=True)
        m3.markdown(metric_card("- Precision (Shorts)",
                                "{:.1f}%".format(np_), np_ >= 55),
                    unsafe_allow_html=True)
        m4.markdown(metric_card("ROC AUC",
                                "{:.3f}".format(auc), auc >= 0.55),
                    unsafe_allow_html=True)
        m5.markdown(metric_card("P and L",
                                ("+" if pnl >= 0 else "") + "${:,.0f}".format(pnl),
                                pnl >= 0),
                    unsafe_allow_html=True)

        # -- CHARTS ------------------------------------------------
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**AUM - Money Under Management**")
            st.pyplot(make_aum_figure(aum), use_container_width=True)

        with c2:
            st.markdown("**ROC Curve**")
            st.pyplot(make_roc_figure(fpr, tpr, roc_auc), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Sign Accuracy Breakdown**")
            st.pyplot(make_accuracy_figure(metrics), use_container_width=True)

        with c4:
            st.markdown("**Summary**")
            summary_data = {
                "Metric": [
                    "Model", "Feature mode", "Train=Test",
                    "Rows loaded", "Train samples", "Test samples",
                    "Overall accuracy", "+ Precision (longs)", "+ Recall",
                    "- Precision (shorts)", "Top-50% confidence acc",
                    "Predicted long", "Predicted short",
                    "ROC AUC", "Final capital", "P and L",
                ],
                "Value": [
                    model_type, feature_mode, str(train_as_test),
                    str(int(max_rows)), str(len(X_train)), str(len(X_test)),
                    "{:.2f}%".format(metrics["overall_accuracy"]),
                    "{:.2f}%".format(metrics["precision_positive"]),
                    "{:.2f}%".format(metrics["recall_positive"]),
                    "{:.2f}%".format(metrics["precision_negative"]),
                    "{:.2f}%".format(metrics["accuracy_threshold"]),
                    str(metrics["n_pred_positive"]),
                    str(metrics["n_pred_negative"]),
                    str(metrics["roc_auc"]),
                    "${:,.2f}".format(metrics["final_capital"]),
                    ("+" if pnl >= 0 else "") + "${:,.2f}".format(pnl),
                ],
            }
            st.dataframe(pd.DataFrame(summary_data),
                         use_container_width=True, hide_index=True)

        # Save metrics JSON
        json_path = os.path.join(HERE, "last_metrics.json")
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        st.caption("Metrics saved to: " + json_path)


# ===================================================================
# TAB 2 -- BENCHMARK
# ===================================================================
with tab_bench:
    st.markdown("### Feature and Model Benchmark")
    st.caption("Runs all feature modes and models. Compares against Henry/Sevan baseline.")

    bench_btn = st.button("RUN BENCHMARK", type="primary",
                           use_container_width=True, disabled=(not MODULES_OK),
                           key="bench_btn")

    if bench_btn:
        if not os.path.exists(csv_full):
            st.error("CSV not found: " + csv_full)
            st.stop()

        EXPERIMENTS = [
            ("raw + SVM",               "raw",        "svm"),
            ("relative + SVM",          "relative",   "svm"),
            ("both + SVM",              "both",       "svm"),
            ("engineered + SVM",        "engineered", "svm"),
            ("engineered + SVM Tuned",  "engineered", "svm_tuned"),
            ("engineered + GradBoost",  "engineered", "gradient_boost"),
            ("Henry/Sevan benchmark",   "raw",        "henry_sevan"),
            ("engineered + Train=Test", "engineered", "svm"),
        ]

        progress_bar = st.progress(0)
        status_text  = st.empty()
        rows = []

        for i, (label, fm, mt) in enumerate(EXPERIMENTS):
            status_text.text("Running: {} ({}/{})".format(label, i + 1, len(EXPERIMENTS)))
            try:
                df_b   = load_data(csv_full, max_rows=3000)
                Xb, yb, _, tsb = build_feature_matrix(df_b, 4, fm)
                Xt, yt, Xe, ye, _, _ = time_split(Xb, yb, tsb, t2_date)

                # overfit check for last experiment
                if "Train=Test" in label:
                    Xe, ye = Xt.copy(), yt.copy()

                if len(Xt) < 20 or len(Xe) < 10:
                    continue

                mod_b, sc_b = train_model(Xt, yt, mt)
                y1, y2, yp  = run_testing(mod_b, sc_b, Xe, ye)
                mb, _, _, _, _ = compute_metrics(y1, y2, yp, ye)

                rows.append({
                    "Experiment":        label,
                    "Overall %":         round(mb["overall_accuracy"], 1),
                    "+ Precision %":     round(mb["precision_positive"], 1),
                    "+ Recall %":        round(mb["recall_positive"], 1),
                    "- Precision %":     round(mb["precision_negative"], 1),
                    "AUC":               round(mb["roc_auc"], 4),
                    "P and L ($)":       round(mb["pnl"], 0),
                    "Pred Long":         mb["n_pred_positive"],
                    "Pred Short":        mb["n_pred_negative"],
                })
            except Exception as e:
                rows.append({"Experiment": label, "Overall %": 0,
                             "+ Precision %": 0, "Error": str(e)})

            progress_bar.progress((i + 1) / len(EXPERIMENTS))

        status_text.text("Benchmark complete.")

        if rows:
            df_results = pd.DataFrame(rows)

            # Colour-code the dataframe using Styler
            def colour_pct(val):
                if not isinstance(val, (int, float)):
                    return ""
                if val >= 55:
                    return "background-color: #1a3a2a; color: #00d97e"
                elif val >= 45:
                    return "background-color: #3a3010; color: #ffd666"
                else:
                    return "background-color: #3a1010; color: #ff4757"

            styled = (df_results.style
                      .applymap(colour_pct, subset=["Overall %", "+ Precision %", "- Precision %"])
                      .format({
                          "Overall %":     "{:.1f}%",
                          "+ Precision %": "{:.1f}%",
                          "+ Recall %":    "{:.1f}%",
                          "- Precision %": "{:.1f}%",
                          "AUC":           "{:.4f}",
                          "P and L ($)":   "${:+.0f}",
                      }))

            st.dataframe(styled, use_container_width=True, hide_index=True)

            # + Precision comparison chart
            st.markdown("---")
            st.markdown("**+ Precision Comparison (target: above 50%)**")
            fig_b, ax_b = plt.subplots(figsize=(10, 3.5))
            fig_b.patch.set_facecolor("#0e1117")
            ax_b.set_facecolor("#0e1117")
            labels_b = [r["Experiment"][:28] for r in rows]
            pps_b    = [r.get("+ Precision %", 0) for r in rows]
            colors_b = ["#00d97e" if v >= 50 else "#ffd666" if v >= 40 else "#ff4757" for v in pps_b]
            bars_b   = ax_b.bar(labels_b, pps_b, color=colors_b, edgecolor="#2d3250", linewidth=0.8)
            ax_b.axhline(50, color="#ff4757", linestyle="--", linewidth=1.5,
                         label="50% random baseline", alpha=0.7)
            ax_b.set_ylim(0, 105)
            ax_b.set_ylabel("+ Precision %", color="#666", fontsize=9)
            ax_b.tick_params(colors="#555", labelsize=8, axis="y")
            ax_b.tick_params(colors="#888", labelsize=8, axis="x", rotation=20)
            ax_b.legend(fontsize=8, frameon=False, labelcolor="#888")
            for bar, val in zip(bars_b, pps_b):
                ax_b.text(bar.get_x() + bar.get_width() / 2,
                          bar.get_height() + 1,
                          "{:.1f}%".format(val),
                          ha="center", va="bottom", fontsize=8, color="#ccc")
            for sp in ax_b.spines.values():
                sp.set_color("#2d3250")
            fig_b.tight_layout()
            st.pyplot(fig_b, use_container_width=True)

            # Save benchmark JSON
            bench_path = os.path.join(HERE, "benchmark_summary.json")
            with open(bench_path, "w") as f:
                json.dump(rows, f, indent=2)
            st.caption("Benchmark results saved to: " + bench_path)


# ===================================================================
# TAB 3 -- STABILITY
# ===================================================================
with tab_stability:
    st.markdown("### Rolling Window Stability")
    st.caption("Tests if accuracy holds across different time periods. Each window = 800 rows, 70/30 split.")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        n_windows   = st.slider("Number of windows", 4, 10, 6)
        window_size = st.slider("Window size (rows)", 400, 2000, 800, step=100)
    with col_s2:
        step_size   = st.slider("Step size (rows)", 200, 1000, 400, step=100)
        stab_model  = st.selectbox("Model", ["svm", "gradient_boost", "henry_sevan"],
                                   key="stab_model")

    stab_btn = st.button("RUN STABILITY ANALYSIS", type="primary",
                          use_container_width=True, disabled=(not MODULES_OK),
                          key="stab_btn")

    if stab_btn:
        if not os.path.exists(csv_full):
            st.error("CSV not found: " + csv_full)
            st.stop()

        rows_needed = window_size + (n_windows - 1) * step_size + 50
        df_s = load_data(csv_full, max_rows=min(rows_needed, 46158))
        Xs, ys, _, tss = build_feature_matrix(df_s, 4, "engineered")
        windows = make_rolling_windows(Xs, ys, tss,
                                        window_size=window_size,
                                        step=step_size)

        progress_s = st.progress(0)
        stab_data  = []

        for i, (Xtr, ytr, Xte, yte, lbl) in enumerate(windows[:n_windows]):
            if len(Xtr) < 20 or len(Xte) < 10:
                continue
            mod_s, sc_s = train_model(Xtr, ytr, stab_model)
            y1, y2, yp  = run_testing(mod_s, sc_s, Xte, yte)
            ms, _, _, _, _ = compute_metrics(y1, y2, yp, yte)
            stab_data.append({
                "label":  lbl,
                "acc":    ms["overall_accuracy"],
                "pp":     ms["precision_positive"],
                "np":     ms["precision_negative"],
                "auc":    ms["roc_auc"],
                "pnl":    ms["pnl"],
            })
            progress_s.progress((i + 1) / min(n_windows, len(windows)))

        if stab_data:
            accs = [d["acc"] for d in stab_data]
            pps  = [d["pp"]  for d in stab_data]

            # Summary stats
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Mean Accuracy",    "{:.1f}%".format(np.mean(accs)))
            sc2.metric("Acc Std Dev",      "+/-{:.1f}%".format(np.std(accs)))
            sc3.metric("Mean + Precision", "{:.1f}%".format(np.mean(pps)))
            sc4.metric("+Prec Std Dev",    "+/-{:.1f}%".format(np.std(pps)))

            st.markdown("---")
            st.markdown("**Rolling Window Chart**")
            st.pyplot(make_stability_figure(stab_data), use_container_width=True)

            st.markdown("**Window Detail Table**")
            df_stab = pd.DataFrame([{
                "Window":       d["label"],
                "Accuracy %":   round(d["acc"], 1),
                "+ Precision %": round(d["pp"],  1),
                "- Precision %": round(d["np"],  1),
                "AUC":           round(d["auc"],  4),
                "P and L ($)":   round(d["pnl"],  0),
            } for d in stab_data])
            st.dataframe(df_stab, use_container_width=True, hide_index=True)


# ===================================================================
# TAB 4 -- ABOUT
# ===================================================================
with tab_about:
    st.markdown("### About This Pipeline")
    st.markdown("""
**GitHub:** https://github.com/NikkyAnil/futures-ml-pipeline

---

#### Pipeline Modules

| Module | File | Purpose |
|--------|------|---------|
| Module 1 | module1_dataprep/dataprep.py | Load CSV, build rolling window features |
| Module 2 | module2_training/training.py | Train SVM or Gradient Boost classifier |
| Module 3 | module3_testing/testing.py   | Predict on test set |
| Module 4 | module4_statistics/statistics.py | Compute metrics, AUM, ROC |

---

#### Feature Modes

- **engineered** - tick_imb + vwap_dev + close_pos (strongest upward prediction signals)
- **both** - raw OHLCV + relative % change features
- **raw** - absolute OHLCV values only (matches Henry/Sevan original setup)
- **relative** - each value as % change from that minute's open price

---

#### Key Finding

Henry/Sevan's 63.6% accuracy uses linear SVM with no class balancing.
The data is 64% down moves, so predicting "down" every time gives 63.6% accuracy.
This is NOT a real signal - it predicts zero long positions.

Our engineered features with balanced SVM get genuine + precision of 38-45%
and the model actually predicts both directions.

---

#### How to Run

```bash
# Install dependencies
pip install streamlit scikit-learn pandas numpy matplotlib flask

# Run dashboard (this file)
streamlit run dashboard.py

# Run pipeline from command line
python run_pipeline.py

# Run all benchmark experiments
python run_benchmark.py
```
    """)
