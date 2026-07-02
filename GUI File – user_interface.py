# dashboard.py
# ============================================================
# Futures ML Pipeline Dashboard
# ============================================================
# HOW TO RUN:
#   Step 1: pip install streamlit scikit-learn pandas numpy matplotlib
#   Step 2: streamlit run dashboard.py
#   Browser opens automatically at http://localhost:8501
#
# In IntelliJ IDEA:
#   - Open Terminal at the bottom of the screen
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

# Add this folder to path so the module files can be found
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ---- Page config ------------------------------------------------------------
st.set_page_config(
    page_title="Futures ML Pipeline",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Custom CSS -------------------------------------------------------------
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0e1117; }
    [data-testid="stSidebar"]          { background: #111827; }
    .metric-box {
        background: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 8px;
        padding: 14px 16px;
        text-align: center;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 10px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .metric-value { font-size: 24px; font-weight: 700; font-family: monospace; }
    .good  { color: #00d97e; }
    .bad   { color: #ff4757; }
    .plain { color: #a0aec0; }
    .step-done    { color: #00d97e; font-weight: bold; }
    .step-active  { color: #63b3ed; font-weight: bold; }
    .step-waiting { color: #444; }
</style>
""", unsafe_allow_html=True)


# ---- Import your real modules -----------------------------------------------
@st.cache_resource(show_spinner=False)
def import_pipeline_modules():
    from module1_dataprep   import prepare_data
    from module2_training   import train_model
    from module3_testing    import test_model
    from module4_summary import evaluate_model
    return prepare_data, train_model, test_model, evaluate_model

MODULES_OK = True
MODULE_ERROR = ""
try:
    prepare_data, train_model, test_model, evaluate_model = import_pipeline_modules()
except Exception as e:
    MODULES_OK = False
    MODULE_ERROR = str(e)


# ---- Chart helpers ----------------------------------------------------------
def chart_aum(capital_history):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    data   = capital_history
    profit = data[-1] >= data[0]
    color  = "#00d97e" if profit else "#ff4757"
    ax.plot(data, color=color, linewidth=2)
    ax.axhline(data[0], color="#444", linestyle="--", linewidth=0.8)
    ax.fill_between(range(len(data)), data[0], data,
                    where=[v >= data[0] for v in data],
                    alpha=0.15, color=color)
    ax.set_xlabel("Time step", color="#666", fontsize=9)
    ax.set_ylabel("Capital ($)", color="#666", fontsize=9)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: "${:,.0f}".format(x)))
    ax.tick_params(colors="#555", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#2d3250")
    fig.tight_layout()
    return fig


def chart_accuracy_bars(dir_acc, mse, mae):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    pct = dir_acc * 100
    bar_color = "#00d97e" if pct >= 50 else "#ff4757"
    ax.bar(["Directional\nAccuracy"], [pct],
           color=bar_color, edgecolor="#2d3250", width=0.35)
    ax.axhline(50, color="#ff4757", linestyle="--",
               linewidth=1.2, label="50% random baseline", alpha=0.7)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy %", color="#666", fontsize=9)
    ax.tick_params(colors="#555", labelsize=9)
    ax.legend(fontsize=8, frameon=False, labelcolor="#888")
    ax.text(0, pct + 2, "{:.1f}%".format(pct),
            ha="center", va="bottom", fontsize=13,
            fontweight="bold", color=bar_color)
    for sp in ax.spines.values():
        sp.set_color("#2d3250")
    fig.tight_layout()
    return fig


def chart_pred_vs_actual(y_true, y_pred, n=200):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    n = min(n, len(y_true))
    x = np.arange(n)
    ax.plot(x, y_true[:n], color="#63b3ed",
            linewidth=1.2, label="Actual", alpha=0.8)
    ax.plot(x, y_pred[:n], color="#f6ad55",
            linewidth=1.2, label="Predicted", alpha=0.8)
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_xlabel("Test sample", color="#666", fontsize=9)
    ax.set_ylabel("Close - Open", color="#666", fontsize=9)
    ax.tick_params(colors="#555", labelsize=8)
    ax.legend(fontsize=8, frameon=False, labelcolor="#aaa")
    for sp in ax.spines.values():
        sp.set_color("#2d3250")
    fig.tight_layout()
    return fig


def chart_sign_scatter(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    correct   = (np.sign(y_true) == np.sign(y_pred))
    incorrect = ~correct
    ax.scatter(y_true[correct],   y_pred[correct],
               color="#00d97e", s=4, alpha=0.5, label="Correct sign")
    ax.scatter(y_true[incorrect], y_pred[incorrect],
               color="#ff4757", s=4, alpha=0.5, label="Wrong sign")
    ax.axhline(0, color="#555", linewidth=0.7)
    ax.axvline(0, color="#555", linewidth=0.7)
    ax.set_xlabel("Actual", color="#666", fontsize=9)
    ax.set_ylabel("Predicted", color="#666", fontsize=9)
    ax.tick_params(colors="#555", labelsize=8)
    ax.legend(fontsize=7, frameon=False, labelcolor="#aaa",
              markerscale=2)
    for sp in ax.spines.values():
        sp.set_color("#2d3250")
    fig.tight_layout()
    return fig


def chart_stability(rows):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    x     = np.arange(len(rows))
    accs  = [r["directional_accuracy"] * 100 for r in rows]
    lbls  = [r["label"][:14] for r in rows]
    color = ["#00d97e" if v >= 50 else "#ff4757" for v in accs]
    ax.bar(x, accs, color=color, edgecolor="#2d3250", linewidth=0.7, width=0.55)
    ax.axhline(50, color="#ff4757", linestyle="--",
               linewidth=1.2, label="50% baseline", alpha=0.7)
    ax.axhline(np.mean(accs), color="#a78bfa", linestyle=":",
               linewidth=1.2, label="Mean {:.1f}%".format(np.mean(accs)))
    ax.set_xticks(x)
    ax.set_xticklabels(lbls, rotation=30, ha="right",
                       fontsize=8, color="#888")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Directional Accuracy %", color="#666", fontsize=9)
    ax.tick_params(colors="#555", labelsize=8, axis="y")
    ax.legend(fontsize=8, frameon=False, labelcolor="#aaa")
    for bar, val in zip(ax.patches, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                "{:.1f}%".format(val),
                ha="center", va="bottom",
                fontsize=7.5, color="#ccc")
    for sp in ax.spines.values():
        sp.set_color("#2d3250")
    fig.tight_layout()
    return fig


def metric_html(label, value, good=None):
    cls = "good" if good is True else "bad" if good is False else "plain"
    return (
        '<div class="metric-box">'
        '<div class="metric-label">{}</div>'
        '<div class="metric-value {}">{}</div>'
        '</div>'
    ).format(label, cls, value)


# ---- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.markdown("## Futures ML Pipeline")
    st.markdown("---")

    st.markdown("**Data**")
    csv_input = st.text_input("CSV file", value="TY.csv")
    max_rows  = st.number_input("Max rows (0 = all)", min_value=0,
                                 max_value=100000, value=5000, step=1000)
    lookback  = st.number_input("Lookback (minutes)", min_value=1,
                                 max_value=20, value=4, step=1)
    split_pct = st.slider("Train split %", 50, 90, 70, step=5)
    target_type = st.selectbox("Target type",
                                ["difference", "return"],
                                format_func=lambda x:
                                    "Close - Open (difference)" if x == "difference"
                                    else "(Close - Open) / Open (return)")

    st.markdown("---")
    st.markdown("**Model (SVR)**")
    kernel    = st.selectbox("Kernel", ["rbf", "linear", "poly"])
    C_val     = st.number_input("C (regularisation)", min_value=0.1,
                                 max_value=20.0, value=1.0, step=0.5)
    epsilon   = st.number_input("Epsilon", min_value=0.001,
                                 max_value=1.0, value=0.1, step=0.05)

    st.markdown("---")
    train_as_test = st.checkbox("Train = Test (overfit check)", value=False)

    st.markdown("---")
    # csv_path = os.path.join(HERE, csv_input)
    csv_path = csv_input # comment
    if os.path.exists(csv_path):
        st.success("{} found".format(csv_input))
    else:
        st.error("File not found:\n{}".format(csv_path))

    if not MODULES_OK:
        st.error("Import error:\n" + MODULE_ERROR)
    else:
        st.success("All 4 modules loaded")


# ---- Tabs -------------------------------------------------------------------
tab_run, tab_bench, tab_stability, tab_about = st.tabs([
    "  Run Pipeline  ",
    "  Benchmark  ",
    "  Stability  ",
    "  About  ",
])


# =============================================================================
# TAB 1 - RUN PIPELINE
# =============================================================================
with tab_run:
    st.markdown("### Run Pipeline")
    st.caption(
        "Uses your exact module files: "
        "module1_dataprep.py / module2_training.py / "
        "module3_testing.py / module4_evaluation.py"
    )

    run_btn = st.button(
        "RUN FULL PIPELINE",
        type="primary",
        use_container_width=True,
        disabled=(not MODULES_OK)
    )

    # Step indicators
    s1, s2, s3, s4 = st.columns(4)
    ph = [s1.empty(), s2.empty(), s3.empty(), s4.empty()]
    step_info = [
        ("Module 1", "prepare_data()"),
        ("Module 2", "train_model()"),
        ("Module 3", "test_model()"),
        ("Module 4", "evaluate_model()"),
    ]

    def show_steps(active):
        for i, (name, fn) in enumerate(step_info):
            if i < active:
                ph[i].success("{} Done".format(name))
            elif i == active:
                ph[i].info("{} Running...".format(name))
            else:
                ph[i].caption("[ {} ]\n{}".format(name, fn))

    show_steps(-1)
    log_ph     = st.empty()
    results_ph = st.empty()

    if run_btn:
        if not os.path.exists(csv_path):
            st.error("CSV not found: " + csv_path)
            st.stop()

        logs = []

        def log(msg):
            logs.append(msg)
            log_ph.code("\n".join(logs[-25:]), language="bash")

        # -- MODULE 1: prepare_data -------------------------------------------
        show_steps(0)
        log("[MODULE 1] prepare_data() ...")
        t0 = time.time()

        rows = int(max_rows) if max_rows > 0 else None
        X, y = prepare_data(csv_path,
                             lookback=int(lookback),
                             target_type=target_type)

        # Limit rows if requested
        if rows is not None and rows < len(X):
            X = X[:rows]
            y = y[:rows]

        log("  Features: {} samples x {} features".format(X.shape[0], X.shape[1]))
        log("  Target type: {}".format(target_type))
        log("  Up moves: {}  Down moves: {}".format(
            int((y > 0).sum()), int((y <= 0).sum())))

        # Train/test split by index
        split_idx = int(len(X) * split_pct / 100)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test,  y_test  = X[split_idx:], y[split_idx:]

        if train_as_test:
            X_test, y_test = X_train.copy(), y_train.copy()
            log("  [!] train_as_test ON - using training set as test set")

        log("  Train: {:,}  |  Test: {:,}".format(len(X_train), len(X_test)))
        log("  Time: {:.1f}s".format(time.time() - t0))

        # -- MODULE 2: train_model --------------------------------------------
        show_steps(1)
        log("[MODULE 2] train_model() ...")
        t1 = time.time()

        model, scaler = train_model(
            X_train, y_train,
            C=float(C_val),
            epsilon=float(epsilon),
            kernel=kernel
        )
        log("  Model: SVR(kernel={}, C={}, epsilon={})".format(
            kernel, C_val, epsilon))
        log("  Scaler: StandardScaler fitted")
        log("  Time: {:.1f}s".format(time.time() - t1))

        # -- MODULE 3: test_model ---------------------------------------------
        show_steps(2)
        log("[MODULE 3] test_model() ...")
        t2 = time.time()

        predictions, y_test_out = test_model(model, scaler, X_test, y_test)
        log("  Predictions: {:,} samples".format(len(predictions)))
        log("  Time: {:.1f}s".format(time.time() - t2))

        # -- MODULE 4: evaluate_model -----------------------------------------
        show_steps(3)
        log("[MODULE 4] evaluate_model() ...")
        t3 = time.time()

        results = evaluate_model(y_test_out, predictions)
        log("  MSE:  {:.6f}".format(results["mse"]))
        log("  MAE:  {:.6f}".format(results["mae"]))
        log("  Directional accuracy: {:.1f}%".format(
            results["directional_accuracy"] * 100))
        log("  Final capital: ${:,.2f}".format(results["final_capital"]))
        log("[COMPLETE] Total time: {:.1f}s".format(time.time() - t0))

        # Mark all steps done
        for i in range(4):
            ph[i].success("{} Done".format(step_info[i][0]))

        # ---- Metrics row ----------------------------------------------------
        st.markdown("---")
        st.markdown("#### Results")

        dir_acc = results["directional_accuracy"]
        pnl     = results["final_capital"] - 10000

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown(metric_html(
            "Directional Accuracy",
            "{:.1f}%".format(dir_acc * 100),
            dir_acc >= 0.50
        ), unsafe_allow_html=True)
        c2.markdown(metric_html(
            "MSE",
            "{:.6f}".format(results["mse"]),
            None
        ), unsafe_allow_html=True)
        c3.markdown(metric_html(
            "MAE",
            "{:.6f}".format(results["mae"]),
            None
        ), unsafe_allow_html=True)
        c4.markdown(metric_html(
            "Final Capital",
            "${:,.2f}".format(results["final_capital"]),
            results["final_capital"] >= 10000
        ), unsafe_allow_html=True)
        c5.markdown(metric_html(
            "P and L",
            ("+" if pnl >= 0 else "") + "${:,.2f}".format(pnl),
            pnl >= 0
        ), unsafe_allow_html=True)

        # ---- Charts ---------------------------------------------------------
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**AUM - Money Under Management**")
            st.pyplot(chart_aum(results["capital_history"]),
                      use_container_width=True)

        with col2:
            st.markdown("**Directional Accuracy**")
            st.pyplot(
                chart_accuracy_bars(
                    results["directional_accuracy"],
                    results["mse"],
                    results["mae"]
                ),
                use_container_width=True
            )

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("**Predicted vs Actual (first 200 test samples)**")
            st.pyplot(
                chart_pred_vs_actual(y_test_out, predictions, n=200),
                use_container_width=True
            )

        with col4:
            st.markdown("**Sign Accuracy Scatter**")
            st.caption("Green = correct direction, Red = wrong direction")
            st.pyplot(
                chart_sign_scatter(y_test_out, predictions),
                use_container_width=True
            )

        # ---- Summary table --------------------------------------------------
        st.markdown("---")
        st.markdown("**Summary**")
        summary = pd.DataFrame({
            "Metric": [
                "CSV file", "Rows used", "Lookback",
                "Train samples", "Test samples", "Train=Test",
                "Kernel", "C", "Epsilon",
                "Directional accuracy", "MSE", "MAE",
                "Starting capital", "Final capital", "P and L",
            ],
            "Value": [
                csv_input,
                str(len(X)),
                str(lookback),
                str(len(X_train)),
                str(len(X_test)),
                "YES" if train_as_test else "NO",
                kernel,
                str(C_val),
                str(epsilon),
                "{:.2f}%".format(dir_acc * 100),
                "{:.6f}".format(results["mse"]),
                "{:.6f}".format(results["mae"]),
                "$10,000.00",
                "${:,.2f}".format(results["final_capital"]),
                ("+" if pnl >= 0 else "") + "${:,.2f}".format(pnl),
            ],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        # Save results to JSON
        save_results = {
            "directional_accuracy": results["directional_accuracy"],
            "mse":                  results["mse"],
            "mae":                  results["mae"],
            "final_capital":        results["final_capital"],
            "pnl":                  pnl,
            "config": {
                "csv":          csv_input,
                "rows":         len(X),
                "lookback":     lookback,
                "split_pct":    split_pct,
                "kernel":       kernel,
                "C":            C_val,
                "epsilon":      epsilon,
                "train_as_test": train_as_test,
            }
        }
        json_path = os.path.join(HERE, "last_run.json")
        with open(json_path, "w") as f:
            json.dump(save_results, f, indent=2)
        st.caption("Results saved to last_run.json")


# =============================================================================
# TAB 2 - BENCHMARK
# =============================================================================
with tab_bench:
    st.markdown("### Benchmark: Kernel and Row Size Comparison")
    st.caption(
        "Runs all three SVR kernels (rbf / linear / poly) on two dataset sizes. "
        "Uses your real module files."
    )

    bench_btn = st.button(
        "RUN BENCHMARK",
        type="primary",
        use_container_width=True,
        disabled=(not MODULES_OK),
        key="bench_btn"
    )

    if bench_btn:
        if not os.path.exists(csv_path):
            st.error("CSV not found: " + csv_path)
            st.stop()

        CONFIGS = [
            ("rbf    | 3k rows",  "rbf",    1.0, 0.1, 3000),
            ("linear | 3k rows",  "linear", 1.0, 0.1, 3000),
            ("poly   | 3k rows",  "poly",   1.0, 0.1, 3000),
            ("rbf    | 5k rows",  "rbf",    1.0, 0.1, 5000),
            ("linear | 5k rows",  "linear", 1.0, 0.1, 5000),
            ("rbf    | Train=Test","rbf",   1.0, 0.1, 3000),
        ]

        progress = st.progress(0)
        status   = st.empty()
        bench_rows = []

        for i, (label, kern, c, eps, nrows) in enumerate(CONFIGS):
            status.text("Running: {} ({}/{})".format(label, i + 1, len(CONFIGS)))
            try:
                Xb, yb = prepare_data(csv_path,
                                      lookback=4,
                                      target_type="difference")
                if nrows < len(Xb):
                    Xb, yb = Xb[:nrows], yb[:nrows]

                sp = int(len(Xb) * 0.7)
                Xt, yt = Xb[:sp], yb[:sp]
                Xe, ye = Xb[sp:], yb[sp:]

                if "Train=Test" in label:
                    Xe, ye = Xt.copy(), yt.copy()

                mb, sb = train_model(Xt, yt, C=c, epsilon=eps, kernel=kern)
                preds_b, ye_out = test_model(mb, sb, Xe, ye)
                res_b = evaluate_model(ye_out, preds_b)

                bench_rows.append({
                    "Experiment":           label,
                    "Directional Acc %":    round(res_b["directional_accuracy"] * 100, 1),
                    "MSE":                  round(res_b["mse"], 6),
                    "MAE":                  round(res_b["mae"], 6),
                    "Final Capital ($)":    round(res_b["final_capital"], 2),
                    "P and L ($)":          round(res_b["final_capital"] - 10000, 2),
                    "Train samples":        len(Xt),
                    "Test samples":         len(Xe),
                })
            except Exception as e:
                bench_rows.append({
                    "Experiment": label,
                    "Directional Acc %": 0,
                    "Error": str(e),
                })

            progress.progress((i + 1) / len(CONFIGS))

        status.text("Benchmark complete.")

        if bench_rows:
            df_bench = pd.DataFrame(bench_rows)

            def colour_acc(val):
                if not isinstance(val, (int, float)):
                    return ""
                if val >= 50:
                    return "background-color: #1a3a2a; color: #00d97e"
                else:
                    return "background-color: #3a1010; color: #ff4757"

            styled = (df_bench.style
                      .applymap(colour_acc, subset=["Directional Acc %"])
                      .format({
                          "Directional Acc %": "{:.1f}%",
                          "MSE":               "{:.6f}",
                          "MAE":               "{:.6f}",
                          "Final Capital ($)": "${:,.2f}",
                          "P and L ($)":       "${:+.2f}",
                      }))
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # Save benchmark JSON
            bench_path = os.path.join(HERE, "benchmark_summary.json")
            with open(bench_path, "w") as f:
                json.dump(bench_rows, f, indent=2)
            st.caption("Saved to benchmark_summary.json")


# =============================================================================
# TAB 3 - STABILITY
# =============================================================================
with tab_stability:
    st.markdown("### Rolling Window Stability")
    st.caption(
        "Tests whether directional accuracy holds across different time windows. "
        "Uses prepare_data -> train_model -> test_model -> evaluate_model."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        n_slices    = st.slider("Number of time slices", 4, 12, 6)
        slice_rows  = st.slider("Rows per slice", 500, 5000, 2000, step=500)
    with col_b:
        stab_kernel = st.selectbox("Kernel for stability test",
                                   ["rbf", "linear"], key="stab_k")
        stab_lookback = st.number_input("Lookback", min_value=1,
                                         max_value=20, value=4,
                                         key="stab_lb")

    stab_btn = st.button(
        "RUN STABILITY ANALYSIS",
        type="primary",
        use_container_width=True,
        disabled=(not MODULES_OK),
        key="stab_btn"
    )

    if stab_btn:
        if not os.path.exists(csv_path):
            st.error("CSV not found: " + csv_path)
            st.stop()

        rows_needed = slice_rows * n_slices
        Xs_all, ys_all = prepare_data(
            csv_path,
            lookback=int(stab_lookback),
            target_type="difference"
        )
        if rows_needed > len(Xs_all):
            rows_needed = len(Xs_all)

        Xs_all = Xs_all[:rows_needed]
        ys_all = ys_all[:rows_needed]

        progress_s = st.progress(0)
        stab_rows  = []

        actual_size  = len(Xs_all) // n_slices
        for i in range(n_slices):
            start = i * actual_size
            end   = start + actual_size
            Xw    = Xs_all[start:end]
            yw    = ys_all[start:end]
            sp    = int(len(Xw) * 0.7)

            if sp < 10 or (len(Xw) - sp) < 5:
                continue

            try:
                mw, sw = train_model(Xw[:sp], yw[:sp],
                                     kernel=stab_kernel)
                pw, yw_out = test_model(mw, sw, Xw[sp:], yw[sp:])
                rw = evaluate_model(yw_out, pw)
                stab_rows.append({
                    "label":                "Slice {}".format(i + 1),
                    "rows":                 "{}..{}".format(start, end),
                    "directional_accuracy": rw["directional_accuracy"],
                    "mse":                  rw["mse"],
                    "pnl":                  rw["final_capital"] - 10000,
                })
            except Exception as e:
                stab_rows.append({
                    "label": "Slice {} ERROR".format(i + 1),
                    "directional_accuracy": 0,
                    "mse": 0,
                    "pnl": 0,
                })

            progress_s.progress((i + 1) / n_slices)

        if stab_rows:
            accs = [r["directional_accuracy"] * 100 for r in stab_rows]

            # Summary stats
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Mean Accuracy",
                       "{:.1f}%".format(np.mean(accs)))
            sc2.metric("Std Dev",
                       "+/-{:.1f}%".format(np.std(accs)))
            sc3.metric("Min",
                       "{:.1f}%".format(np.min(accs)))
            sc4.metric("Max",
                       "{:.1f}%".format(np.max(accs)))

            st.markdown("---")
            st.markdown("**Accuracy Across Time Slices**")
            st.pyplot(chart_stability(stab_rows), use_container_width=True)

            st.markdown("**Detail Table**")
            df_stab = pd.DataFrame([{
                "Slice":        r["label"],
                "Row range":    r["rows"],
                "Dir Acc %":    round(r["directional_accuracy"] * 100, 1),
                "MSE":          round(r["mse"], 6),
                "P and L ($)":  round(r["pnl"], 2),
            } for r in stab_rows])
            st.dataframe(df_stab, use_container_width=True, hide_index=True)


# =============================================================================
# TAB 4 - ABOUT
# =============================================================================
with tab_about:
    st.markdown("### About This Pipeline")
    st.markdown("""
**GitHub:** https://github.com/NikkyAnil/futures-ml-pipeline

---

#### Your Module Files

| Module | File | Function called |
|--------|------|-----------------|
| Module 1 | module1_dataprep.py | prepare_data(file_path, lookback, target_type) |
| Module 2 | module2_training.py | train_model(X_train, y_train, C, epsilon, kernel) |
| Module 3 | module3_testing.py  | test_model(model, scaler, X_test, y_test) |
| Module 4 | module4_evaluation.py | evaluate_model(y_true, y_pred) |

---

#### What Each Module Returns

- **prepare_data** returns X (feature matrix), y (target vector)
- **train_model** returns model (SVR), scaler (StandardScaler)
- **test_model** returns predictions, y_test
- **evaluate_model** returns dict with mse, mae, directional_accuracy, final_capital, capital_history

---

#### How to Run

Open IntelliJ IDEA Terminal and run:

```
pip install streamlit scikit-learn pandas numpy matplotlib
streamlit run dashboard.py
```

Browser opens at http://localhost:8501

---

#### Pipeline Flow

```
TY.csv
   |
   v
prepare_data()    -- builds rolling window features
   |
   v
train_model()     -- trains SVR
   |
   v
test_model()      -- generates predictions
   |
   v
evaluate_model()  -- computes MSE, accuracy, AUM
```
    """)
