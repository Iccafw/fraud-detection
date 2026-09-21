import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Demo Deteksi Fraud Kartu Kredit", page_icon="💳", layout="wide")


# ---------- Load model & data ----------
@st.cache_resource
def load_model():
    bundle = joblib.load("fraud_model.joblib")
    return bundle["model"], float(bundle["threshold"]), list(bundle["features"])


@st.cache_data
def load_samples():
    return pd.read_csv("sample_transactions.csv")


try:
    model, default_thr, features = load_model()
    samples = load_samples()
except FileNotFoundError:
    st.error(
        "File `fraud_model.joblib` atau `sample_transactions.csv` tidak ditemukan. "
        "Letakkan keduanya di folder yang sama dengan `app.py`."
    )
    st.stop()


def predict_proba(df_feat: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df_feat[features])[:, 1]


# ---------- Sidebar ----------
st.sidebar.header("Pengaturan")
threshold = st.sidebar.slider(
    "Threshold fraud",
    min_value=0.01, max_value=0.99,
    value=float(np.clip(default_thr, 0.01, 0.99)), step=0.01,
    help="Transaksi dengan probabilitas fraud di atas nilai ini ditandai FRAUD.",
)
st.sidebar.caption(
    f"Threshold bawaan ({default_thr:.2f}) dipilih dari analisis biaya bisnis di notebook. "
    "Threshold lebih rendah: lebih banyak fraud tertangkap, tapi lebih banyak false alarm."
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Tentang model**  \n"
    "XGBoost dilatih pada dataset *Credit Card Fraud Detection* (ULB, 2013). "
    "Fitur V1-V28 adalah hasil PCA yang sudah dianonimkan, jadi artinya tidak diketahui.  \n\n"
    "⚠️ Ini demo portofolio, bukan sistem produksi."
)


# ---------- Tampilan hasil ----------
def show_result(row: pd.DataFrame, actual=None):
    proba = float(predict_proba(row)[0])
    is_fraud = proba >= threshold

    cols = st.columns(4 if actual is not None else 3)
    cols[0].metric("Amount", f"{float(row['Amount'].iloc[0]):,.2f}")
    cols[1].metric("Probabilitas fraud", f"{proba:.1%}")
    cols[2].metric("Keputusan model", "🚨 FRAUD" if is_fraud else "✅ NORMAL")
    if actual is not None:
        cols[3].metric("Label sebenarnya", "🚨 FRAUD" if actual == 1 else "✅ NORMAL")

    st.progress(float(np.clip(proba, 0, 1)))

    if actual is not None:
        if is_fraud == bool(actual):
            st.success("Prediksi model **benar**.")
        elif is_fraud and not actual:
            st.warning("**False alarm**: transaksi normal, tapi ditandai fraud.")
        else:
            st.error("**Fraud lolos**: transaksi fraud, tapi tidak terdeteksi.")


st.title("💳 Demo Deteksi Penipuan Kartu Kredit")
st.write("Pilih transaksi contoh, atur nilainya sendiri, atau upload CSV, lalu lihat apakah model menganggapnya fraud.")

tab1, tab2, tab3 = st.tabs(["🎲 Transaksi contoh", "🎛️ Input manual", "📄 Upload CSV"])

# ---------- Tab 1: transaksi contoh ----------
with tab1:
    st.caption("Transaksi diambil dari data uji (transaksi yang tidak dilihat model saat training).")

    if "sample_idx" not in st.session_state:
        st.session_state.sample_idx = None

    def pick(kind: str):
        if kind == "fraud":
            pool = samples[samples["Class"] == 1]
        elif kind == "normal":
            pool = samples[samples["Class"] == 0]
        else:
            pool = samples
        st.session_state.sample_idx = int(pool.sample(1).index[0])

    c1, c2, c3 = st.columns(3)
    c1.button("🎲 Transaksi acak", on_click=pick, args=("random",), use_container_width=True)
    c2.button("🚨 Transaksi fraud", on_click=pick, args=("fraud",), use_container_width=True)
    c3.button("✅ Transaksi normal", on_click=pick, args=("normal",), use_container_width=True)

    if st.session_state.sample_idx is None:
        st.info("Klik salah satu tombol di atas untuk mengambil transaksi.")
    else:
        row = samples.loc[[st.session_state.sample_idx]]
        show_result(row[features], actual=int(row["Class"].iloc[0]))
        with st.expander("Lihat semua fitur transaksi"):
            st.dataframe(row[features].T.rename(columns={row.index[0]: "nilai"}))

# ---------- Tab 2: input manual ----------
with tab2:
    st.caption(
        "Karena V1-V28 dianonimkan, di sini kamu mengatur `Amount` dan 6 fitur terpenting menurut model. "
        "Fitur lainnya diisi dengan median transaksi normal."
    )

    normal_median = samples[samples["Class"] == 0][features].median()
    fraud_mean = samples[samples["Class"] == 1][features].mean()

    importance = pd.Series(model.feature_importances_, index=features)
    top_feats = [f for f in importance.sort_values(ascending=False).index if f != "Amount"][:6]

    bounds = {f: (float(samples[f].quantile(0.005)), float(samples[f].quantile(0.995))) for f in top_feats}

    # nilai awal widget
    if "man_Amount" not in st.session_state:
        st.session_state["man_Amount"] = float(round(normal_median["Amount"], 2))
    for f in top_feats:
        if f"man_{f}" not in st.session_state:
            st.session_state[f"man_{f}"] = float(np.clip(normal_median[f], *bounds[f]))

    def fill_profile(profile: pd.Series):
        st.session_state["man_Amount"] = float(round(profile["Amount"], 2))
        for f in top_feats:
            st.session_state[f"man_{f}"] = float(np.clip(profile[f], *bounds[f]))

    b1, b2 = st.columns(2)
    b1.button("Isi dengan profil transaksi normal", on_click=fill_profile, args=(normal_median,), use_container_width=True)
    b2.button(
        "Isi dengan profil fraud rata-rata", on_click=fill_profile, args=(fraud_mean,),
        use_container_width=True, disabled=fraud_mean.isna().any(),
    )

    st.number_input("Amount", min_value=0.0, step=1.0, key="man_Amount")
    sl_cols = st.columns(3)
    for i, f in enumerate(top_feats):
        lo, hi = bounds[f]
        sl_cols[i % 3].slider(f, lo, hi, step=(hi - lo) / 200, key=f"man_{f}")

    manual = normal_median.copy()
    manual["Amount"] = st.session_state["man_Amount"]
    for f in top_feats:
        manual[f] = st.session_state[f"man_{f}"]

    st.markdown("---")
    show_result(pd.DataFrame([manual]))

# ---------- Tab 3: upload CSV ----------
with tab3:
    st.caption(f"CSV harus berisi kolom: {', '.join(features[:4])}, ... ({len(features)} kolom fitur).")
    st.download_button(
        "Download contoh CSV (50 transaksi)",
        samples.head(50)[features].to_csv(index=False).encode("utf-8"),
        file_name="contoh_transaksi.csv", mime="text/csv",
    )

    up = st.file_uploader("Upload CSV transaksi", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        missing = [c for c in features if c not in data.columns]
        if missing:
            st.error(f"Kolom tidak ada: {', '.join(missing)}")
        else:
            out = data.copy()
            out["P(fraud)"] = predict_proba(out)
            out["Prediksi"] = np.where(out["P(fraud)"] >= threshold, "FRAUD", "NORMAL")
            n_fraud = int((out["Prediksi"] == "FRAUD").sum())
            st.write(f"**{len(out):,}** transaksi diproses, **{n_fraud:,}** ditandai FRAUD.")
            out = out.sort_values("P(fraud)", ascending=False)
            st.dataframe(out.head(500), use_container_width=True)
            st.download_button(
                "Download hasil (CSV)", out.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi.csv", mime="text/csv",
            )
