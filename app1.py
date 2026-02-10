import streamlit as st
import pandas as pd
from evaluation.evaluator import evaluate_sample
from reports.report_generator import generate_report
from PIL import Image
import os

st.set_page_config(page_title="Hallucination Detection", layout="wide")
st.title("🧠 Image Prompt Adherence & Hallucination Detection")

# -----------------------------
# DATASET SOURCE SELECTION
# -----------------------------
st.sidebar.header("Dataset Source")

dataset_mode = st.sidebar.radio(
    "Choose dataset source",
    options=["Use existing local dataset", "Upload new dataset"]
)

df = None

# -----------------------------
# OPTION 1: USE EXISTING DATASET
# -----------------------------
if dataset_mode == "Use existing local dataset":
    dataset_path = "data/dataset.csv"

    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        st.success("✅ Loaded existing dataset from data/dataset.csv")
        st.write(f"Total samples: {len(df)}")
    else:
        st.error("❌ data/dataset.csv not found. Please prepare dataset first.")

# -----------------------------
# OPTION 2: UPLOAD DATASET
# -----------------------------
else:
    uploaded_csv = st.file_uploader("Upload Dataset CSV", type=["csv"])
    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.success("✅ Uploaded dataset loaded successfully")
        st.write(f"Total samples: {len(df)}")

# -----------------------------
# RUN EVALUATION
# -----------------------------
if df is not None:

    if st.button("▶ Run Evaluation"):
        results = []

        progress = st.progress(0)
        total = len(df)

        for idx, row in df.iterrows():
            res = evaluate_sample(row["prompt"], row["image_path"])
            results.append(res)
            progress.progress((idx + 1) / total)

        report_df = generate_report(results)
        st.session_state["report_df"] = report_df
        st.success("✅ Evaluation completed")

# -----------------------------
# SHOW RESULTS
# -----------------------------
if "report_df" in st.session_state:
    report_df = st.session_state["report_df"]

    st.subheader("📊 Evaluation Results")
    st.dataframe(report_df)

    st.subheader("🔍 Sample Viewer")

    sample_idx = st.selectbox(
        "Select sample",
        options=range(len(report_df))
    )

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            Image.open(df.iloc[sample_idx]["image_path"]),
            caption="Generated Image",
            use_column_width=True
        )

    with col2:
        st.markdown(f"**Prompt:** {report_df.iloc[sample_idx]['prompt']}")
        st.markdown(f"**Image Caption:** {report_df.iloc[sample_idx]['caption']}")

        st.metric(
            "Prompt Adherence Score",
            f"{report_df.iloc[sample_idx]['adherence_score']}%"
        )
        st.metric(
            "CLIP Similarity",
            report_df.iloc[sample_idx]['clip_score']
        )

        st.write("✔ Matched Objects:", report_df.iloc[sample_idx]["matched"])
        st.write("❌ Missing Objects:", report_df.iloc[sample_idx]["missing"])
        st.write("⚠ Extra Objects:", report_df.iloc[sample_idx]["extra"])

    st.download_button(
        "⬇ Download Report CSV",
        report_df.to_csv(index=False),
        file_name="hallucination_report.csv",
        mime="text/csv"
    )
