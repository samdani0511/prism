import streamlit as st
import pandas as pd
from evaluation.evaluator import evaluate_sample
from reports.report_generator import generate_report
from PIL import Image

st.set_page_config(page_title="Hallucination Detection", layout="wide")

st.title("🧠 Image Prompt Adherence & Hallucination Detection")

uploaded_csv = st.file_uploader("Upload Dataset CSV", type=["csv"])

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    results = []

    for _, row in df.iterrows():
        with st.spinner(f"Evaluating sample {row['id']}..."):
            res = evaluate_sample(row["prompt"], row["image_path"])
            results.append(res)

    report_df = generate_report(results)

    st.subheader("📊 Overall Metrics")
    st.dataframe(report_df)

    st.subheader("🔍 Sample Viewer")
    idx = st.selectbox("Select Sample", range(len(report_df)))

    col1, col2 = st.columns(2)

    with col1:
        st.image(Image.open(df.iloc[idx]["image_path"]), caption="Generated Image")

    with col2:
        st.markdown(f"**Prompt:** {report_df.iloc[idx]['prompt']}")
        st.markdown(f"**Image Caption:** {report_df.iloc[idx]['caption']}")
        st.metric("Prompt Adherence Score", f"{report_df.iloc[idx]['adherence_score']}%")
        st.metric("CLIP Similarity", report_df.iloc[idx]['clip_score'])

        st.write("✔ Matched Objects:", report_df.iloc[idx]["matched"])
        st.write("❌ Missing Objects:", report_df.iloc[idx]["missing"])
        st.write("⚠ Extra Objects:", report_df.iloc[idx]["extra"])

    st.download_button(
        "⬇ Download Report CSV",
        report_df.to_csv(index=False),
        file_name="hallucination_report.csv"
    )
