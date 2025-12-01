import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(page_title="Survey Statistics Analyzer", layout="wide")

st.title("📊 Survey Statistics Analyzer")
st.write("Upload your CSV/Excel survey dataset to generate statistical analysis, tables, and charts.")

# -----------------------------
# File Upload Section
# -----------------------------
uploaded_file = st.file_uploader("Upload your dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("File uploaded successfully!")
    st.subheader("Preview of Data")
    st.dataframe(df.head())

    # Select numerical columns
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        st.error("No numerical columns found. Ensure your Likert-scale items are numeric (1-5).")
    else:
        # --------------------------------
        # Sidebar: Select variable analysis
        # --------------------------------
        st.sidebar.header("Analysis Settings")
        selected_col = st.sidebar.selectbox("Select variable", numeric_df.columns)

        col_data = numeric_df[selected_col]
        st.header(f"Statistics for: **{selected_col}**")

        # -----------------------------
        # Descriptive Statistics
        # -----------------------------
        st.subheader("Descriptive Statistics")

        stats = {
            "Mean": col_data.mean(),
            "Median": col_data.median(),
            "Mode": col_data.mode().iloc[0] if not col_data.mode().empty else "N/A",
            "Minimum": col_data.min(),
            "Maximum": col_data.max(),
            "Standard Deviation": col_data.std()
        }

        st.table(pd.DataFrame(stats, index=[selected_col]))

        # -----------------------------
        # Frequency Table
        # -----------------------------
        st.subheader("Frequency & Percentage Table")

        freq_table = col_data.value_counts().sort_index().to_frame("Frequency")
        freq_table["Percentage"] = round((freq_table["Frequency"] / freq_table["Frequency"].sum()) * 100, 2)

        st.table(freq_table)

        # -----------------------------
        # Histogram
        # -----------------------------
        st.subheader("Histogram")

        fig, ax = plt.subplots()
        ax.hist(col_data, bins=10)
        ax.set_xlabel(selected_col)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram of {selected_col}")
        st.pyplot(fig)

        # -----------------------------
        # Boxplot
        # -----------------------------
        st.subheader("Boxplot")

        fig2, ax2 = plt.subplots()
        ax2.boxplot(col_data, vert=False)
        ax2.set_title(f"Boxplot of {selected_col}")
        st.pyplot(fig2)

else:
    st.info("Please upload a dataset to begin.")
