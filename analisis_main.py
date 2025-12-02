# streamlit_app_fixed.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, chi2_contingency, normaltest
import nltk
from nltk.corpus import stopwords
import string
from collections import Counter
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader

# -------------------------
# Page style - Orange bg
# -------------------------
st.set_page_config(page_title="Survey Data Analyzer", layout="wide")
st.markdown(
    """
    <style>
        .stApp { background-color: #FFA64D !important; }
        /* Improve readability on orange background */
        .css-1d391kg { color: #000 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# TEXTS dictionary (minimal)
# -------------------------
TEXTS = {
    "EN": {
        "title": "Survey Data Analysis App",
        "upload": "Upload CSV / XLS / XLSX",
        "preview": "Data preview (max 1000 rows)",
        "rows": "Rows",
        "cols": "Columns",
        "num_cols": "Numeric columns",
        "non_num_cols": "Non-numeric columns",
        "tab1": "Descriptive Stats",
        "tab2": "Visualizations",
        "tab3": "Correlations & Tests",
        "tab4": "Text Processing",
        "select_numeric": "Select numeric column",
        "select_two_numeric": "Select two numeric columns",
        "select_two_categorical": "Select two categorical columns",
        "hist": "Histogram",
        "box": "Boxplot",
        "scatter": "Scatter Plot",
        "barchart": "Bar Chart (Top 20)",
        "generate_pdf": "Generate PDF Report",
        "download_pdf": "Download PDF",
        "dark_mode": "Dark Mode",
        "language": "Language",
        "insufficient": "Insufficient data for this operation"
    },
    "ID": {
        "title": "Aplikasi Analisis Data Survei",
        "upload": "Unggah CSV / XLS / XLSX",
        "preview": "Pratinjau data (maks 1000 baris)",
        "rows": "Baris",
        "cols": "Kolom",
        "num_cols": "Kolom numerik",
        "non_num_cols": "Kolom non-numerik",
        "tab1": "Statistik Deskriptif",
        "tab2": "Visualisasi",
        "tab3": "Korelasi & Uji",
        "tab4": "Pemrosesan Teks",
        "select_numeric": "Pilih kolom numerik",
        "select_two_numeric": "Pilih dua kolom numerik",
        "select_two_categorical": "Pilih dua kolom kategorikal",
        "hist": "Histogram",
        "box": "Boxplot",
        "scatter": "Scatter Plot",
        "barchart": "Diagram Batang (Top 20)",
        "generate_pdf": "Buat Laporan PDF",
        "download_pdf": "Unduh PDF",
        "dark_mode": "Mode Gelap",
        "language": "Bahasa",
        "insufficient": "Data tidak cukup untuk operasi ini"
    }
    # For brevity only EN and ID provided; fallback to EN if missing.
}

def get_text(key: str) -> str:
    lang = st.session_state.get("language", "EN")
    if lang not in TEXTS:
        lang = "EN"
    return TEXTS[lang].get(key, TEXTS["EN"].get(key, key))

# -------------------------
# NLTK stopwords init
# -------------------------
try:
    _ = stopwords.words("english")
except Exception:
    try:
        nltk.download("stopwords", quiet=True)
    except Exception:
        pass

# -------------------------
# Helpers
# -------------------------
def safe_load_data(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return None
    name = uploaded.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded)
        else:
            return pd.read_excel(uploaded)
    except Exception as e:
        # try pandas auto
        uploaded.seek(0)
        try:
            return pd.read_table(uploaded)
        except Exception as e2:
            raise e

def preprocess_text_series(series: pd.Series):
    stop = set()
    try:
        stop = set(stopwords.words("english"))
    except Exception:
        stop = set()
    clean = series.dropna().astype(str).str.lower()
    translator = str.maketrans("", "", string.punctuation)
    clean = clean.apply(lambda x: x.translate(translator))
    tokens = clean.apply(lambda x: [w for w in x.split() if w and w not in stop])
    counter = Counter([w for sub in tokens for w in sub])
    return tokens, counter

def fig_to_image_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf

def build_survey_report_pdf(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []

    # Metadata
    flow.append(Paragraph("Survey Data Report", styles["Title"]))
    flow.append(Spacer(1, 8))
    flow.append(Paragraph(f"Rows: {len(df)} &nbsp;&nbsp; Columns: {len(df.columns)}", styles["Normal"]))
    flow.append(Spacer(1, 8))

    num_df = df.select_dtypes(include=np.number)
    cat_df = df.select_dtypes(exclude=np.number)

    # Numeric columns: stats + plots
    for col in num_df.columns:
        series = num_df[col].dropna()
        flow.append(Paragraph(f"<b>{col}</b>", styles["Heading2"]))
        if series.empty:
            flow.append(Paragraph(get_text("insufficient"), styles["Normal"]))
            flow.append(Spacer(1, 6))
            continue
        desc = series.describe()
        flow.append(Paragraph(f"Count: {int(desc['count'])}, Mean: {desc['mean']:.4f}, Median: {series.median():.4f}, Std: {desc['std']:.4f}", styles["Normal"]))
        flow.append(Paragraph(f"Min: {desc['min']}, Max: {desc['max']}", styles["Normal"]))
        # Normaltest safe
        nt_stat = nt_p = None
        try:
            if len(series) >= 8:
                nt_stat, nt_p = normaltest(series)
                flow.append(Paragraph(f"Normaltest stat={nt_stat:.4f}, p={nt_p:.6f}", styles["Normal"]))
            else:
                flow.append(Paragraph("Normaltest: not enough samples (need >=8)", styles["Normal"]))
        except Exception:
            flow.append(Paragraph("Normaltest: error computing", styles["Normal"]))

        flow.append(Spacer(1, 6))

        # Histogram
        try:
            fig1, ax1 = plt.subplots(figsize=(6,3))
            sns.histplot(series, kde=True, ax=ax1)
            buf1 = fig_to_image_bytes(fig1)
            flow.append(Image(ImageReader(buf1), width=400, height=180))
            flow.append(Spacer(1, 6))
        except Exception:
            pass

        # Boxplot
        try:
            fig2, ax2 = plt.subplots(figsize=(6,2))
            sns.boxplot(x=series, ax=ax2)
            buf2 = fig_to_image_bytes(fig2)
            flow.append(Image(ImageReader(buf2), width=400, height=150))
            flow.append(Spacer(1, 12))
        except Exception:
            pass

    # Correlation matrix (pearson) as table if enough numeric cols
    if num_df.shape[1] >= 2:
        corr = num_df.corr(method="pearson")
        flow.append(Paragraph("Pearson Correlation Matrix", styles["Heading2"]))
        headers = [""] + list(corr.columns)
        table_data = [headers]
        for idx in corr.index:
            row = [idx] + [f"{corr.loc[idx, c]:.3f}" for c in corr.columns]
            table_data.append(row)
        try:
            flow.append(Table(table_data))
            flow.append(Spacer(1, 12))
        except Exception:
            pass

    # Categorical top 10
    if not cat_df.empty:
        flow.append(Paragraph("Categorical Frequency (Top 10)", styles["Heading2"]))
        for col in cat_df.columns:
            flow.append(Paragraph(f"<b>{col}</b>", styles["Heading3"]))
            vc = df[col].fillna("(Missing)").astype(str).value_counts().head(10)
            table_data = [["Category", "Count"]]
            for k, v in vc.items():
                table_data.append([str(k), str(int(v))])
            flow.append(Table(table_data))
            flow.append(Spacer(1, 8))

    # Text processing top words
    text_cols = [c for c in df.columns if df[c].dtype == object]
    if text_cols:
        flow.append(Paragraph("Text Processing Summary (Top words)", styles["Heading2"]))
        for col in text_cols:
            flow.append(Paragraph(f"<b>{col}</b>", styles["Heading3"]))
            try:
                _, counter = preprocess_text_series(df[col])
                table_data = [["Word", "Count"]]
                for w, cnt in counter.most_common(10):
                    table_data.append([w, str(cnt)])
                flow.append(Table(table_data))
                flow.append(Spacer(1, 8))
            except Exception:
                flow.append(Paragraph("Error processing text column", styles["Normal"]))

    # Build PDF
    try:
        doc.build(flow)
        buffer.seek(0)
        pdf_bytes = buffer.getvalue()
    finally:
        buffer.close()
    return pdf_bytes

# -------------------------
# UI Top controls
# -------------------------
st.session_state.setdefault("language", "EN")
if "last_pdf" not in st.session_state:
    st.session_state["last_pdf"] = None

col_a, col_b = st.columns([1, 1])
with col_a:
    st.session_state["dark_mode"] = st.checkbox(get_text("dark_mode"), value=st.session_state.get("dark_mode", False))
with col_b:
    st.session_state["language"] = st.radio(get_text("language"), options=["EN", "ID"], index=["EN", "ID"].index(st.session_state.get("language", "EN")), horizontal=True)

st.title(get_text("title"))

uploaded = st.file_uploader(get_text("upload"), type=["csv", "xls", "xlsx"])
df = None
if uploaded is not None:
    try:
        df = safe_load_data(uploaded)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        df = None

if df is None:
    st.info("No data loaded. Please upload a file.")
    st.stop()

# Preview
st.subheader(get_text("preview"))
st.dataframe(df.head(1000))

# Summary metrics
num_df = df.select_dtypes(include=np.number)
cat_df = df.select_dtypes(exclude=np.number)
col1, col2, col3, col4 = st.columns(4)
col1.metric(get_text("rows"), df.shape[0])
col2.metric(get_text("cols"), df.shape[1])
col3.metric(get_text("num_cols"), num_df.shape[1])
col4.metric(get_text("non_num_cols"), cat_df.shape[1])

# Tabs (single-page but organized)
tab1, tab2, tab3, tab4 = st.tabs([get_text("tab1"), get_text("tab2"), get_text("tab3"), get_text("tab4")])

# -------------------------
# Tab 1 - Descriptive Stats
# -------------------------
with tab1:
    st.header(get_text("tab1"))
    if num_df.shape[1] == 0:
        st.info(get_text("insufficient"))
    else:
        sel_col = st.selectbox(get_text("select_numeric"), options=list(num_df.columns))
        if sel_col:
            series = pd.to_numeric(df[sel_col], errors="coerce").dropna()
            if series.empty:
                st.write(get_text("insufficient"))
            else:
                st.write(series.describe().to_frame().T)
                # normaltest safe
                if len(series) >= 8:
                    try:
                        stat, p = normaltest(series)
                        st.write(f"Normality test: stat={stat:.4f}, p={p:.6f}")
                        st.write("Normal" if p > 0.05 else "Not normal")
                    except Exception:
                        st.write("Normality test: error computing")
                else:
                    st.write("Normality test: need at least 8 non-missing values")

                # plots
                fig, ax = plt.subplots()
                sns.histplot(series, kde=True, ax=ax)
                ax.set_title(f"{get_text('hist')}: {sel_col}")
                st.pyplot(fig)
                plt.close(fig)

                fig2, ax2 = plt.subplots()
                sns.boxplot(x=series, ax=ax2)
                ax2.set_title(f"{get_text('box')}: {sel_col}")
                st.pyplot(fig2)
                plt.close(fig2)

    # Frequency tables for categorical
    st.subheader("Frequency tables")
    if cat_df.shape[1] == 0:
        st.info(get_text("insufficient"))
    else:
        for c in cat_df.columns:
            st.write(f"Column: {c}")
            vc = df[c].fillna("(Missing)").astype(str).value_counts()
            pct = (vc / vc.sum()) * 100
            out = pd.DataFrame({"count": vc, "percent": pct})
            st.dataframe(out.head(50))

# -------------------------
# Tab 2 - Visualizations
# -------------------------
with tab2:
    st.header(get_text("tab2"))
    # Histogram & boxplot
    if num_df.shape[1] > 0:
        sel_hist = st.selectbox(get_text("hist"), options=list(num_df.columns), key="hist_sel")
        if sel_hist:
            s = pd.to_numeric(df[sel_hist], errors="coerce").dropna()
            if not s.empty:
                fig, ax = plt.subplots()
                sns.histplot(s, kde=True, ax=ax)
                ax.set_title(f"{get_text('hist')}: {sel_hist}")
                st.pyplot(fig)
                plt.close(fig)

                fig2, ax2 = plt.subplots()
                sns.boxplot(x=s, ax=ax2)
                ax2.set_title(f"{get_text('box')}: {sel_hist}")
                st.pyplot(fig2)
                plt.close(fig2)
            else:
                st.write(get_text("insufficient"))

    # Scatter plot (use synchronized dropna)
    st.subheader(get_text("scatter"))
    if num_df.shape[1] >= 2:
        sx = st.selectbox("X (numeric)", options=list(num_df.columns), key="scatter_x")
        sy = st.selectbox("Y (numeric)", options=list(num_df.columns), index=1 if len(num_df.columns) > 1 else 0, key="scatter_y")
        if sx and sy:
            clean = df[[sx, sy]].dropna()
            if clean.shape[0] < 2:
                st.write(get_text("insufficient"))
            else:
                fig, ax = plt.subplots()
                ax.scatter(clean[sx], clean[sy], alpha=0.7)
                ax.set_xlabel(sx); ax.set_ylabel(sy)
                ax.set_title(f"{sx} vs {sy}")
                st.pyplot(fig)
                plt.close(fig)
    else:
        st.info("Need at least two numeric columns for scatter")

    # Bar chart for categorical
    st.subheader(get_text("barchart"))
    if cat_df.shape[1] > 0:
        sel_cat = st.selectbox(get_text("barchart"), options=list(cat_df.columns), key="bar_cat")
        if sel_cat:
            vc = df[sel_cat].fillna("(Missing)").astype(str).value_counts().head(20)
            fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(vc))))
            sns.barplot(x=vc.values, y=vc.index, ax=ax)
            ax.set_xlabel("Count"); ax.set_ylabel(sel_cat)
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("No categorical columns for bar chart")

# -------------------------
# Tab 3 - Correlations & Tests
# -------------------------
with tab3:
    st.header(get_text("tab3"))

    # Pearson & Spearman (safe)
    st.subheader("Pearson & Spearman")
    if num_df.shape[1] >= 2:
        a = st.selectbox(get_text("select_two_numeric"), options=list(num_df.columns), key="corr_a")
        b = st.selectbox("Select second numeric", options=list(num_df.columns), index=1 if len(num_df.columns) > 1 else 0, key="corr_b")
        if a and b:
            clean = df[[a, b]].dropna()
            if clean.shape[0] < 2:
                st.write(get_text("insufficient"))
            else:
                try:
                    r, p = pearsonr(clean[a], clean[b])
                    st.write(f"Pearson r = {r:.4f}, p = {p:.6f}")
                except Exception:
                    st.write("Pearson: error computing")

                try:
                    rho, p2 = spearmanr(clean[a], clean[b])
                    st.write(f"Spearman rho = {rho:.4f}, p = {p2:.6f}")
                except Exception:
                    st.write("Spearman: error computing")
    else:
        st.info("Need at least two numeric columns for correlations")

    # Chi-square test (safe)
    st.subheader(get_text("chi"))
    if cat_df.shape[1] >= 2:
        ca = st.selectbox(get_text("select_two_categorical"), options=list(cat_df.columns), key="chi_a")
        cb = st.selectbox("Second categorical", options=list(cat_df.columns), index=1 if len(cat_df.columns) > 1 else 0, key="chi_b")
        if ca and cb:
            ct = pd.crosstab(df[ca].fillna("(Missing)").astype(str), df[cb].fillna("(Missing)").astype(str))
            if ct.size == 0 or ct.values.sum() == 0 or ct.shape[0] < 2 or ct.shape[1] < 2:
                st.write(get_text("insufficient"))
            else:
                try:
                    chi2, pval, dof, expected = chi2_contingency(ct)
                    st.write(f"chi2 = {chi2:.4f}, p = {pval:.6f}, dof = {dof}")
                    st.write("Observed:")
                    st.dataframe(ct)
                    st.write("Expected:")
                    st.dataframe(pd.DataFrame(expected, index=ct.index, columns=ct.columns))
                except Exception as e:
                    st.write(f"Chi-square error: {e}")
    else:
        st.info("Need at least two categorical columns for chi-square")

    # Correlation matrix
    if num_df.shape[1] > 1:
        st.subheader("Pearson Correlation Matrix")
        st.dataframe(num_df.corr())
    else:
        st.info("Correlation matrix requires at least 2 numeric columns")

# -------------------------
# Tab 4 - Text Processing
# -------------------------
with tab4:
    st.header(get_text("tab4"))
    text_cols = [c for c in df.columns if df[c].dtype == object]
    if not text_cols:
        st.info("No text columns found")
    else:
        tcol = st.selectbox("Select text column", options=text_cols)
        if tcol:
            tokens, counter = preprocess_text_series(df[tcol])
            st.subheader("Sample tokens (first 5 rows)")
            try:
                st.write(tokens.head(5).to_dict())
            except Exception:
                st.write(list(tokens)[:5])
            st.subheader("Top 10 words")
            st.write(counter.most_common(10))

# -------------------------
# PDF generation & download
# -------------------------
st.sidebar.header("Export")
if st.sidebar.button(get_text("generate_pdf")):
    with st.spinner("Building PDF..."):
        try:
            pdf_bytes = build_survey_report_pdf(df)
            st.session_state["last_pdf"] = pdf_bytes
            st.success("PDF generated")
        except Exception as e:
            st.error(f"Error generating PDF: {e}")
            st.session_state["last_pdf"] = None

if st.session_state.get("last_pdf"):
    st.sidebar.download_button(
        label=get_text("download_pdf"),
        data=st.session_state["last_pdf"],
        file_name="survey_report.pdf",
        mime="application/pdf",
    )
