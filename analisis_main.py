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

# ============================
# Background ORANGE
# ============================
st.markdown("""
<style>
    .stApp {
        background-color: #FFA64D !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# TEXT DICTIONARY
# ============================
TEXTS = {
    "EN": {
        "title": "Survey Data Analysis App",
        "upload": "Upload CSV/XLS/XLSX File",
        "preview": "Data Preview",
        "rows": "Number of Rows",
        "cols": "Number of Columns",
        "num_cols": "Numeric Columns",
        "non_num_cols": "Non-numeric Columns",
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
        "download_pdf": "Download PDF Report",
        "generate_pdf": "Generate Report PDF"
    },
    "ID": {
        "title": "Aplikasi Analisis Data Survei",
        "upload": "Upload File CSV/XLS/XLSX",
        "preview": "Preview Data",
        "rows": "Jumlah Baris",
        "cols": "Jumlah Kolom",
        "num_cols": "Kolom Numerik",
        "non_num_cols": "Kolom Non-numerik",
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
        "download_pdf": "Download Laporan PDF",
        "generate_pdf": "Generate PDF"
    },
    "JP": {
        "title": "アンケートデータ分析アプリ",
        "upload": "CSV/XLS/XLSX ファイルをアップロード",
        "preview": "データプレビュー",
        "rows": "行数",
        "cols": "列数",
        "num_cols": "数値列",
        "non_num_cols": "非数値列",
        "tab1": "記述統計",
        "tab2": "ビジュアライゼーション",
        "tab3": "相関と検定",
        "tab4": "テキスト処理",
        "select_numeric": "数値列を選択",
        "select_two_numeric": "数値列2つを選択",
        "select_two_categorical": "カテゴリ列2つを選択",
        "hist": "ヒストグラム",
        "box": "箱ひげ図",
        "scatter": "散布図",
        "barchart": "棒グラフ（上位20）",
        "download_pdf": "PDF レポートをダウンロード",
        "generate_pdf": "PDF を生成"
    },
    "KR": {
        "title": "설문 데이터 분석 앱",
        "upload": "CSV/XLS/XLSX 파일 업로드",
        "preview": "데이터 미리보기",
        "rows": "행 수",
        "cols": "열 수",
        "num_cols": "숫자 열",
        "non_num_cols": "비숫자 열",
        "tab1": "기술 통계",
        "tab2": "시각화",
        "tab3": "상관 및 검정",
        "tab4": "텍스트 처리",
        "select_numeric": "숫자 열 선택",
        "select_two_numeric": "숫자 열 2개 선택",
        "select_two_categorical": "범주형 열 2개 선택",
        "hist": "히스토그램",
        "box": "박스플롯",
        "scatter": "산점도",
        "barchart": "막대 차트 (상위 20)",
        "download_pdf": "PDF 보고서 다운로드",
        "generate_pdf": "PDF 생성"
    },
    "CN": {
        "title": "调查数据分析应用",
        "upload": "上传 CSV/XLS/XLSX 文件",
        "preview": "数据预览",
        "rows": "行数",
        "cols": "列数",
        "num_cols": "数值列",
        "non_num_cols": "非数值列",
        "tab1": "描述统计",
        "tab2": "可视化",
        "tab3": "相关与检验",
        "tab4": "文本处理",
        "select_numeric": "选择数值列",
        "select_two_numeric": "选择两个数值列",
        "select_two_categorical": "选择两个分类列",
        "hist": "直方图",
        "box": "箱线图",
        "scatter": "散点图",
        "barchart": "柱状图（前20）",
        "download_pdf": "下载 PDF 报告",
        "generate_pdf": "生成 PDF"
    }
}

def get_text(key):
    lang = st.session_state.get("language", "EN")
    return TEXTS.get(lang, TEXTS["EN"]).get(key, TEXTS["EN"].get(key, key))

# ============================
# NLTK stopwords init
# ============================
try:
    stopwords.words("english")
except:
    nltk.download("stopwords")

# ============================
# Functions
# ============================
def load_data(uploaded):
    if uploaded.name.endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)

def preprocess_text_series(series):
    stop = set(stopwords.words("english"))
    clean = series.dropna().astype(str).str.lower()
    clean = clean.apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))
    tokens = clean.apply(lambda x: [w for w in x.split() if w not in stop])
    all_words = [w for sub in tokens for w in sub]
    counter = Counter(all_words)
    return tokens, counter

def build_survey_report_pdf(df):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []

    # Metadata
    text = f"Dataset Report\nRows: {len(df)}\nColumns: {len(df.columns)}"
    flow.append(Paragraph(text.replace("\n", "<br/>"), styles["Normal"]))
    flow.append(Spacer(1, 12))

    num_df = df.select_dtypes(include=np.number)
    cat_df = df.select_dtypes(exclude=np.number)

    # Numeric stats
    for col in num_df.columns:
        desc = num_df[col].describe()
        stat, p = normaltest(num_df[col].dropna())
        para = f"<b>{col}</b><br/>Mean={desc['mean']:.3f}, Std={desc['std']:.3f}<br/>Min={desc['min']}, Max={desc['max']}<br/>Normaltest: stat={stat:.3f}, p={p:.4f}"
        flow.append(Paragraph(para, styles["Normal"]))
        flow.append(Spacer(1, 12))

        # Histogram img
        fig, ax = plt.subplots()
        sns.histplot(num_df[col], ax=ax)
        img = BytesIO()
        fig.savefig(img, format="png", bbox_inches="tight")
        plt.close(fig)
        img.seek(0)
        flow.append(Image(ImageReader(img), width=250, height=200))
        flow.append(Spacer(1, 12))

    doc.build(flow)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

# ============================
# UI TOP BAR
# ============================
st.session_state.setdefault("language", "EN")

c1, c2 = st.columns([1,1])
with c1:
    dark = st.checkbox("Dark Mode", value=False)

with c2:
    st.session_state["language"] = st.radio("Language", ["EN","ID","JP","KR","CN"], horizontal=True)

# ============================
# TITLE
# ============================
st.title(get_text("title"))

# ============================
# FILE UPLOAD
# ============================
uploaded = st.file_uploader(get_text("upload"), type=["csv","xls","xlsx"])

if uploaded:
    df = load_data(uploaded)

    # Preview
    st.subheader(get_text("preview"))
    st.dataframe(df.head(1000))

    # Summary
    st.write(f"{get_text('rows')}: {df.shape[0]}")
    st.write(f"{get_text('cols')}: {df.shape[1]}")
    num_df = df.select_dtypes(include=np.number)
    st.write(f"{get_text('num_cols')}: {len(num_df.columns)}")
    cat_df = df.select_dtypes(exclude=np.number)
    st.write(f"{get_text('non_num_cols')}: {len(cat_df.columns)}")

    # TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        get_text("tab1"),
        get_text("tab2"),
        get_text("tab3"),
        get_text("tab4")
    ])

    # ============================
    # TAB 1 – Descriptive Stats
    # ============================
    with tab1:
        if len(num_df.columns) > 0:
            col = st.selectbox(get_text("select_numeric"), num_df.columns)

            series = num_df[col].dropna()

            st.write(series.describe())

            stat, p = normaltest(series)
            st.write(f"Normaltest stat={stat:.3f} p={p:.4f}")
            st.write("Normal" if p > 0.05 else "Not Normal")

            # Histogram
            fig, ax = plt.subplots()
            sns.histplot(series, ax=ax)
            st.pyplot(fig)

            # Boxplot
            fig, ax = plt.subplots()
            sns.boxplot(x=series, ax=ax)
            st.pyplot(fig)

        # Frequency Table for categorical
        for c in cat_df.columns:
            st.subheader(f"Frequency: {c}")
            freq = df[c].value_counts(dropna=False)
            pct = freq / len(df) * 100
            out = pd.DataFrame({"Count": freq, "Percent": pct})
            st.dataframe(out)

    # ============================
    # TAB 2 – Visualizations
    # ============================
    with tab2:
        # Histogram + Boxplot
        if len(num_df.columns) > 0:
            col = st.selectbox(get_text("hist"), num_df.columns)
            fig, ax = plt.subplots()
            sns.histplot(num_df[col], ax=ax)
            st.pyplot(fig)

            fig, ax = plt.subplots()
            sns.boxplot(x=num_df[col], ax=ax)
            st.pyplot(fig)

        # Scatter
        if len(num_df.columns) >= 2:
            c1 = st.selectbox("X", num_df.columns)
            c2 = st.selectbox("Y", num_df.columns, index=1)

            fig, ax = plt.subplots()
            ax.scatter(df[c1], df[c2])
            ax.set_xlabel(c1)
            ax.set_ylabel(c2)
            st.pyplot(fig)

        # Bar chart categorical
        if len(cat_df.columns) > 0:
            cc = st.selectbox(get_text("barchart"), cat_df.columns)
            vc = df[cc].value_counts().head(20)
            fig, ax = plt.subplots()
            sns.barplot(x=vc.values, y=vc.index, ax=ax)
            st.pyplot(fig)

    # ============================
    # TAB 3 – Correlations & Tests
    # ============================
    with tab3:
        if len(num_df.columns) >= 2:
            c1 = st.selectbox(get_text("select_two_numeric"), num_df.columns)
            c2 = st.selectbox("Select second numeric", num_df.columns, index=1)

            r, p = pearsonr(df[c1].dropna(), df[c2].dropna())
            st.write(f"Pearson r={r:.3f} p={p:.4f}")

            r2, p2 = spearmanr(df[c1].dropna(), df[c2].dropna())
            st.write(f"Spearman r={r2:.3f} p={p2:.4f}")

        # Chi-square
        if len(cat_df.columns) >= 2:
            c1 = st.selectbox(get_text("select_two_categorical"), cat_df.columns)
            c2 = st.selectbox("Second categorical", cat_df.columns, index=1)

            table = pd.crosstab(df[c1], df[c2])
            chi2, p, dof, exp = chi2_contingency(table)

            st.write("Chi-square:", chi2)
            st.write("p-value:", p)
            st.write("df:", dof)
            st.write("Observed:")
            st.dataframe(table)
            st.write("Expected:")
            st.dataframe(pd.DataFrame(exp, index=table.index, columns=table.columns))

        # Correlation matrix
        if len(num_df.columns) > 1:
            st.subheader("Correlation Matrix")
            st.dataframe(num_df.corr())

    # ============================
    # TAB 4 – Text Processing
    # ============================
    with tab4:
        text_cols = df.select_dtypes(include="object").columns
        if len(text_cols) == 0:
            st.write("No text columns found.")
        else:
            col = st.selectbox("Select text column", text_cols)
            tokens, counter = preprocess_text_series(df[col])
            st.write("Sample tokens:", tokens.head())
            top10 = counter.most_common(10)
            st.write("Top 10 words:", top10)

    # ============================
    # PDF EXPORT
    # ============================
    st.subheader(get_text("generate_pdf"))
    if st.button(get_text("generate_pdf")):
        pdf_bytes = build_survey_report_pdf(df)
        st.download_button(get_text("download_pdf"), data=pdf_bytes, file_name="survey_report.pdf")
