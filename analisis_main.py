# streamlit_full_survey.py
"""
Full-featured, single-file Streamlit app for survey data analysis.
Features:
- Upload CSV/XLS/XLSX
- Preview (max 1000 rows)
- Summary (rows, cols, numeric/non-numeric/text)
- Tabs: Descriptive Stats, Visualizations, Correlations & Tests, Text Processing
- Safe Pearson/Spearman/Chi-square, Normality (scipy.stats.normaltest)
- Histograms, Boxplots, Scatter, Bar charts (matplotlib & seaborn)
- Text preprocessing (lowercase, remove punctuation, tokenize, remove NLTK English stopwords)
- PDF export (reportlab) with dataset metadata, numeric stats, normality, plots, correlation matrix, top categorical frequencies, top words per text column
- UI: Orange background, Dark Mode toggle, Language selector (EN, ID, JP, KR, CN)
- All UI text via TEXTS dictionary and get_text(key) fallback to English
- Robust error handling and synchronized dropna for pairwise tests/plots
"""

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
from datetime import datetime
import math
import warnings
warnings.filterwarnings("ignore")

# --------------------------
# TEXTS: EN, ID, CN, JP, KR
# --------------------------
TEXTS = {
    "EN": {
        "title": "Survey Data Analyzer",
        "subtitle": "Upload, explore, analyze survey data and export a PDF report",
        "upload": "Upload CSV / XLS / XLSX",
        "preview": "Data preview (max 1000 rows)",
        "summary": "Dataset summary",
        "rows": "Rows",
        "cols": "Columns",
        "num_cols": "Numeric columns",
        "cat_cols": "Categorical columns",
        "text_cols": "Text columns",
        "descriptive_tab": "Descriptive Stats",
        "visual_tab": "Visualizations",
        "corr_tab": "Correlations & Tests",
        "text_tab": "Text Processing",
        "select_numeric": "Select numeric column",
        "stats_title": "Descriptive statistics",
        "normality_title": "Normality test (D'Agostino & Pearson)",
        "statistic": "Statistic",
        "p_value": "p-value",
        "interpretation": "Interpretation",
        "histogram": "Histogram",
        "boxplot": "Boxplot",
        "frequency_table": "Frequency table (count + percent)",
        "select_x": "Select X (numeric)",
        "select_y": "Select Y (numeric)",
        "scatter_title": "Scatter plot",
        "bar_chart": "Bar chart (top 20)",
        "select_categorical": "Select categorical column",
        "pearson": "Pearson correlation",
        "spearman": "Spearman correlation",
        "chi_square": "Chi-square test",
        "observed": "Observed",
        "expected": "Expected",
        "corr_matrix": "Pearson correlation matrix",
        "detect_text": "Detected text columns",
        "preprocess": "Preprocess text (lowercase, remove punctuation, tokenize, remove stopwords)",
        "sample_tokens": "Sample tokens",
        "top_words": "Top 10 words",
        "generate_pdf": "Generate PDF Report",
        "download_pdf": "Download PDF",
        "dark_mode": "Dark Mode",
        "language": "Language",
        "no_data": "No data loaded. Please upload a file.",
        "insufficient": "Insufficient data for this operation",
        "pdf_generating": "Building PDF — this may take a moment...",
        "pdf_done": "PDF generated",
        "pdf_name": "survey_report"
    },
    "ID": {
        "title": "Aplikasi Analisis Data Survei",
        "subtitle": "Unggah, jelajahi, analisis data survei dan ekspor laporan PDF",
        "upload": "Unggah CSV / XLS / XLSX",
        "preview": "Pratinjau data (maks 1000 baris)",
        "summary": "Ringkasan dataset",
        "rows": "Baris",
        "cols": "Kolom",
        "num_cols": "Kolom numerik",
        "cat_cols": "Kolom kategorikal",
        "text_cols": "Kolom teks",
        "descriptive_tab": "Statistik Deskriptif",
        "visual_tab": "Visualisasi",
        "corr_tab": "Korelasi & Uji",
        "text_tab": "Pemrosesan Teks",
        "select_numeric": "Pilih kolom numerik",
        "stats_title": "Statistik deskriptif",
        "normality_title": "Uji normalitas (D'Agostino & Pearson)",
        "statistic": "Statistik",
        "p_value": "p-value",
        "interpretation": "Interpretasi",
        "histogram": "Histogram",
        "boxplot": "Boxplot",
        "frequency_table": "Tabel frekuensi (count + persen)",
        "select_x": "Pilih X (numerik)",
        "select_y": "Pilih Y (numerik)",
        "scatter_title": "Plot scatter",
        "bar_chart": "Diagram batang (top 20)",
        "select_categorical": "Pilih kolom kategorikal",
        "pearson": "Korelasi Pearson",
        "spearman": "Korelasi Spearman",
        "chi_square": "Uji Chi-square",
        "observed": "Observed",
        "expected": "Expected",
        "corr_matrix": "Matriks korelasi Pearson",
        "detect_text": "Kolom teks terdeteksi",
        "preprocess": "Preproses teks (lowercase, hapus tanda baca, token, hapus stopwords)",
        "sample_tokens": "Contoh token",
        "top_words": "10 kata terbanyak",
        "generate_pdf": "Buat Laporan PDF",
        "download_pdf": "Unduh PDF",
        "dark_mode": "Mode Gelap",
        "language": "Bahasa",
        "no_data": "Tidak ada data. Silakan unggah file.",
        "insufficient": "Data tidak cukup untuk operasi ini",
        "pdf_generating": "Membangun PDF — mohon tunggu...",
        "pdf_done": "PDF berhasil dibuat",
        "pdf_name": "laporan_survey"
    },
    "CN": {
        "title": "调查数据分析器",
        "subtitle": "上传、探索、分析调查数据并导出 PDF 报告",
        "upload": "上传 CSV / XLS / XLSX",
        "preview": "数据预览（最多1000行）",
        "summary": "数据集摘要",
        "rows": "行数",
        "cols": "列数",
        "num_cols": "数值列",
        "cat_cols": "分类列",
        "text_cols": "文本列",
        "descriptive_tab": "描述性统计",
        "visual_tab": "可视化",
        "corr_tab": "相关与检验",
        "text_tab": "文本处理",
        "select_numeric": "选择数值列",
        "stats_title": "描述性统计",
        "normality_title": "正态性检验 (D'Agostino & Pearson)",
        "statistic": "统计量",
        "p_value": "p 值",
        "interpretation": "解释",
        "histogram": "直方图",
        "boxplot": "箱线图",
        "frequency_table": "频率表（计数 + 百分比）",
        "select_x": "选择 X（数值）",
        "select_y": "选择 Y（数值）",
        "scatter_title": "散点图",
        "bar_chart": "柱状图（前20）",
        "select_categorical": "选择分类列",
        "pearson": "皮尔逊相关",
        "spearman": "斯皮尔曼相关",
        "chi_square": "卡方检验",
        "observed": "观察值",
        "expected": "期望值",
        "corr_matrix": "皮尔逊相关矩阵",
        "detect_text": "检测到的文本列",
        "preprocess": "文本预处理（小写、去标点、分词、去停用词）",
        "sample_tokens": "示例词",
        "top_words": "前10词",
        "generate_pdf": "生成 PDF 报告",
        "download_pdf": "下载 PDF",
        "dark_mode": "深色模式",
        "language": "语言",
        "no_data": "未加载数据。请上传文件。",
        "insufficient": "此操作的数据不足",
        "pdf_generating": "正在生成 PDF，请稍候...",
        "pdf_done": "PDF 已生成",
        "pdf_name": "调查报告"
    },
    "JP": {
        "title": "調査データ分析ツール",
        "subtitle": "データをアップロードして解析し、PDFレポートを出力します",
        "upload": "CSV / XLS / XLSX をアップロード",
        "preview": "データプレビュー（最大1000行）",
        "summary": "データセット概要",
        "rows": "行数",
        "cols": "列数",
        "num_cols": "数値列",
        "cat_cols": "カテゴリ列",
        "text_cols": "テキスト列",
        "descriptive_tab": "記述統計",
        "visual_tab": "可視化",
        "corr_tab": "相関と検定",
        "text_tab": "テキスト処理",
        "select_numeric": "数値列を選択",
        "stats_title": "記述統計量",
        "normality_title": "正規性検定 (D'Agostino & Pearson)",
        "statistic": "統計量",
        "p_value": "p値",
        "interpretation": "解釈",
        "histogram": "ヒストグラム",
        "boxplot": "箱ひげ図",
        "frequency_table": "頻度表（個数 + 割合）",
        "select_x": "Xを選択（数値）",
        "select_y": "Yを選択（数値）",
        "scatter_title": "散布図",
        "bar_chart": "棒グラフ（上位20）",
        "select_categorical": "カテゴリ列を選択",
        "pearson": "ピアソン相関",
        "spearman": "スピアマン相関",
        "chi_square": "カイ二乗検定",
        "observed": "観測値",
        "expected": "期待値",
        "corr_matrix": "ピアソン相関行列",
        "detect_text": "検出されたテキスト列",
        "preprocess": "テキスト前処理（小文字化・句読点削除・分割・ストップワード除去）",
        "sample_tokens": "トークン例",
        "top_words": "上位10語",
        "generate_pdf": "PDFレポートを作成",
        "download_pdf": "PDFをダウンロード",
        "dark_mode": "ダークモード",
        "language": "言語",
        "no_data": "データが読み込まれていません。ファイルをアップロードしてください。",
        "insufficient": "この操作にはデータが不十分です",
        "pdf_generating": "PDFを生成しています、しばらくお待ちください...",
        "pdf_done": "PDFが生成されました",
        "pdf_name": "調査レポート"
    },
    "KR": {
        "title": "설문 데이터 분석기",
        "subtitle": "데이터 업로드, 탐색, 분석 및 PDF 보고서 생성",
        "upload": "CSV / XLS / XLSX 업로드",
        "preview": "데이터 미리보기(최대 1000행)",
        "summary": "데이터셋 요약",
        "rows": "행",
        "cols": "열",
        "num_cols": "숫자 열",
        "cat_cols": "범주 열",
        "text_cols": "텍스트 열",
        "descriptive_tab": "기술 통계",
        "visual_tab": "시각화",
        "corr_tab": "상관 및 검정",
        "text_tab": "텍스트 처리",
        "select_numeric": "숫자 열 선택",
        "stats_title": "기술 통계량",
        "normality_title": "정규성 검정 (D'Agostino & Pearson)",
        "statistic": "검정통계량",
        "p_value": "p값",
        "interpretation": "해석",
        "histogram": "히스토그램",
        "boxplot": "박스플롯",
        "frequency_table": "빈도표(개수 + 백분율)",
        "select_x": "X 선택(숫자)",
        "select_y": "Y 선택(숫자)",
        "scatter_title": "산점도",
        "bar_chart": "막대 차트(상위20)",
        "select_categorical": "범주형 열 선택",
        "pearson": "피어슨 상관",
        "spearman": "스피어만 상관",
        "chi_square": "카이제곱 검정",
        "observed": "관측값",
        "expected": "기대값",
        "corr_matrix": "피어슨 상관 행렬",
        "detect_text": "감지된 텍스트 열",
        "preprocess": "텍스트 전처리(소문자, 구두점 제거, 토큰화, 불용어 제거)",
        "sample_tokens": "토큰 샘플",
        "top_words": "상위10개 단어",
        "generate_pdf": "PDF 보고서 생성",
        "download_pdf": "PDF 다운로드",
        "dark_mode": "다크 모드",
        "language": "언어",
        "no_data": "데이터가 없습니다. 파일을 업로드하세요.",
        "insufficient": "이 작업에 데이터가 충분하지 않습니다",
        "pdf_generating": "PDF 생성 중입니다... 잠시만 기다려 주세요",
        "pdf_done": "PDF 생성 완료",
        "pdf_name": "설문_보고서"
    }
}

# --------------------------
# Helper: get_text with fallback
# --------------------------
def get_text(key: str) -> str:
    lang = st.session_state.get("language", "EN")
    if lang not in TEXTS:
        lang = "EN"
    return TEXTS.get(lang, TEXTS["EN"]).get(key, TEXTS["EN"].get(key, key))

# --------------------------
# Ensure NLTK stopwords available
# --------------------------
try:
    _ = stopwords.words("english")
except Exception:
    try:
        nltk.download("stopwords", quiet=True)
    except Exception:
        pass

# --------------------------
# Utility functions
# --------------------------
@st.cache_data
def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        else:
            return pd.read_excel(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        try:
            return pd.read_table(uploaded_file)
        except Exception as e:
            raise e

def safe_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()

def descriptive_stats(series: pd.Series) -> dict:
    s = safe_numeric_series(series)
    if s.empty:
        return {}
    return {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "mode": s.mode().tolist()[:3],
        "min": float(s.min()),
        "max": float(s.max()),
        "std": float(s.std()),
    }

def preprocess_text_series(series: pd.Series, remove_stopwords=True):
    if series is None:
        return [], Counter()
    try:
        sw = set(stopwords.words("english")) if remove_stopwords else set()
    except Exception:
        sw = set()
    translator = str.maketrans("", "", string.punctuation)
    cleaned = series.dropna().astype(str).str.lower().str.translate(translator)
    tokens = cleaned.apply(lambda x: [t for t in x.split() if t and (t not in sw)])
    cnt = Counter([w for sub in tokens for w in sub])
    return tokens, cnt

def fig_to_image_reader(fig, dpi=150):
    buf = BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return ImageReader(buf), buf

# --------------------------
# PDF builder (reportlab)
# --------------------------
def build_survey_report_pdf(df: pd.DataFrame, lang: str = "EN") -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=18)
    styles = getSampleStyleSheet()
    flow = []

    # Header
    flow.append(Paragraph(get_text("title"), styles["Title"]))
    flow.append(Paragraph(get_text("subtitle"), styles["Normal"]))
    flow.append(Spacer(1, 8))
    flow.append(Paragraph(f"{get_text('rows')}: {len(df)}  |  {get_text('cols')}: {len(df.columns)}", styles["Normal"]))
    flow.append(Spacer(1, 12))

    num_df = df.select_dtypes(include=np.number)
    cat_df = df.select_dtypes(exclude=np.number)
    text_cols = [c for c in df.columns if df[c].dtype == object]

    # Numeric columns (limit to 6 to avoid huge PDF)
    for col in list(num_df.columns)[:6]:
        s = safe_numeric_series(df[col])
        flow.append(Paragraph(f"<b>{col}</b>", styles["Heading2"]))
        if s.empty:
            flow.append(Paragraph(get_text("insufficient"), styles["Normal"]))
            flow.append(Spacer(1, 6))
            continue
        desc = descriptive_stats(s)
        flow.append(Paragraph(f"Count: {desc['count']}, Mean: {desc['mean']:.4f}, Median: {desc['median']:.4f}, Std: {desc['std']:.4f}", styles["Normal"]))
        flow.append(Paragraph(f"Min: {desc['min']}, Max: {desc['max']}", styles["Normal"]))
        # Normality
        try:
            if len(s) >= 8:
                stat, p = normaltest(s)
                flow.append(Paragraph(f"{get_text('normality_title')}: stat={stat:.4f}, p={p:.6f}", styles["Normal"]))
            else:
                flow.append(Paragraph(f"{get_text('normality_title')}: need >=8 values", styles["Normal"]))
        except Exception:
            flow.append(Paragraph(f"{get_text('normality_title')}: error computing", styles["Normal"]))
        flow.append(Spacer(1, 6))
        # Plots
        try:
            fig1, ax1 = plt.subplots(figsize=(6, 2.2))
            sns.histplot(s, kde=True, ax=ax1)
            ax1.set_title(f"{col} - {get_text('histogram')}")
            img1, buf1 = fig_to_image_reader(fig1)
            flow.append(Image(img1, width=400, height=140))
            flow.append(Spacer(1, 6))
        except Exception:
            pass
        try:
            fig2, ax2 = plt.subplots(figsize=(6, 1.4))
            sns.boxplot(x=s, ax=ax2)
            ax2.set_title(f"{col} - {get_text('boxplot')}")
            img2, buf2 = fig_to_image_reader(fig2)
            flow.append(Image(img2, width=400, height=110))
            flow.append(Spacer(1, 12))
        except Exception:
            pass

    # Correlation matrix
    if num_df.shape[1] >= 2:
        flow.append(Paragraph(get_text("corr_matrix"), styles["Heading2"]))
        corr = num_df.corr().round(3)
        headers = [""] + list(corr.columns)
        table_data = [headers]
        for idx in corr.index:
            row = [idx] + [str(corr.loc[idx, c]) for c in corr.columns]
            table_data.append(row)
        flow.append(Table(table_data))
        flow.append(Spacer(1, 12))

    # Categorical top 10
    if not cat_df.empty:
        flow.append(Paragraph("Categorical frequency (top 10)", styles["Heading2"]))
        for col in list(cat_df.columns)[:8]:
            flow.append(Paragraph(f"<b>{col}</b>", styles["Heading3"]))
            vc = df[col].fillna("(Missing)").astype(str).value_counts().head(10)
            table_data = [["Category", "Count", "Percent"]]
            for k, v in vc.items():
                pct = (v / len(df)) * 100
                table_data.append([str(k), str(int(v)), f"{pct:.1f}%"])
            flow.append(Table(table_data))
            flow.append(Spacer(1, 6))

    # Text top words
    if text_cols:
        flow.append(Paragraph(get_text("processing_text"), styles["Heading2"]))
        for col in text_cols[:6]:
            flow.append(Paragraph(f"<b>{col}</b>", styles["Heading3"]))
            try:
                _, counter = preprocess_text_series(df[col])
                table_data = [["Word", "Count"]]
                for w, cnt in counter.most_common(10):
                    table_data.append([w, str(cnt)])
                flow.append(Table(table_data))
                flow.append(Spacer(1, 6))
            except Exception:
                flow.append(Paragraph("Error processing text", styles["Normal"]))

    # Footer
    flow.append(Spacer(1, 10))
    flow.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))

    # Build PDF
    try:
        doc.build(flow)
        buffer.seek(0)
        pdf_bytes = buffer.getvalue()
    finally:
        buffer.close()
    return pdf_bytes

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Survey Data Analyzer", layout="wide")
# orange background
st.markdown("""
    <style>
    .stApp { background-color: #FFA64D !important; }
    .block-container { padding: 1rem 2rem; }
    </style>
    """, unsafe_allow_html=True)

# init session state
if "language" not in st.session_state:
    st.session_state["language"] = "EN"
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False
if "last_pdf" not in st.session_state:
    st.session_state["last_pdf"] = None

# top controls
col_a, col_b = st.columns([1, 2])
with col_a:
    st.session_state["dark_mode"] = st.checkbox(get_text("dark_mode"), value=st.session_state["dark_mode"])
with col_b:
    st.session_state["language"] = st.radio(get_text("language"), options=["EN", "ID", "CN", "JP", "KR"], index=["EN", "ID", "CN", "JP", "KR"].index(st.session_state["language"]), horizontal=True)

st.title(get_text("title"))
st.caption(get_text("subtitle"))

# file uploader
uploaded = st.file_uploader(get_text("upload"))
if uploaded is None:
    st.info(get_text("no_data"))
    st.stop()

# load data
try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# preview & summary
st.subheader(get_text("preview"))
st.dataframe(df.head(1000))

num_df = df.select_dtypes(include=np.number)
cat_df = df.select_dtypes(exclude=np.number)
text_cols = [c for c in df.columns if df[c].dtype == object]

c1, c2, c3, c4 = st.columns(4)
c1.metric(get_text("rows"), df.shape[0])
c2.metric(get_text("cols"), df.shape[1])
c3.metric(get_text("num_cols"), num_df.shape[1])
c4.metric(get_text("cat_cols"), cat_df.shape[1])

# tabs
tabs = st.tabs([get_text("descriptive_tab"), get_text("visual_tab"), get_text("corr_tab"), get_text("text_tab")])

# Descriptive Stats tab
with tabs[0]:
    st.header(get_text("descriptive_tab"))
    if num_df.shape[1] == 0:
        st.info(get_text("insufficient"))
    else:
        sel_num = st.selectbox(get_text("select_numeric"), options=list(num_df.columns))
        if sel_num:
            s = safe_numeric_series(df[sel_num])
            if s.empty:
                st.write(get_text("insufficient"))
            else:
                stats = descriptive_stats(s)
                st.write(pd.DataFrame(stats, index=[sel_num]).T)
                # normality
                if len(s) >= 8:
                    try:
                        stat, p = normaltest(s)
                        st.write(get_text("normality_title"))
                        st.write({get_text("statistic"): float(stat), get_text("p_value"): float(p)})
                        st.write(get_text("interpretation") + ": " + (get_text("interpretation") if p > 0.05 else get_text("insufficient")))
                    except Exception:
                        st.write("Normality test: error")
                else:
                    st.write(f"{get_text('normality_title')}: need at least 8 non-missing values")

                # plots
                fig, ax = plt.subplots()
                sns.histplot(s, kde=True, ax=ax)
                ax.set_title(f"{sel_num} - {get_text('histogram')}")
                st.pyplot(fig)
                plt.close(fig)

                fig2, ax2 = plt.subplots()
                sns.boxplot(x=s, ax=ax2)
                ax2.set_title(f"{sel_num} - {get_text('boxplot')}")
                st.pyplot(fig2)
                plt.close(fig2)

    # categorical frequency tables
    st.subheader(get_text("frequency_table"))
    if cat_df.shape[1] == 0:
        st.info(get_text("insufficient"))
    else:
        sel_cat = st.selectbox(get_text("select_categorical"), options=list(cat_df.columns), key="freq_cat")
        if sel_cat:
            vc = df[sel_cat].fillna("(Missing)").astype(str).value_counts()
            pct = (vc / vc.sum()) * 100
            out = pd.DataFrame({"count": vc, "percent": pct})
            st.dataframe(out.head(100))

# Visualizations tab
with tabs[1]:
    st.header(get_text("visual_tab"))
    # Histogram & Boxplot
    if num_df.shape[1] > 0:
        sel_hist = st.selectbox(get_text("histogram"), options=list(num_df.columns), key="vis_hist")
        s = safe_numeric_series(df[sel_hist])
        if s.empty:
            st.write(get_text("insufficient"))
        else:
            fig, ax = plt.subplots()
            sns.histplot(s, kde=True, ax=ax)
            ax.set_title(f"{sel_hist} - {get_text('histogram')}")
            st.pyplot(fig)
            plt.close(fig)

            fig2, ax2 = plt.subplots()
            sns.boxplot(x=s, ax=ax2)
            ax2.set_title(f"{sel_hist} - {get_text('boxplot')}")
            st.pyplot(fig2)
            plt.close(fig2)
    else:
        st.info(get_text("insufficient"))

    # Scatter plot (synchronized)
    st.subheader(get_text("scatter_title"))
    if num_df.shape[1] >= 2:
        xcol = st.selectbox(get_text("select_x"), options=list(num_df.columns), key="scatter_x")
        ycol = st.selectbox(get_text("select_y"), options=list(num_df.columns), index=1 if len(num_df.columns) > 1 else 0, key="scatter_y")
        if xcol and ycol:
            clean = df[[xcol, ycol]].dropna()
            if clean.shape[0] < 2:
                st.write(get_text("insufficient"))
            else:
                fig, ax = plt.subplots()
                ax.scatter(clean[xcol], clean[ycol], alpha=0.7)
                ax.set_xlabel(xcol); ax.set_ylabel(ycol)
                ax.set_title(f"{xcol} vs {ycol}")
                st.pyplot(fig)
                plt.close(fig)
    else:
        st.info("Need at least two numeric columns for scatter")

    # Bar chart for categorical
    st.subheader(get_text("bar_chart"))
    if cat_df.shape[1] > 0:
        sel_cat_bar = st.selectbox(get_text("select_categorical") + " (for bar chart)", options=list(cat_df.columns), key="bar_cat")
        vc = df[sel_cat_bar].fillna("(Missing)").astype(str).value_counts().head(20)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(vc))))
        sns.barplot(x=vc.values, y=vc.index, ax=ax)
        ax.set_xlabel("Count"); ax.set_ylabel(sel_cat_bar)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No categorical columns for bar chart")

# Correlations & Tests tab
with tabs[2]:
    st.header(get_text("corr_tab"))
    # Pearson & Spearman
    st.subheader(f"{get_text('pearson')} & {get_text('spearman')}")
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
                    st.write(f"{get_text('pearson')}: r = {r:.4f}, p = {p:.6f}")
                    st.write("Interpretation (Pearson):", ("Positive" if r > 0 else "Negative" if r < 0 else "No relation"), f" (|r|={abs(r):.3f})")
                except Exception:
                    st.write("Pearson: error computing")
                try:
                    rho, p2 = spearmanr(clean[a], clean[b])
                    st.write(f"{get_text('spearman')}: rho = {rho:.4f}, p = {p2:.6f}")
                    st.write("Interpretation (Spearman):", ("Positive" if rho > 0 else "Negative" if rho < 0 else "No relation"), f" (|rho|={abs(rho):.3f})")
                except Exception:
                    st.write("Spearman: error computing")
    else:
        st.info("Need at least two numeric columns for correlations")

    # Chi-square
    st.subheader(get_text("chi_square"))
    if cat_df.shape[1] >= 2:
        ca = st.selectbox(get_text("select_two_categorical") + " 1", options=list(cat_df.columns), key="chi_a")
        cb = st.selectbox(get_text("select_two_categorical") + " 2", options=list(cat_df.columns), index=1 if len(cat_df.columns) > 1 else 0, key="chi_b")
        if ca and cb:
            ct = pd.crosstab(df[ca].fillna("(Missing)").astype(str), df[cb].fillna("(Missing)").astype(str))
            if ct.size == 0 or ct.values.sum() == 0 or ct.shape[0] < 2 or ct.shape[1] < 2:
                st.write(get_text("insufficient"))
            else:
                try:
                    chi2, pval, dof, expected = chi2_contingency(ct)
                    st.write(f"chi2 = {chi2:.4f}, p = {pval:.6f}, dof = {dof}")
                    st.write(get_text("observed"))
                    st.dataframe(ct)
                    st.write(get_text("expected"))
                    st.dataframe(pd.DataFrame(expected, index=ct.index, columns=ct.columns))
                except Exception as e:
                    st.write(f"Chi-square error: {e}")
    else:
        st.info("Need at least two categorical columns for chi-square")

    # Correlation matrix
    st.subheader(get_text("corr_matrix"))
    if num_df.shape[1] > 1:
        st.dataframe(num_df.corr().round(4))
    else:
        st.info(get_text("insufficient"))

# Text Processing tab
with tabs[3]:
    st.header(get_text("text_tab"))
    if not text_cols:
        st.info("No text columns found")
    else:
        sel_text = st.selectbox("Select text column", options=text_cols, key="text_select")
        remove_sw = st.checkbox("Remove English stopwords (NLTK)", value=True)
        tokens, counter = preprocess_text_series(df[sel_text], remove_stopwords=remove_sw)
        st.subheader(get_text("sample_tokens"))
        try:
            st.write(tokens.head(5).to_dict())
        except Exception:
            st.write(list(tokens)[:5])
        st.subheader(get_text("top_words"))
        st.table(pd.DataFrame(counter.most_common(10), columns=["word", "count"]))

# Export / PDF (main page, tidy)
st.markdown("---")
st.header("Export / Report")
col_info, col_gen, col_down = st.columns([3, 1, 1])
with col_info:
    st.write("Generate a PDF report that includes dataset metadata, numeric statistics, normality results, plots (histogram & boxplot), correlation matrix, top categorical frequencies, and top words for text columns. Limited number of items included to keep PDF manageable.")
with col_gen:
    if st.button(get_text("generate_pdf")):
        with st.spinner(get_text("pdf_generating")):
            try:
                pdf_bytes = build_survey_report_pdf(df, lang=st.session_state.get("language", "EN"))
                st.session_state["last_pdf"] = pdf_bytes
                st.success(get_text("pdf_done"))
            except Exception as e:
                st.error(f"Error generating PDF: {e}")
                st.session_state["last_pdf"] = None
with col_down:
    if st.session_state.get("last_pdf"):
        fname = f"{get_text('pdf_name')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        st.download_button(get_text("download_pdf"), data=st.session_state["last_pdf"], file_name=fname, mime="application/pdf")

# End
