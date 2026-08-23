"""
app.py — Medical Image Segmentation + Report Generation
Run: streamlit run app.py
"""

import streamlit as st
from PIL import Image
import numpy as np
import io

from segmentation.model import generate_masks, select_best_mask, get_all_mask_stats
from segmentation.postprocess import overlay_mask_on_image, extract_region_stats, interpret_stats
from report.generator import generate_report
from report.pdf_export import generate_pdf

# -- Page Config ---------------------------------------------------------------
st.set_page_config(
    page_title="MediScan AI",
    page_icon="🩺",
    layout="wide",
)

# -- Theme Definitions ---------------------------------------------------------
THEME_CSS = {
    "Dark": """
        <style>
        :root {
            --bg-color: #0b1220;
            --secondary-bg: #111a2b;
            --surface-color: #162237;
            --text-color: #f4f7fb;
            --muted-text: #aab7c8;
            --accent: #62d7c1;
            --border-color: #2a3a52;
        }
        .stApp { background-color: var(--bg-color) !important; color: var(--text-color) !important; }
        header[data-testid="stHeader"] { background-color: var(--bg-color) !important; }
        section[data-testid="stSidebar"] { background-color: var(--secondary-bg) !important; border-right: 1px solid var(--border-color) !important; }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp label,
        .stApp [data-testid="stMarkdownContainer"] p,
        .stApp [data-testid="stMarkdownContainer"] li { color: var(--text-color) !important; }
        .stApp [data-testid="stCaptionContainer"] { color: var(--muted-text) !important; }
        .stButton > button { border: 1px solid var(--accent) !important; color: var(--accent) !important; background: transparent !important; }
        .stRadio label { color: var(--text-color) !important; }
        div[data-baseweb="select"] > div,
        div[data-testid="stFileUploader"],
        div[data-testid="stFileUploader"] section,
        div[data-testid="stFileUploaderDropzone"] { background-color: var(--surface-color) !important; border-color: var(--border-color) !important; }
        div[data-testid="stFileUploader"] small,
        div[data-testid="stFileUploader"] span { color: var(--muted-text) !important; }
        div[data-testid="stFileUploader"] button { background-color: #22324a !important; border: 1px solid #3b506d !important; color: var(--text-color) !important; }
        div[data-testid="stFileUploader"] button:hover { border-color: var(--accent) !important; color: var(--accent) !important; }
        div[data-testid="stAlert"] { background-color: #18263a !important; border: 1px solid #38516d !important; border-left: 3px solid var(--accent) !important; color: var(--text-color) !important; }
        div[data-testid="stAlert"] p { color: var(--text-color) !important; }
        div[data-testid="stExpander"] { background-color: var(--surface-color) !important; border: 1px solid var(--border-color) !important; }
        hr { border-color: var(--border-color) !important; }
        </style>
    """,
    "Light": """
        <style>
        .stApp { background-color: #f8fafc !important; }
        header[data-testid="stHeader"] { background-color: #ffffff !important; }
        [data-testid="stSidebar"] { background-color: #e8edf5 !important; border-right: 1px solid #d1d9e6 !important; }
        .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stApp span, .stApp label, .stApp li, .stApp div[data-testid="stMarkdownContainer"] * { color: #1a1a2e !important; }
        button[kind="primary"] * { color: white !important; }
        div[data-baseweb="select"] > div,
        div[data-testid="stFileUploader"],
        div[data-testid="stFileUploader"] section,
        div[data-testid="stFileUploaderDropzone"],
        div[data-testid="stExpander"] { background-color: #ffffff !important; border-color: #d1d9e6 !important; }
        div[data-baseweb="select"], div[data-baseweb="select"] > div,
        div[data-baseweb="select"] input { background-color: #ffffff !important; color: #1a1a2e !important; }
        [data-testid="stSelectbox"] [data-baseweb="select"] * { color: #1a1a2e !important; }
        div[data-baseweb="select"] svg { fill: #344054 !important; color: #344054 !important; }
        div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"], li[role="option"] { background-color: #ffffff !important; color: #1a1a2e !important; }
        div[data-baseweb="popover"] *, div[data-baseweb="menu"] *, ul[role="listbox"] *, li[role="option"] * { color: #1a1a2e !important; }
        li[role="option"]:hover, li[role="option"][aria-selected="true"] { background-color: #e8f1f8 !important; color: #1a1a2e !important; }
        [data-testid="stRadio"] * { color: #1a1a2e !important; }
        [data-testid="stRadio"] div[role="radio"], [data-baseweb="radio"] div { background-color: #ffffff !important; border-color: #98a2b3 !important; color: #1a1a2e !important; }
        [data-testid="stRadio"] div[role="radio"][aria-checked="true"] { background-color: #62d7c1 !important; border-color: #149f8b !important; }
        [data-testid="stRadio"] input[type="radio"] { accent-color: #16b8a2 !important; }
        [data-testid="stRadio"] label span, [data-testid="stSelectbox"] [data-baseweb="select"] * { color: #344054 !important; }
        [data-testid="stRadioGroup"], [data-testid="stRadioGroup"] label,
        [data-testid="stRadioGroup"] label *, [data-testid="stRadioGroup"] p { color: #344054 !important; }
        [data-testid="stRadioGroup"] input[type="radio"] { accent-color: #16b8a2 !important; }
        [data-testid="stRadioOption"] > div > div > div:first-child { background-color: #ffffff !important; border: 1px solid #98a2b3 !important; }
        [data-testid="stRadioOption"] > div > div > div:first-child > div { background-color: #ffffff !important; }
        [data-testid="stRadioOption"][data-selected="true"] > div > div > div:first-child { background-color: #16b8a2 !important; border-color: #16b8a2 !important; }
        [data-testid="stRadioOption"][data-selected="true"] > div > div > div:first-child > div { background-color: #ffffff !important; }
        [data-testid="stSelectbox"] [role="group"] { background-color: #ffffff !important; }
        [data-testid="stSelectbox"] [role="combobox"], [data-testid="stSelectbox"] [role="combobox"] * { color: #344054 !important; background-color: #ffffff !important; }
        [data-testid="stSelectbox"] [role="group"] button { background-color: #ffffff !important; color: #344054 !important; }
        body [role="listbox"], body [role="listbox"] *, body [role="option"] { color: #344054 !important; background-color: #ffffff !important; }
        div[data-testid="stFileUploader"] button { background-color: #ffffff !important; border: 1px solid #c7d0dc !important; color: #1a1a2e !important; }
        div[data-testid="stFileUploader"] button:hover { border-color: #0077cc !important; color: #0077cc !important; }
        div[data-testid="stFileUploader"] small, div[data-testid="stFileUploader"] span { color: #667085 !important; }
        div[data-testid="stAlert"] { background-color: #fff9d9 !important; border: 1px solid #eadb87 !important; color: #1a1a2e !important; }
        div[data-testid="stAlert"] p { color: #1a1a2e !important; }
        [data-testid="stMetricValue"] div { color: #0077cc !important; }
        hr { border-color: #d1d9e6 !important; }
        </style>
    """,
    "System": """
        <style>
        @media (prefers-color-scheme: dark) {
            .stApp { background-color: #0b1220 !important; color: #f4f7fb !important; }
            header[data-testid="stHeader"] { background-color: #0b1220 !important; }
            [data-testid="stSidebar"] { background-color: #111a2b !important; border-right: 1px solid #2a3a52 !important; }
            .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp label, .stApp [data-testid="stMarkdownContainer"] p, .stApp [data-testid="stMarkdownContainer"] li { color: #f4f7fb !important; }
            .stApp [data-testid="stCaptionContainer"] { color: #aab7c8 !important; }
            div[data-baseweb="select"] > div, div[data-testid="stFileUploader"], div[data-testid="stFileUploader"] section, div[data-testid="stFileUploaderDropzone"] { background-color: #162237 !important; border-color: #2a3a52 !important; }
            div[data-testid="stFileUploader"] small, div[data-testid="stFileUploader"] span { color: #aab7c8 !important; }
            div[data-testid="stFileUploader"] button { background-color: #22324a !important; border: 1px solid #3b506d !important; color: #f4f7fb !important; }
            div[data-testid="stAlert"] { background-color: #18263a !important; border: 1px solid #38516d !important; border-left: 3px solid #62d7c1 !important; color: #f4f7fb !important; }
            div[data-testid="stAlert"] p { color: #f4f7fb !important; }
            div[data-testid="stExpander"] { background-color: #162237 !important; border: 1px solid #2a3a52 !important; }
            hr { border-color: #2a3a52 !important; }
        }
        @media (prefers-color-scheme: light) {
            .stApp { background-color: #f8fafc !important; }
            header[data-testid="stHeader"] { background-color: #ffffff !important; }
            [data-testid="stSidebar"] { background-color: #e8edf5 !important; border-right: 1px solid #d1d9e6 !important; }
            .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp span, .stApp label, .stApp li, .stApp div[data-testid="stMarkdownContainer"] * { color: #1a1a2e !important; }
            button[kind="primary"] * { color: white !important; }
            div[data-baseweb="select"] > div, div[data-testid="stFileUploader"], div[data-testid="stFileUploader"] section, div[data-testid="stFileUploaderDropzone"], div[data-testid="stExpander"] { background-color: #ffffff !important; border-color: #d1d9e6 !important; }
            div[data-baseweb="select"], div[data-baseweb="select"] > div, div[data-baseweb="select"] input { background-color: #ffffff !important; color: #1a1a2e !important; }
            [data-testid="stSelectbox"] [data-baseweb="select"] * { color: #1a1a2e !important; }
            div[data-baseweb="select"] svg { fill: #344054 !important; color: #344054 !important; }
            div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"], li[role="option"] { background-color: #ffffff !important; color: #1a1a2e !important; }
            div[data-baseweb="popover"] *, div[data-baseweb="menu"] *, ul[role="listbox"] *, li[role="option"] * { color: #1a1a2e !important; }
            li[role="option"]:hover, li[role="option"][aria-selected="true"] { background-color: #e8f1f8 !important; color: #1a1a2e !important; }
            [data-testid="stRadio"] * { color: #1a1a2e !important; }
            [data-testid="stRadio"] div[role="radio"], [data-baseweb="radio"] div { background-color: #ffffff !important; border-color: #98a2b3 !important; color: #1a1a2e !important; }
            [data-testid="stRadio"] div[role="radio"][aria-checked="true"] { background-color: #62d7c1 !important; border-color: #149f8b !important; }
            [data-testid="stRadio"] input[type="radio"] { accent-color: #16b8a2 !important; }
            [data-testid="stRadio"] label span, [data-testid="stSelectbox"] [data-baseweb="select"] * { color: #344054 !important; }
            [data-testid="stRadioGroup"], [data-testid="stRadioGroup"] label, [data-testid="stRadioGroup"] label *, [data-testid="stRadioGroup"] p { color: #344054 !important; }
            [data-testid="stRadioGroup"] input[type="radio"] { accent-color: #16b8a2 !important; }
            [data-testid="stRadioOption"] > div > div > div:first-child { background-color: #ffffff !important; border: 1px solid #98a2b3 !important; }
            [data-testid="stRadioOption"] > div > div > div:first-child > div { background-color: #ffffff !important; }
            [data-testid="stRadioOption"][data-selected="true"] > div > div > div:first-child { background-color: #16b8a2 !important; border-color: #16b8a2 !important; }
            [data-testid="stRadioOption"][data-selected="true"] > div > div > div:first-child > div { background-color: #ffffff !important; }
            [data-testid="stSelectbox"] [role="group"] { background-color: #ffffff !important; }
            [data-testid="stSelectbox"] [role="combobox"], [data-testid="stSelectbox"] [role="combobox"] * { color: #344054 !important; background-color: #ffffff !important; }
            [data-testid="stSelectbox"] [role="group"] button { background-color: #ffffff !important; color: #344054 !important; }
            body [role="listbox"], body [role="listbox"] *, body [role="option"] { color: #344054 !important; background-color: #ffffff !important; }
            div[data-testid="stFileUploader"] button { background-color: #ffffff !important; border: 1px solid #c7d0dc !important; color: #1a1a2e !important; }
            div[data-testid="stFileUploader"] small, div[data-testid="stFileUploader"] span { color: #667085 !important; }
            div[data-testid="stAlert"] { background-color: #fff9d9 !important; border: 1px solid #eadb87 !important; color: #1a1a2e !important; }
            div[data-testid="stAlert"] p { color: #1a1a2e !important; }
            [data-testid="stMetricValue"] div { color: #0077cc !important; }
            hr { border-color: #d1d9e6 !important; }
        }
        </style>
    """,
}

# ── Apply selected theme ──────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"
st.markdown(THEME_CSS[st.session_state.theme], unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🩺 MediScan AI")


st.caption("AI-powered medical image segmentation and preliminary report generation.")
st.warning(
    "⚠️ **Disclaimer:** This tool is NOT a substitute for professional medical diagnosis. Always consult a licensed physician.",
    icon="⚠️",
)
st.divider()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # ── Theme Toggle ──────────────────────────────────────────────────────────
    theme_icons = {"Dark": "🌙 Dark", "Light": "☀️ Light", "System": "💻 System"}
    selected_theme = st.radio(
        "Theme",
        options=list(theme_icons.keys()),
        format_func=lambda x: theme_icons[x],
        index=list(theme_icons.keys()).index(st.session_state.theme),
        horizontal=True,
        key="theme_radio",
    )
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()

    st.divider()

    scan_type = st.selectbox(
        "Scan Type",
        options=["skin_lesion", "chest_xray", "ultrasound"],
        format_func=lambda x: {
            "skin_lesion": "🔬 Skin Lesion",
            "chest_xray": "🫁 Chest X-Ray",
            "ultrasound": "🔊 Breast Ultrasound",
        }[x],
    )

    st.divider()
    st.subheader("📌 How It Works")
    with st.container(border=True):
        st.markdown("""
        1. Upload a medical scan image
        2. SegFormer-b2 (fine-tuned) segments the region of interest
        3. Region statistics are extracted
        4. Groq LLM generates a structured report
        5. Download the PDF report
        """)

    st.divider()
    st.subheader("📂 Sample Images")
    with st.container(border=True):
        st.markdown("No image? Try a public sample from [ISIC Archive](https://www.isic-archive.com/)")

# ── DICOM loader ─────────────────────────────────────────────────────────────
def load_dicom(file) -> Image.Image:
    """Convert a DICOM file to a displayable RGB PIL Image."""
    try:
        import pydicom
    except ImportError:
        st.error("pydicom not installed. Run: pip install pydicom")
        st.stop()

    ds  = pydicom.dcmread(file)
    arr = ds.pixel_array.astype(np.float32)

    # handle multi-frame DICOM (take first frame)
    if arr.ndim == 3 and arr.shape[0] > 3:
        arr = arr[0]

    # normalize to 0–255
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
    arr = arr.astype(np.uint8)

    # convert to RGB (DICOM is often grayscale)
    img = Image.fromarray(arr)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def load_image(uploaded_file) -> Image.Image:
    """Load any supported format into a PIL RGB Image."""
    filename = uploaded_file.name.lower()
    if filename.endswith(".dcm"):
        return load_dicom(uploaded_file)
    else:
        return Image.open(uploaded_file).convert("RGB")


# ── Main Area ─────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a medical scan image",
    type=["jpg", "jpeg", "png", "tiff", "tif", "bmp", "webp", "dcm"],
    help="Supported: JPG, PNG, TIFF, BMP, WEBP, DICOM (.dcm)",
)

if uploaded_file is not None:
    # ── Load image (handles DICOM + standard formats) ─────────────────────────
    try:
        image = load_image(uploaded_file)
    except Exception as e:
        st.error(f"Could not read image: {e}. Please upload a valid medical scan.")
        st.stop()

    # ── Input validation ──────────────────────────────────────────────────────
    img_array = np.array(image)
    img_std   = float(img_array.std())

    if img_std < 5.0:
        st.error("⚠️ Image has very low contrast — this doesn't look like a medical scan. "
                 "Please upload a dermatoscopy, X-ray, or ultrasound image.")
        st.stop()

    if image.size[1] < 50:
        st.error("⚠️ Image height is too small. Please upload a proper medical scan.")
        st.stop()

    # DICOM badge
    if uploaded_file.name.lower().endswith(".dcm"):
        st.success("✅ DICOM file loaded successfully.")

    # resize if too large (speeds up CPU inference)
    MAX_SIZE = 1024
    if max(image.size) > MAX_SIZE:
        ratio    = MAX_SIZE / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
        st.info(f"Image resized to {new_size} for faster processing.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, width="stretch")

    # ── Segmentation ─────────────────────────────────────────────────────────
    with st.spinner("🔍 Running segmentation (SegFormer-b2 fine-tuned)..."):
        try:
            masks     = generate_masks(image, scan_type=scan_type)
            best_mask = select_best_mask(masks, image_size=(image.size[1], image.size[0]))

            if best_mask is None:
                st.error("Segmentation returned no mask. Check that the model file exists in models/.")
                st.stop()

            mask_array    = best_mask["segmentation"]
            overlay_image = overlay_mask_on_image(
                image, mask_array, color=(0, 255, 100), alpha=0.45
            )
            stats          = extract_region_stats(image, mask_array)
            interpretations = interpret_stats(stats, scan_type)

        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Segmentation failed: {e}")
            st.stop()

    # ── Model info ────────────────────────────────────────────────────────────
    with st.expander("🔬 Model Details"):
        mask_stats = get_all_mask_stats(masks, (image.size[1], image.size[0]))
        dice_scores = {"skin_lesion": 0.9166, "ultrasound": 0.8176, "chest_xray": 0.9643}
        iou_scores  = {"skin_lesion": 0.8474, "ultrasound": 0.7065, "chest_xray": 0.9313}
        st.write(f"**Model:** SegFormer-b2 fine-tuned on "
                 f"{'ISIC 2018' if scan_type == 'skin_lesion' else 'BUSI' if scan_type == 'ultrasound' else 'Montgomery+Shenzhen'}")
        st.write(f"**Val Dice:** {dice_scores[scan_type]:.4f} | **Val IoU:** {iou_scores[scan_type]:.4f}")
        st.write(f"**Segmented area:** {mask_stats[0]['area_pct']}% of image")
        st.write(f"**Model confidence:** {mask_stats[0]['score']:.3f}")

    with col2:
        st.subheader("Segmentation Overlay")
        st.image(overlay_image, width="stretch")

    st.divider()

    # ── Stats Display ─────────────────────────────────────────────────────────
    st.subheader("📊 Region Statistics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Area Coverage", f"{stats['area_pct']}%")
    m2.metric("Location", stats["location"].title())
    m3.metric("Shape", interpretations["shape"])
    m4.metric("Border", interpretations["border"].split()[0].title())

    with st.expander("🔢 Full Quantitative Measurements"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"**Mean Intensity:** {stats['mean_intensity']} / 255")
            st.write(f"**Std Intensity:** {stats['std_intensity']}")
            st.write(f"**Contrast Ratio:** {stats['contrast_ratio']}")
            st.write(f"**SAM2 Score:** {round(best_mask['score'], 3)}")
        with col_b:
            st.write(f"**Irregularity Index:** {stats['irregularity']}")
            st.write(f"**Solidity:** {stats['solidity']}")
            st.write(f"**BBox Aspect Ratio:** {stats['bbox_wh_ratio']}")
            st.write(f"**Total Masks Generated:** {len(masks)}")

    st.divider()

    # ── Report Generation ─────────────────────────────────────────────────────
    st.subheader("📋 AI-Generated Report")

    with st.spinner("📝 Generating clinical report via Groq (GPT-OSS-120B)..."):
        result = generate_report(stats, interpretations, scan_type)

    if not result["success"]:
        st.warning(f"Report generation issue: {result.get('error', 'Unknown error')}. Showing fallback report.")

    report = result["report"]

    # confidence color
    confidence_colors = {"High": "green", "Moderate": "orange", "Low": "red"}
    conf = report.get("confidence", "Low")
    conf_color = confidence_colors.get(conf, "gray")

    # display report cards
    col_r1, col_r2 = st.columns([2, 1])

    with col_r1:
        st.markdown("#### 🔍 Findings")
        st.info(report.get("findings", "N/A"))

        st.markdown("#### 💡 Impression")
        st.success(report.get("impression", "N/A"))

        st.markdown("#### 📋 Recommendations")
        st.write(report.get("recommendations", "N/A"))

    with col_r2:
        st.markdown(
    f"""
    <div style="font-size: 24px; font-weight: bold;">
        🎯 AI Confidence:
        <span style="color:{conf_color};">{conf}</span>
    </div>
    """,
    unsafe_allow_html=True
)

        st.markdown("#### 🔄 Differential Considerations")
        for item in report.get("differential", []):
            st.markdown(f"• {item}")

    st.markdown("---")
    st.error(f"⚠️ {report.get('disclaimer', '')}")

    # ── PDF Export ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📥 Export Report")

    with st.spinner("Generating PDF..."):
        pdf_bytes = generate_pdf(
            report=report,
            stats=stats,
            original_image=image,
            overlay_image=overlay_image,
            scan_type=scan_type,
        )

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name="mediscan_report.pdf",
        mime="application/pdf",
        type="primary",
    )

else:
    with st.container(border=True):
        st.info("👆 Upload a medical scan image to get started.")
        st.markdown("""
        **Supported scan types:**
        - 🔬 **Skin Lesion** — dermatoscopy or clinical skin lesion photographs
        - 🫁 **Chest X-Ray** — frontal chest radiographs
        - 🔊 **Breast Ultrasound** — breast ultrasound images

        **Supported formats:** JPG, PNG, TIFF, BMP, WEBP, DICOM (.dcm)

        **Please upload a clean scan image.**
        """)