import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from datetime import datetime
from extractive_summarizer import extractive_summarize, extract_keywords
from persian_preprocessing import normalize_text, sentence_segment
st.set_page_config(
    page_title="خلاصه‌سازی هوشمند متن فارسی",
    page_icon="📝",
    layout="wide",
)

st.markdown(
    """
    <style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;700&display=swap');

html, body, [class*="css"], p, div, span, h1, h2, h3, h4, textarea, input, button, label {
    font-family: 'Vazirmatn', sans-serif;
}    label, .stMarkdown p, .stRadio label, .stSlider label, h1, h2, h3, h4 {
        direction: rtl;
        text-align: right;
    }
    .stTextArea textarea {
        unicode-bidi: plaintext;
        text-align: start;
    }
    div[data-testid="stAlert"] {
        direction: rtl;
        text-align: right;
    }
    div[data-testid="stAlert"] > div {
        flex-direction: row-reverse;
        text-align: right;
    }
    div[data-testid="stMetric"] {
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []


@st.cache_resource
def load_transformer_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def abstractive_summarize(text, max_length=150):
    tokenizer, model = load_transformer_model()
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=30,
        num_beams=4,
        no_repeat_ngram_size=2,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# ---------- نوار کناری: تنظیمات ----------
with st.sidebar:
    st.markdown("## ⚙️ تنظیمات")
    st.divider()

    input_method = st.radio("روش ورودی:", ["تایپ مستقیم", "آپلود فایل TXT"])

    method = st.radio(
        "روش خلاصه‌سازی:",
        ["استخراجی (سریع)", "Transformer (کندتر، انتزاعی)", "مقایسه هر دو روش"],
    )

    ratio = st.slider("نسبت خلاصه (روش استخراجی):", min_value=10, max_value=30, value=20, step=5)
    ratio = ratio / 100

    if "Transformer" in method or "مقایسه" in method:
        st.warning("⏱️ روش Transformer روی این سیستم (بدون GPU) ممکن است ۳۰ تا ۶۰ ثانیه طول بکشد.")

    st.divider()
    st.markdown("### درباره پروژه")
    st.caption(
        "سیستم هوشمند خلاصه‌سازی خودکار متون طولانی فارسی و انگلیسی، "
        "مبتنی بر مدل ترکیبی استخراجی–انتزاعی."
    )
    st.caption(f"تعداد خلاصه‌های تولیدشده در این نشست: {len(st.session_state.history)}")

# ---------- هدر اصلی ----------
st.divider()
st.markdown(
    """
    <div style='background: linear-gradient(135deg, #1C2333 0%, #0E1117 100%); padding: 30px; border-radius: 15px; margin-bottom: 20px;'>
        <h1 style='text-align: right; margin-bottom: 5px;'>📝 سیستم هوشمند خلاصه‌سازی متن</h1>
        <p style='text-align: right; color: gray; margin-bottom: 20px;'>خلاصه‌سازی خودکار متون طولانی فارسی و انگلیسی با معماری ترکیبی استخراجی–انتزاعی</p>
        <div style='display: flex; justify-content: flex-end; gap: 40px; direction: rtl;'>
            <div style='text-align: center;'>
                <div style='font-size: 24px; font-weight: bold; color: #2E86DE;'>547</div>
                <div style='color: gray; font-size: 13px;'>سند آموزشی</div>
            </div>
            <div style='text-align: center;'>
                <div style='font-size: 24px; font-weight: bold; color: #2E86DE;'>2</div>
                <div style='color: gray; font-size: 13px;'>الگوریتم هوشمند</div>
            </div>
            <div style='text-align: center;'>
                <div style='font-size: 24px; font-weight: bold; color: #2E86DE;'>%16</div>
                <div style='color: gray; font-size: 13px;'>ROUGE-2</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### ورودی متن")

    user_text = ""
    if input_method == "تایپ مستقیم":
        user_text = st.text_area("متن خود را اینجا وارد کنید:", height=280, placeholder="متن فارسی خود را اینجا بنویسید یا جای‌گذاری کنید...")
    else:
        uploaded_file = st.file_uploader("فایل TXT را انتخاب کنید", type=["txt"])
        if uploaded_file is not None:
            user_text = uploaded_file.read().decode("utf-8")
            st.text_area("متن بارگذاری‌شده:", value=user_text, height=280)

    summarize_button = st.button("🔍 تولید خلاصه", type="primary", use_container_width=True)

with col2:
    st.markdown("#### خلاصه تولیدشده")

    if summarize_button:
        word_count = len(user_text.split())
        if not user_text.strip():
            st.warning("⚠️ لطفاً ابتدا متنی وارد کنید.")
        elif word_count < 30:
            st.warning(f"⚠️ متن وارد‌شده خیلی کوتاه است ({word_count} کلمه). حداقل ۳۰ کلمه لازم است.")
        else:
            normalized = normalize_text(user_text)
            sentences = sentence_segment(normalized)

            if len(sentences) < 3:
                st.error("متن باید حداقل ۳ جمله داشته باشد.")
            else:
                extractive_result = None
                abstractive_result = None

                if method in ["استخراجی (سریع)", "مقایسه هر دو روش"]:
                    with st.spinner("در حال تولید خلاصه استخراجی..."):
                        summary_sentences = extractive_summarize(sentences, ratio=ratio)
                        extractive_result = " ".join(summary_sentences)

                if method in ["Transformer (کندتر، انتزاعی)", "مقایسه هر دو روش"]:
                    with st.spinner("در حال تولید خلاصه با مدل Transformer... (ممکن است تا ۶۰ ثانیه طول بکشد)"):
                        abstractive_result = abstractive_summarize(normalized)

                st.success("✅ خلاصه با موفقیت تولید شد.")

                if extractive_result:
                    st.markdown("**🔹 روش استخراجی:**")
                    st.markdown(
                        f"""<div dir="auto" style='background-color:#1C2333; padding:16px; border-radius:10px; border-right: 4px solid #2E86DE; margin-bottom:16px;'>{extractive_result}</div>""",
                        unsafe_allow_html=True,
                    )

                if abstractive_result:
                    st.markdown("**🔸 روش Transformer (انتزاعی):**")
                    st.markdown(
                        f"""<div dir="auto" style='background-color:#1C2333; padding:16px; border-radius:10px; border-right: 4px solid #E67E22;'>{abstractive_result}</div>""",
                        unsafe_allow_html=True,
                    )
                if extractive_result:
                    keywords = extract_keywords(sentences)
                    if keywords:
                        tags_html = " ".join(
                            f"<span style='background-color:#2E86DE; color:white; padding:4px 12px; border-radius:15px; margin:3px; display:inline-block; font-size:13px;'>{kw}</span>"
                            for kw in keywords
                        )
                        st.markdown(
                            f"<div style='direction:rtl; text-align:right; margin-top:10px;'>{tags_html}</div>",
                            unsafe_allow_html=True,
                        )
                if extractive_result:
                    st.write("")
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("📄 جملات اصلی", len(sentences))
                    col_b.metric("✂️ جملات خلاصه", len(summary_sentences))
                    col_c.metric("📉 فشرده‌سازی", f"{(1 - len(summary_sentences)/len(sentences))*100:.0f}%")


                final_summary_for_download = extractive_result or abstractive_result
                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "summary": final_summary_for_download,
                })

                st.write("")
                st.download_button(
                    label="⬇️ دانلود خلاصه (TXT)",
                    data=final_summary_for_download,
                    file_name="خلاصه.txt",
                    mime="text/plain",
                )
    else:
        st.info("💡 برای مشاهده خلاصه، ابتدا متن را وارد کرده و دکمه «تولید خلاصه» را بزنید.")