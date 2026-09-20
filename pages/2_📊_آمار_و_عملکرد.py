import streamlit as st
import pandas as pd

st.set_page_config(page_title="آمار و عملکرد", page_icon="📊")

st.markdown(
    """
    <h1 style='text-align: right;'>📊 آمار و عملکرد سیستم</h1>
    """,
    unsafe_allow_html=True,
)
st.divider()

st.markdown("### مقایسه دو روش خلاصه‌سازی")
st.write("نتایج ارزیابی با معیار ROUGE روی ۵۵ سند اعتبارسنجی از مجموعه داده ویکی‌پدیای فارسی:")

data = {
    "روش": ["لایه ۱ - استخراجی (TF-IDF)", "لایه ۲ - انتزاعی (mT5 Fine-tuned)"],
    "ROUGE-1": [20.49, 26.11],
    "ROUGE-2": [15.97, 15.37],
    "ROUGE-L": [20.22, 26.11],
}
df = pd.DataFrame(data)

st.dataframe(df, use_container_width=True, hide_index=True)

st.write("")
st.markdown("### نمودار مقایسه‌ای")
chart_df = df.set_index("روش")
st.bar_chart(chart_df)

st.write("")
st.markdown("### آمار مجموعه داده")
col1, col2, col3 = st.columns(3)
col1.metric("تعداد اسناد", "547")
col2.metric("مجموع کلمات", "۷۵۰,۸۱۸")
col3.metric("میانگین طول سند", "۱۳۶۲ کلمه")

st.write("")
st.success("✅ هر دو روش از هدف پروپوزال (ROUGE-2 بالای ۱۵٪) عبور کرده‌اند.")

st.write("")
st.markdown("### جمع‌بندی")
st.write(
    "لایه انتزاعی در معیارهای ROUGE-1 و ROUGE-L بهبود قابل‌توجهی نسبت به Baseline "
    "استخراجی نشان داد. در معیار ROUGE-2 نتیجه‌ای نزدیک به Baseline حاصل شد که با "
    "افزایش حجم داده آموزشی در آینده قابل بهبود است."
)