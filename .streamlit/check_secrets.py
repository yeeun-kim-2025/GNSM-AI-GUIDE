import streamlit as st

st.write("📂 st.secrets 내용:", dict(st.secrets))
st.write("🔑 OPENAI_API_KEY:", st.secrets.get("OPENAI_API_KEY", "없음"))
