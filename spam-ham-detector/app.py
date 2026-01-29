import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.predict import predict_message

st.set_page_config(page_title="Spam-Ham Detector", layout="centered")

st.title(" Spam vs Ham Detector")

text = st.text_area("Enter a message")

if st.button("Detect"):
    if text.strip() == "":
        st.warning("Please enter a message")
    else:
        result = predict_message(text)

        if result == 1:
            st.error(" Spam Message")
        else:
            st.success(" Ham Message")

        chart_df = pd.DataFrame({
            "Class": ["Ham", "Spam"],
            "Confidence": [0.7, 0.3] if result == 0 else [0.3, 0.7]
        })

        fig, ax = plt.subplots()
        sns.barplot(x="Class", y="Confidence", data=chart_df, ax=ax)
        st.pyplot(fig)
