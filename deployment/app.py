# import libraries
import streamlit as st
import deployment.eda as eda
import deployment.prediction as prediction

# navigation section
navigation = st.sidebar.selectbox("Choose Page", ("Sentiment Analysis","EDA"))

# page
if navigation == "Sentiment Analysis":
    prediction.run()
else:
    eda.run()