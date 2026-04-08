import streamlit as st
import pandas as pd


st.title('Streamlit Project')



st.write("News Classification")

#taking input from users
data  = st.text_area("Enter New for the Classification")

if st.button('Submit'):

    d = {"news": [data]}
    df = pd.DataFrame(d)
    st.write(df)