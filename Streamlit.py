#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
st.set_page_config(page_title='My NBA Award Predictions', layout='wide' )
#Header

st.subheader("Using Linear Regression to predict 3 NBA Awards in real time. Will add other awards soon")
st.title("MVP Prediction")

df = pd.read_csv('mvp_leaders.csv')
df = df.drop('Unnamed: 0', axis=1)

st.dataframe(df, height=300)



st.title("DPOY Prediction")

df1 = pd.read_csv('dpoy_leaders.csv')
df1 = df1.drop('Unnamed: 0', axis=1)

st.dataframe(df1, height=300)


st.title("ROTY Prediction")

df2 = pd.read_csv('roty_leaders.csv')
df2 = df2.drop('Unnamed: 0', axis=1)

st.dataframe(df2, height=300)

# streamlit run untitled.py


# In[ ]:




