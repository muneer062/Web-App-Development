# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add title
st.title("Data Analysis Application")

# Add subheader
st.subheader('This is a simple data analysis application created by @Codenics')

# Dropdown to select a dataset
dataset_options = ['penguins', 'iris', 'titanic', 'tips', 'diamonds']
selected_dataset = st.selectbox('Select a dataset:', dataset_options)

# Load the selected dataset
if selected_dataset:
    df = sns.load_dataset(selected_dataset)
    st.write(f"Loaded `{selected_dataset}` dataset:")
    st.write(df)

# Button to upload a custom dataset
uploaded_file = st.file_uploader("Or upload your own dataset (CSV format):", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded dataset:")
    st.write(df)
#display the number of rows and coulmns from the selected dataset
st.write(f"Number of rows: {df.shape[0]}")
st.write(f"Number of columns: {df.shape[1]}")
# Display the first few rows of the dataset
st.write("First few rows:")

#display the columns name of the selected dataset
st.write("Columns names:")
st.write(df.columns.tolist())
