import streamlit as st

#adding title of your app
st.title("My First Streamlit App for Learning Purposes and delegating to my brilliant Student Saima")
#adding simple text to your app
st.write("This is my first Streamlit app. I am learning how to use it.")
#adding input text box to your app
st.text_input("Enter your name:")
#adding input number btw 01 to 100 to your app
st.slider("Select a number between 1 and 100:", min_value=1, max_value=100, value=50)
#print the number entered by the user
st.write("You entered:", st.number_input("Enter a number between 1 and 100:", min_value=1, max_value=100))


#adding a buton
if st.button("Good to See You!"):
    #action to be performed when button is clicked
    st.write("That's my Plesure!")
else:
    st.write('Goodbye')
#add radio button with options
st.radio("Select an option:", ("Darama", "Comdey", "Movie"))
#adding a checkbox
st.checkbox("Why you are here to learning Data Science?")
#adding a select box with options
st.selectbox("Select an option:", ("Passion", "Time Pass", "Yeah kuch kar dhakana ka irada hai"))
#add fropdown menu with options
#option=st.selectbox("How would like to be connected:", 
#                    ("Email", "Home Contact", "Linkden"))
#add a dropdown list on the side bar
st.sidebar.selectbox("Select an option:", ("Email", "Home Contact", "Linkden"))

#add your cell phone number or email address
st.sidebar.text_input("Enter your cell phone number or email address:")
#add file uploader to upload a file
st.file_uploader("Upload a file:", type=["csv", "txt", "pdf"])

#create a line plot using matplotlib
import pandas as pd
data = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [10, 20, 30, 40, 50]
})
st.line_chart(data)
