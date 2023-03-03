import streamlit as st

st.sidebar.markdown(" :)( ")

st.title("Spending Analysis")

import pandas as pd

data = pd.read_csv("static/File_Name.csv")

st.write("Dataset Overview")

st.write(data)

st.write("Graphs are being generated using Tableau")
st.write("https://prod-apnortheast-a.online.tableau.com/#/site/gauravsarkar/workbooks/847253?:origin=card_share_link")

st.image("screenshots/Screenshot from 2023-01-11 03-01-53.png")

st.image("screenshots/Screenshot from 2023-01-11 03-01-43.png")

st.image("screenshots/Screenshot from 2023-01-11 03-01-33.png")

st.image("screenshots/Screenshot from 2023-01-11 03-01-18.png")

st.image("screenshots/Screenshot from 2023-01-11 03-01-07.png")

st.image("screenshots/Screenshot from 2023-01-11 03-00-54.png")

st.image("screenshots/Screenshot from 2023-01-11 03-00-37.png")

st.image("screenshots/Screenshot from 2023-01-11 03-00-19.png")
