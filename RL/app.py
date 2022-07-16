from email import header
from unicodedata import name
import streamlit as st
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import pandas as pd

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title('Welcome to Skarb!')

with model_training:
    st.header('Start investing in a smarter way')
    #st.text('Fill the form to start the risk profiling process')
    sel_col, disp_col = st.columns(2)
    with sel_col:
        st.subheader('Your risk profile has a score of 64 !', anchor=None)
        st.progress(64)
        st.write("That is your recomended strategy:")
        chart = pd.read_csv('./chart.csv')

        st.bar_chart(chart.tail(1))




    side = st.sidebar
    with side:
        with st.form("my_form"):
            st.write("Tell us something about you")
            age = st.slider("How old are you?", min_value = 18, max_value = 120)
            marital_status = st.selectbox('What is your marital status?',options=['Single','Married'])
            education = st.selectbox('What is the highest educational level you have reached?',options=['Never went to school','Elementary school','Highschool','College','FP','Bachelor','Master','PHD'])
            st.write("Lets talk about money  💸")
            income = st.number_input('What is your total monthly income?',min_value=0,value=0,step=100)
            expenses = st.number_input('What are the total monthly expenses of your household?',min_value=0,value=0,step=100) #passar a negatiu
            cash_acc = st.number_input('How much money do you have saved in your bank accounts?',min_value=0,value=0,step=100)
            has_house = st.selectbox("Do you own a house?",options=["Yes","No"])
            val_house = st.number_input('If you own a house, what is its current value approximately?',min_value=0,value=0,step=10000)
            debts = st.number_input("In the event that you have some type of loan or pending debt, how much money do you still have to pay?",min_value=0,value=0,step=1000)
            st.write("In case you have a pension plan...")
            pension_plan_year = st.number_input('How much money do you allocate annually to pension plans?',min_value=0,value=0,step=100)
            val_pension_plan = st.number_input('How much are all your pension plans currently worth?',min_value=0,value=0,step=1000)
            st.write("Lets talk about investments")
            val_investments = st.number_input('What is the total value of your investments?',min_value=0,value=0,step=1000)
            income_investments_year = st.number_input('What is the annual income from your investments?',min_value=0,value=0,step=1000)
            st.write("Lets talk about financial behaviour")
            lottery = st.slider("If you won the lottery tomorrow, what percentage of the prize would you spend in a year?", min_value = 0, max_value = 100)
            risk = st.selectbox("How much financial risk are you willing to take when you make an investment?",options=['High risks for High profits','Medium - High risks for Medium - High profits','Medium - Low risks for Medium - Low profits', "I'm not willing to take any risk"])

            # Every form must have a submit button.
            submitted = st.form_submit_button("Submit")
            if submitted:
                st.balloons()
                print(submitted)

  
    
    