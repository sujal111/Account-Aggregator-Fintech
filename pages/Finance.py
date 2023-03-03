import streamlit as st
st.title("UPI CASHFLOW & WEALTH MANAGEMENT")
st.image("static/1 (1).png")

st.header("1.Payment History")
st.header("2.Credit Score Governing Factors")
st.header("3.Fraud & Transaction Data Frequency")




st.title("1.Payment history")
st.image("static/5.png")

st.header("1.1 Statistical Formulae ")


st.write("Consider Daily avg income of a vegetable vendor in India, this would act as a metadata")
st.write("Take Montly income of a vegetable vendor")
st.write("Net Income=AvgCredit-AvgDebit")
st.write("If netincome>Dailyavgincome , then it’s a positive sign and it's a good UPI cashflow quality")
st.write("If netincome>(Dailyavgincome+amount and frequency of Penalties), then it shows a GOOD UPI CASHFLOW QUALITY ")



st.header("1.2 Analysis for non salary folks")
st.image("static/3.PNG")
st.header("Statistical Formulae for non salried class")
st.write("check average monthly icome of every quarter - 3 months,6 months, 9 months")
st.write("check volatility of the amount")
st.write("compare with banking metadata")

st.write("if average montly income of every quarter>averge income of a vegetable vendor in India,then it's Good quality UPI cashflow")
st.write("Also check if he/she is deliberately doing tinkering with the financial transaction in order to improve the quality of UPI Cashflow. Compare since  the time account has been created")
st.image("static/100.png")
st.write("If it falls in true positive, then it shows Good UPI cashflow")

st.title("2.CREDITSCORE GOVERNING FACTORS ")
st.image("static/4.png")
st.header("2.1 EMIs")
st.write("Factors considered")
st.write("Debited on every specific date and amount Penalties Total no of penalties totalObligationAmountLast90DaystotalObligationAmountLast3Months totalObligationAmountLast6Months avgMonthlyObligationLast6Months volatilityOfAmount")
st.write("If debited on specific date =Yes  &&  amountdebited<incomeThen shows positive sign If debited on specific date =Yes  &&  amountdebited>income Then okay take care If debited on specific date =No  &&  amountdebited>income && no of penalties are more  Then UPI cashflow quality is bad ")


st.header("2.2 Bounce")
st.image("static/11.png")
st.image("static/12.png")
st.header("Formulae for checking bounce")
st.write(" Total reversal + Total charge amount+ also weight of no of bounces>incomThen Red signaDefaults")

st.header("2.3 Few financial factors for credit score for improving the quality of UPI Cashflow")
st.header("2.3.1 Repay history")
st.write("If repayment is on time, then good credit score. If not , then bad credit score")

st.write("If the repayment is done on a specific date, it shows good finance habits and eventually good credit score ")
st.header("2.3.2 Mix of loans - unsecured loans")
st.write("Given the possibility of unsecured loans such as loans without collateral, it leads to bad credit score ")
st.header("2.3.3 Credit limit")
st.write("One can't spend more than credit limit ,if one does so, it shows bad credit score")

st.header("2.3.4 Credit History Length")
st.write("Length of the credit score is considered as the significant parameter to improve the quality of credit scoreMore the time, more the transactions and better the results.")

st.header("2.3.5 Account related to a person who is involved in default")
st.write("It affects the credit score ")

st.image("static/100.png")
st.write("TP shows that bounce is less and shows Good UPI cashflow")

st.title("3. FRAUD")
st.image("static/spike.png")
st.write("As seen on the chart, two bars are marked with a red background color which is our way to indicate an anomaly visually. If you look at the error count before 10/19/2019 they are limited to only a few instances per day, while we see a sudden spike on that day.Again, the bug is indicated by a new spike on 10/22/2019. This sudden spike shows potential fraud or deliberate tampering of finance data is being conducted")
