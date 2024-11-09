# %%
import streamlit as st
import pandas as pd
import joblib
def custom_score_func(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    new_score = 0.7 * recall + 0.3 * acc
    return new_score


loaded_model = joblib.load('loan_approval_model2.pkl')  


st.title("Loan Application")

st.write("""
This app predicts whether a loan application will be approved or not based on the input details.
""")


st.header("Enter your Details:")


loan_amnt = st.number_input("Loan Amount", min_value=0, value=10000)
funded_amnt = st.number_input("Funded Amount", min_value=0, value=10000)
funded_amnt_inv = st.number_input("Funded Amount by Investors", min_value=0, value=9500)
term = st.selectbox("Term", options=[36, 60])
int_rate = st.slider("Interest Rate (%)", min_value=5.0, max_value=30.0, value=13.5)
installment = st.number_input("Installment", min_value=0.0, value=325.0)
home_ownership = st.selectbox("Home Ownership", options=['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
annual_inc = st.number_input("Annual Income", min_value=0, value=60000)
verification_status = st.selectbox("Verification Status", options=['Verified', 'Source Verified', 'Not Verified'])
purpose = st.selectbox("Purpose of Loan", options=['debt_consolidation', 'credit_card', 'home_improvement', 'other'])
dti = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=15.0)
open_acc = st.number_input("Number of Open Accounts", min_value=0, value=10)
revol_bal = st.number_input("Revolving Balance", min_value=0, value=5000)
revol_util = st.slider("Revolving Utilization Rate (%)", min_value=0.0, max_value=100.0, value=30.0)
total_acc = st.number_input("Total Number of Accounts", min_value=0, value=25)
recoveries = st.number_input("Recoveries", min_value=0.0, value=0.0)


if st.button("Submit"):
    # Create a DataFrame with input values for prediction
    input_data = pd.DataFrame({
        'loan_amnt': [loan_amnt],
        'funded_amnt': [funded_amnt],
        'funded_amnt_inv': [funded_amnt_inv],
        'term': [term],
        'int_rate': [int_rate],
        'installment': [installment],
        'home_ownership': [home_ownership],
        'annual_inc': [annual_inc],
        'verification_status': [verification_status],
        'purpose': [purpose],
        'dti': [dti],
        'open_acc': [open_acc],
        'revol_bal': [revol_bal],
        'revol_util': [revol_util],
        'total_acc': [total_acc],
        'recoveries': [recoveries]
    })

    
    prediction = loaded_model.predict(input_data)
    probabilities = loaded_model.predict_proba(input_data)[0]  # Probability of each class

    
    if prediction[0] == 1:
        st.success(f"Loan Approved! Probability: {probabilities[1]:.2%}")
    else:
        st.error(f"Loan Not Approved. Probability: {probabilities[1]:.2%}")
    
   
    st.write("**Probability of Not Approved:**", f"{probabilities[0]:.2%}")
    st.write("**Probability of Approved:**", f"{probabilities[1]:.2%}")


# %%
