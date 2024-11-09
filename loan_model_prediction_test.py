import joblib
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score

def custom_score_func(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    new_score = 0.7 * recall + 0.3 * acc 
    return new_score

loaded_model = joblib.load('loan_approval_model2.pkl')

new_data = pd.DataFrame({
    'loan_amnt': [10000],
    'funded_amnt': [10000],
    'funded_amnt_inv': [9500],
    'term': [36],           
    'int_rate': [13.5],
    'installment': [325.0],
    'home_ownership': ['RENT'], 
    'annual_inc': [60000],
    'verification_status': ['Verified'],
    'purpose': ['debt_consolidation'],
    'dti': [15.0],
    'open_acc': [10],                                                   
    'revol_bal': [5000],
    'revol_util': [30.0],
    'total_acc': [25],
    'recoveries': [0]
})

prediction = loaded_model.predict(new_data)
print("Loan Approval Prediction:", "Approved" if prediction[0] == 1 else "Not Approved")


probabilities = loaded_model.predict_proba(new_data)

print("Probability of Not Approved:", probabilities[0][0])
print("Probability of Approved:", probabilities[0][1])
