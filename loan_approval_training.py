import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder  
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder, OneHotEncoder
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import recall_score, accuracy_score, make_scorer, roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import joblib


file_path = "Downloads\loan.csv\loan.csv"
loan_data = pd.read_csv(file_path)
loan_data_copy = loan_data.copy()
loan_data_copy['loan_approved'] = loan_data_copy['loan_status'].apply(lambda x: 1 if x in ['Fully Paid', 'Current'] else 0)
X=loan_data_copy.drop(columns=[
    "total_il_high_credit_limit", "tot_hi_cred_lim", "total_bc_limit", "id", "member_id",
    "grade", "sub_grade", "emp_title", "emp_length", "pymnt_plan", "url", "desc", "title",
    "zip_code", "addr_state", "delinq_2yrs", "mths_since_last_delinq", "mths_since_last_record",
    "pub_rec", "initial_list_status", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "collection_recovery_fee", "next_pymnt_d",
    "last_credit_pull_d", "collections_12_mths_ex_med", "mths_since_last_major_derog", "policy_code",
    "annual_inc_joint", "dti_joint", "verification_status_joint", "acc_now_delinq", "tot_coll_amt",
    "tot_cur_bal", "open_acc_6m", "open_il_6m", "open_il_12m", "open_il_24m", "mths_since_rcnt_il",
    "total_bal_il", "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc", "all_util", "total_rev_hi_lim",
    "inq_fi", "total_cu_tl", "inq_last_12m", "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy",
    "bc_util", "chargeoff_within_12_mths", "delinq_amnt", "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op",
    "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc", "mths_since_recent_bc", "mths_since_recent_bc_dlq",
    "mths_since_recent_inq", "mths_since_recent_revol_delinq", "num_accts_ever_120_pd", "num_actv_bc_tl",
    "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl", "num_op_rev_tl", "num_rev_accts",
    "num_rev_tl_bal_gt_0", "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies", "tax_liens",
    "total_bal_ex_mort","loan_status","application_type","last_pymnt_d","last_pymnt_amnt","issue_d","earliest_cr_line","inq_last_6mths"
])
y = X.pop("loan_approved")

X['funded_amnt_inv'] = pd.to_numeric(X['funded_amnt_inv'], errors='coerce')
X['installment'] = pd.to_numeric(X['installment'], errors='coerce')
X['annual_inc'] = pd.to_numeric(X['annual_inc'], errors='coerce')
X['dti'] = pd.to_numeric(X['dti'], errors='coerce')
X['recoveries'] = pd.to_numeric(X['recoveries'], errors='coerce')
X['int_rate'] = pd.to_numeric(X['int_rate'], errors='coerce')
X['term'] = pd.to_numeric(X['term'], errors='coerce')
X['revol_util'] = pd.to_numeric(X['revol_util'], errors='coerce')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_columns = ['home_ownership', 'verification_status', 'purpose']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns)
    ],
    remainder='passthrough',
    force_int_remainder_cols=False  # Opt-in to new behavior to avoid warning
)

# Define the pipeline with the preprocessor and classifier
rf_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('rf', RandomForestClassifier(random_state=42))
])

def custom_score_func(y_true, y_pred):
  recall = recall_score(y_true, y_pred)
  acc = accuracy_score(y_true, y_pred)
  new_score = 0.7*recall + 0.3*acc 
  return new_score

custom_score = make_scorer(custom_score_func)


param_grid = {
    # Trying out several different class weight dictionaries
    'rf__class_weight': ['balanced'] + [{0:1, 1:w} for w in range(1, 20)] #20 options
}                                                                                       
# Performing grid search to identify optimal hyperparameter combination
rf_search = GridSearchCV(rf_pipe,
                         param_grid,
                         cv=5,                                           
                         scoring=custom_score,
                         n_jobs=-1, 
                         verbose=1)

rf_search.fit(X_train, y_train)

rf_search.best_params_


ConfusionMatrixDisplay.from_estimator(
    rf_search, X_train, y_train, display_labels=['Loan_Approved', 'Not_Approved'])


rf_probs = rf_search.predict_proba(X_train)[:,1]
rf_fpr, rf_tpr, rf_threshold = roc_curve(y_train, rf_probs)
rf_auc = roc_auc_score(y_train,rf_probs)
plt.plot(rf_fpr, rf_tpr, marker='.', label='Random forest (AUROC = %0.3f)' % rf_auc)
plt.title('Train ROC Plot')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend() 
plt.show()


probs = rf_search.predict_proba(X_train)[:,1]
fpr, tpr, thresh = roc_curve(y_train, probs)
threshold = max(thresh[i] for i in range(len(thresh)) if tpr[i] == 1) 
safe_thresholds = []
for i in range(len(thresh)):
    if tpr[i] == 1:
       safe_thresholds.append(thresh[i])
threshold = max(safe_thresholds)
safe_pred = (probs >= threshold).astype(int) #conditional returns True/False, astype converts to 1/0

threshold

ConfusionMatrixDisplay(confusion_matrix(y_train, safe_pred)).plot()
plt.show()

test_probs = rf_search.predict_proba(X_test)[:,1]
safe_test_pred = (test_probs >= threshold).astype(int)

ConfusionMatrixDisplay(confusion_matrix(y_test, safe_test_pred)).plot()
plt.show()

recall_score(y_test,
             safe_test_pred)

accuracy_score(y_test,
               safe_test_pred)

joblib.dump(rf_search, 'loan_approval_model2.pkl')

loaded_model = joblib.load('loan_approval_model1.pkl')

predictions = loaded_model.predict(X_test)
print(predictions)

