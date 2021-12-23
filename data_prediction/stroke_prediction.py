
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df_num = df.copy()
df = df.drop(['id'],axis=1)
X = df.drop(['stroke'], axis = 1)
y = df['stroke']
scaler = StandardScaler()
imp_knn = KNNImputer(n_neighbors=5)
imp_constant = SimpleImputer(strategy='constant')
ohe = OneHotEncoder(handle_unknown='ignore')
num_cols = ['age', 'bmi', 'avg_glucose_level']
cat_cols = [
    'gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'
]
preprocessor = make_column_transformer(
    (make_pipeline(scaler, imp_knn), num_cols),
    (make_pipeline(imp_constant, ohe), cat_cols),
    remainder='passthrough')
over = SMOTE(random_state=42)
logreg_pipeline = make_pipeline(
    preprocessor, over,
    LogisticRegression(C=0.01, penalty='l2', solver='lbfgs', max_iter=500))
logreg_pipeline.fit(X, y)
predictions = logreg_pipeline.predict(X)
# print(classification_report(y, predictions))
def stroke_prediction(patient_info):
    user = pd.DataFrame(patient_info)
    prediction = logreg_pipeline.predict_proba(user)
    print(40 * '=')
    print('Predicted probability of patient having a stroke:')
    print(str(100 * round(prediction[0][1], 2)) + '%')
    return prediction
# Patient1 = {
#
#     'gender': ['Male'],
#     'age': [32.0],
#     'hypertension': [0],
#     'heart_disease': [1],
#     'ever_married': ['Yes'],
#     'work_type': ['Private'],
#     'Residence_type': ['Urban'],
#     'avg_glucose_level': [100.00],
#     'bmi': [24.6],
#     'smoking_status': ['never smoked']
# }
Patient2 = {
    'gender': ['Female'],
    'age': [76.0],
    'hypertension': [1],
    'heart_disease': [0],
    'ever_married': ['Yes'],
    'work_type': ['Self-employed'],
    'Residence_type': ['Urban'],
    'avg_glucose_level': [78.00],
    'bmi': [27.0],
    'smoking_status': ['formerly smoked']
}

stroke_prediction(Patient2)
# stroke_prediction(Patient1)
