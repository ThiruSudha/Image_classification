
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from imblearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
# df_num = df.copy()
df = df.drop(['id'],axis=1)
df.dropna(subset = ["bmi"], inplace=True)
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
# logreg_pipeline = make_pipeline(
#     preprocessor, over,
#     DecisionTreeClassifier())
#
# logreg_pipeline = make_pipeline(
#     preprocessor, over,
#     KNeighborsClassifier())
logreg_pipeline = make_pipeline(
    preprocessor, over,
    RandomForestClassifier())
# logreg_pipeline = make_pipeline(
#     preprocessor, over,
#     LogisticRegression())
logreg_pipeline.fit(X, y)
predictions = logreg_pipeline.predict(X)
print(predictions)
for x in predictions:
    print(x)
# class_names = ['Stroke NO','Stroke']
# def stroke_prediction(patient_info):
#     # print(type(patient_info))
#     user = pd.DataFrame(patient_info)
#     # print(type(user))
#     prediction = logreg_pipeline.predict_proba(user)
#     print(prediction[0])
#     result = class_names[np.argmax(prediction[0])]
#     print(result)

    # print(str(100 * round(prediction[0][1], 2)) + '%')
    # return prediction



# Patient2 = {
#     'gender': ['Female'],
#     'age': [7],
#     'hypertension': [0],
#     'heart_disease': [0],
#     'ever_married': ['No'],
#     'work_type': ['children'],
#     'Residence_type': ['Urban'],
#     'avg_glucose_level': [98],
#     'bmi': [34],
#     'smoking_status': ['Unknown']
# }
# 24977,Female,72,1,0,Yes,Private,Rural,74.63,23.1,formerly smoked,1
# 21025,Female,7,0,0,No,children,Urban,98.22,34,Unknown,0

# stroke_prediction(Patient2)
