import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from imblearn.under_sampling import NearMiss
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder
import joblib

# Cargar los datos de entrenamiento
connection = psycopg2.connect(
    user="postgres",
    password="daniela",
    host="localhost",
    port="5432",
    database="Fraude"
)

# Crear un cursor para ejecutar consultas
cursor = connection.cursor()

# Consulta SQL para seleccionar todos los registros de la tabla
query = "SELECT * FROM fraudes"
cursor.execute(query)

data=cursor.fetchall()
fraude_df=pd.read_sql(query, connection)

# Definir los límites de los intervalos de edad
bins = [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 80]

# Definir las etiquetas para los intervalos
labels = ['<20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '>60']

# Aplicar la división en intervalos de edad
fraude_df['ageint'] = pd.cut(fraude_df['age'], bins=bins, labels=labels)


columns_to_remove = ['monthh', 'weekofmonth', 'dayofweek',
       'dayofweekclaimed', 'monthclaimed', 'weekofmonthclaimed', 'policynumber', 'repnumber', 'yearr',
       'age','month_number']
df_filtered = fraude_df.drop(columns_to_remove, axis=1)

df_filtered['deductible']=fraude_df['deductible'].astype(object)
df_filtered['driverrating']=fraude_df['driverrating'].astype(object)
df_filtered['ageint']=fraude_df['ageint'].astype(object)

categorical_cols=df_filtered.columns[df_filtered.dtypes==object]
encoder=OneHotEncoder(sparse=False)
data_encoded = pd.DataFrame (encoder.fit_transform(df_filtered[categorical_cols]),index=df_filtered.index)
data_encoded.columns = encoder.get_feature_names_out(categorical_cols)
df_filtered= pd.concat([df_filtered.drop(categorical_cols ,axis=1), data_encoded ], axis=1)


# Dividir los datos en características (X) y variable objetivo (y)
X = df_filtered.drop('fraudfound_p', axis=1)
Y = df_filtered['fraudfound_p']


# Dividir los datos en conjuntos de entrenamiento y prueba de manera estratificada
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,train_size=0.8, stratify=Y, random_state=123)

column_names = X.columns.tolist()  # Guarda los nombres de las columnas en una lista

# Convertir los nombres de las columnas en índices numéricos
x_train.columns = range(x_train.shape[1])
x_test.columns = range(x_test.shape[1])

model = XGBClassifier(random_state=123, scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum())

params = {
 "learning_rate" : [0.05,0.10,0.15,0.20,0.25,0.30],
 "max_depth" : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma": [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
}

random_grid = RandomizedSearchCV(model,param_distributions=params,n_iter=5,scoring='recall',n_jobs=-1,cv=5,verbose=3,random_state=123)
random_grid.fit(X = x_train, y = y_train) 

best_model=random_grid.best_estimator_
joblib_model = "model.pkl"
joblib.dump(best_model, joblib_model)
modelo_final= joblib.load(joblib_model)

# Crear y entrenar el modelo
modelo_final = XGBClassifier(random_state=123)
modelo_final.fit(x_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = modelo_final.predict(x_test)

# Calcular las métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Imprimir las métricas de evaluación
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)

# Guardar el modelo entrenado
joblib.dump(model, './models/modelo_entrenado.pkl')