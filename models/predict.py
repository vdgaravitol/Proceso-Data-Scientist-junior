import pandas as pd
import joblib
import psycopg2
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Cargar el modelo entrenado
model = joblib.load('./models/modelo_entrenado.pkl')

# Cargar los datos de entrada para la predicción

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


data = x_test

# Realizar la predicción
y_pred = model.predict(x_train)

# Obtener las probabilidades predichas
y_pred_prob = model.predict_proba(data)[:, 1]

# Imprimir el resultado de la predicción y las probabilidades
for i in range(len(data)):
    print('Caso:', data.iloc[i])
    print('Probabilidad predicha de fraude:', y_pred_prob[i])
    print('Fraude detectado:', 'Sí' if y_pred[i] == 1 else 'No')
    print()


