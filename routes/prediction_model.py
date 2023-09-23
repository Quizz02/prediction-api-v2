from fastapi import APIRouter
import joblib
from sklearn.calibration import LabelEncoder
from schemas.prediction import Prediction
import pandas as pd
import numpy as np

prediction = APIRouter()

posts = []

@prediction.get("/predictions")
def get_prediction():
    return posts

@prediction.get("/prediction/{id_}")
def get_prediction_id(id_ : int):
    for post in posts:
        if post["Id"] == id_:
            return post

    return "Data not found"

@prediction.post("/prediction")
def create_data(data_prediction: Prediction):
    modeloArbolReg = joblib.load('routes/DecisionTreeRegressor.pkl')

    datos_predecir = {
        'Tipo_animal': data_prediction.animalType,
        'Tamanio': data_prediction.size,
        'Color' : data_prediction.color,
        'Distrito' : data_prediction.district
    }
    
    tp = tam = color = distrito = 0

    datos_predecir['Tipo_animal'] = datos_predecir['Tipo_animal'].lower()
    datos_predecir['Tamanio'] = datos_predecir['Tamanio'].lower()
    datos_predecir['Color'] = datos_predecir['Color'].lower()
    datos_predecir['Distrito'] = datos_predecir['Distrito'].lower()

    if datos_predecir['Tipo_animal'] == 'gato':
        tp = 0
    elif datos_predecir['Tipo_animal'] == 'perro':
        tp = 1

    if datos_predecir['Tamanio'] == 'grande':
        tam = 0
    elif datos_predecir['Tamanio'] == 'mediano':
        tam = 1
    elif datos_predecir['Tamanio'] == 'pequeño':
        tam = 2

    if datos_predecir['Color'] == 'blanco':
        color = 0
    elif datos_predecir['Color'] == 'crema':
        color = 1
    elif datos_predecir['Color'] == 'gris':
        color = 2
    elif datos_predecir['Color'] == 'marrón':
        color = 3
    elif datos_predecir['Color'] == 'negro':
        color = 4

    if datos_predecir['Distrito'] == 'ancón':
        distrito = 0
    elif datos_predecir['Distrito'] == 'ate':
        distrito = 1
    elif datos_predecir['Distrito'] == 'barranco':
        distrito = 2
    elif datos_predecir['Distrito'] == 'breña':
        distrito = 3
    elif datos_predecir['Distrito'] == 'Carabayllo':
        distrito = 4
    elif datos_predecir['Distrito'] == 'cercado de lima':
        distrito = 5
    elif datos_predecir['Distrito'] == 'chaclacayo':
        distrito = 6
    elif datos_predecir['Distrito'] == 'chorrillos':
        distrito = 7
    elif datos_predecir['Distrito'] == 'cieneguilla':
        distrito = 8
    elif datos_predecir['Distrito'] == 'comas':
        distrito = 9
    elif datos_predecir['Distrito'] == 'el agustino':
        distrito = 10
    elif datos_predecir['Distrito'] == 'independencia':
        distrito = 11
    elif datos_predecir['Distrito'] == 'jesús maría':
        distrito = 12
    elif datos_predecir['Distrito'] == 'la molina':
        distrito = 13
    elif datos_predecir['Distrito'] == 'la victoria':
        distrito = 14
    elif datos_predecir['Distrito'] == 'lince':
        distrito = 15
    elif datos_predecir['Distrito'] == 'los olivos':
        distrito = 16
    elif datos_predecir['Distrito'] == 'lurigancho':
        distrito = 17
    elif datos_predecir['Distrito'] == 'lurín':
        distrito = 18
    elif datos_predecir['Distrito'] == 'magdalena del mar':
        distrito = 19
    elif datos_predecir['Distrito'] == 'miraflores':
        distrito = 20
    elif datos_predecir['Distrito'] == 'pachacámac':
        distrito = 21
    elif datos_predecir['Distrito'] == 'pucusana':
        distrito = 22
    elif datos_predecir['Distrito'] == 'pueblo libre':
        distrito = 23
    elif datos_predecir['Distrito'] == 'puente piedra':
        distrito = 24
    elif datos_predecir['Distrito'] == 'punta hermosa':
        distrito = 25
    elif datos_predecir['Distrito'] == 'punta negra':
        distrito = 26
    elif datos_predecir['Distrito'] == 'rímac':
        distrito = 27
    elif datos_predecir['Distrito'] == 'san bartolo':
        distrito = 28
    elif datos_predecir['Distrito'] == 'san borja':
        distrito = 29
    elif datos_predecir['Distrito'] == 'san isidro':
        distrito = 30
    elif datos_predecir['Distrito'] == 'san juan de lurigancho':
        distrito = 31
    elif datos_predecir['Distrito'] == 'san juan de miraflores':
        distrito = 32
    elif datos_predecir['Distrito'] == 'san luis':
        distrito = 33
    elif datos_predecir['Distrito'] == 'san martin de porres':
        distrito = 34
    elif datos_predecir['Distrito'] == 'san miguel':
        distrito = 35
    elif datos_predecir['Distrito'] == 'santa anita':
        distrito = 36
    elif datos_predecir['Distrito'] == 'santa maría del mar':
        distrito = 37
    elif datos_predecir['Distrito'] == 'santa rosa':
        distrito = 38
    elif datos_predecir['Distrito'] == 'santiago de surco':
        distrito = 39
    elif datos_predecir['Distrito'] == 'surquillo':
        distrito = 40
    elif datos_predecir['Distrito'] == 'villa maría del triunfo':
        distrito = 41
    elif datos_predecir['Distrito'] == 'villa el valvador':
        distrito = 42
    
    tp2 = tp
    distrito2 = distrito
    tam2 = tam
    color2 = color

    le = LabelEncoder()

    tp_animal = pd.Series(np.array([data_prediction.animalType]))
    le.fit(tp_animal)
    tp = le.transform(tp_animal)

    le = LabelEncoder() 

    tp_distrito = pd.Series(np.array([data_prediction.district]))
    le.fit(tp_distrito)
    distrito = le.transform(tp_distrito)

    le = LabelEncoder() 

    tp_Tamanio = pd.Series(np.array([data_prediction.size]))
    le.fit(tp_Tamanio)
    tam = le.transform(tp_Tamanio)

    le = LabelEncoder() 

    tpcolor = pd.Series(np.array([data_prediction.color]))
    le.fit(tpcolor)
    color = le.transform(tpcolor)

    tp[0] = tp2
    distrito[0] = distrito2
    tam[0] = tam2
    color[0] = color2
    
    data_prediction.p_color = np.array(color2)
    data_prediction.p_size = np.array(tam2)
    data_prediction.p_animal = np.array(tp2)
    data_prediction.p_distrito = np.array(distrito2)

    
    datos_a_predecir = {
        'Distrito' : distrito,
        'Tipo_animal': tp,
        'Tamanio': tam,
        'Color': color
        #'Distrito' : [data_prediction.p_distrito],
        #'Tipo_animal': [data_prediction.p_animal],
        #'Tamanio': [data_prediction.p_size],
        #'Color': [data_prediction.p_color]
    }

    datos_a_predecir = pd.DataFrame(datos_a_predecir)

    result1 = modeloArbolReg.predict_proba(datos_a_predecir)
    final_data = {
        'Id' : data_prediction.id,
        'Distrito' : data_prediction.district,
        'Tipo_animal': data_prediction.animalType,
        'Tamanio': data_prediction.size,
        'Color': data_prediction.color,  
        'Grave' : result1[0][0],
        'Herido' : result1[0][1], 
        'Saludable' : result1[0][2],
    }

    #print(datos_a_predecir)
    #print(result1)

    posts.append(final_data)
    return final_data