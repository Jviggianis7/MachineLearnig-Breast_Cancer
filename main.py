import streamlit as st
import pickle as pickle
import pandas as pd
import numpy as np

def get_limpiar_datos():
  data = pd.read_csv('breast-cancer.csv')
  
  data = data.drop(['id'], axis=1)
  
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
  return data



def add_sidebar():
  st.sidebar.header("Mediciones de núcleos celulares")
  
  data = get_limpiar_datos()
  
  slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

  input_dict = {}

  for label, key in slider_labels:
    input_dict[key] = st.sidebar.slider(
      label,
      min_value=float(0),
      max_value=float(data[key].max()),
      value=float(data[key].mean())
    )
    
  return input_dict

def add_predicciones(input_data):
  model = pickle.load(open('model.sav', 'rb'))
  input_array = np.array(list(input_data.values())).reshape(1,-1)

  prediccion = model.predict(input_array)

  st.subheader("Predicción del conglomerado de células")
  st.write("el conglomerado de células es: ")

  if prediccion[0] == 0:
    st.write("Benigno")
    
  else:
    st.write("Maligno")
  
  st.write("La probabilidad de que el tumor sea Benigno (no canceroso) es: ", str(model.predict_proba(input_array)[0][0])) 
  st.write("La probabilidad de que el tumor sea Maligno (canceroso) es: ", str(model.predict_proba(input_array)[0][1]))  
    
  st.warning("ATENCIÓN: Este modelo de aprendizaje puede AYUDAR a los profesionales médicos a realizar un diagnóstico, pero NO debe utilizarse como SUSTITUTO de un diagnóstico profesional.")

def main():
  st.set_page_config(
    page_title="Predicción de cáncer de mama",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
  )
  
  
  input_data = add_sidebar()
  
  with st.container():
    st.title("Predicción de cáncer de mama: Benigno o Maligno")
    st.write("El cáncer de mama es el cáncer más común entre las mujeres en el mundo. Representa el 25% de todos los casos de cáncer y afectó a más de 2,1 millones de personas solo en 2015. Comienza cuando las células en el seno comienzan a crecer sin control. Estas células generalmente forman tumores que se pueden ver a través de rayos X o sentir como bultos en el área del seno.")
    st.write("Esta aplicación predice utilizando un modelo de aprendizaje automático si un tumor es benigno (no canceroso) o maligno (canceroso) en función de las mediciones que recibe de su laboratorio de citosis. También puede actualizar las medidas a mano usando los controles deslizantes en la barra lateral.")


  col1, col2 = st.columns([2,4])
  
  with col1:
    st.image('cancer-mama.jpg', caption="Síntomas del cáncer de mama")
  
  with col2:
    add_predicciones(input_data)


 
if __name__ == '__main__':
  main()