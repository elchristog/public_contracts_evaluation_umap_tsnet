import urllib.request
import gender_guesser.detector as gender
import requests
import xmltodict
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandasql import sqldf
import plotly.express as px
import plotly.graph_objects as go
from pandas import json_normalize
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_lottie import st_lottie
from pandasql import sqldf



@st.cache_data
def load_data():
    data = pd.read_csv("anomaly_data/historico_mensual.csv")
    data = data.drop('Unnamed: 0',axis=1)
    return data

@st.cache_data
def get_data(mes):
    query = "SELECT * FROM anomaly_data WHERE mes= '"+mes+"';"
    resultado = pysqldf(query)    
    return resultado

@st.cache_data
def get_evolucion_transparencia():
    query = """SELECT mes,
                sum(case when bandera_de_anomaly_score='Verde' then 1 end)as Verde,
                sum(case when bandera_de_anomaly_score='Amarillo' then 1 end)as Amarillo,
                sum(case when bandera_de_anomaly_score='Naranja' then 1 end)as Naranja,
                sum(case when bandera_de_anomaly_score='Rojo' then 1 end)as Rojo
                from datos group by mes;"""
    resultado = pysqldf(query)    
    return resultado

@st.cache_data
def get_valor_evolucion_transparencia():
    query = """SELECT mes,
                sum(case when bandera_de_anomaly_score='Verde' then valor_contrato end)as Verde,
                sum(case when bandera_de_anomaly_score='Amarillo' then valor_contrato end)as Amarillo,
                sum(case when bandera_de_anomaly_score='Naranja' then valor_contrato end)as Naranja,
                sum(case when bandera_de_anomaly_score='Rojo' then valor_contrato end)as Rojo
                from datos group by mes;"""
    resultado = pysqldf(query)    
    return resultado

def get_valor_clase_evolucion_transparencia(mes):
    query = "SELECT mes, clase, sum(valor_contrato)as Valor from datos where mes='"+mes+"'  group by mes, clase;"
    resultado = pysqldf(query)    
    return resultado

@st.cache_data
def get_valor_acumulado_clase_mes(mes,clase):
    query ="""SELECT
            mes,
            clase,
            sum(case when bandera_de_anomaly_score='Verde' then valor_contrato end)as Verde,
            sum(case when bandera_de_anomaly_score='Amarillo' then valor_contrato end)as Amarillo,
            sum(case when bandera_de_anomaly_score='Naranja' then valor_contrato end)as Naranja,
            sum(case when bandera_de_anomaly_score='Rojo' then valor_contrato end)as Rojo
            from datos where mes='"""+mes+"""'  and clase= '"""+clase+"""' group by mes,clase;"""            
    resultado = pysqldf(query)    
    return resultado

@st.cache_data
def get_valor_acumulado_clase_mes_2(mes):
    query ="""SELECT
            mes,
            clase,
            sum(case when bandera_de_anomaly_score='Verde' then valor_contrato end)as Verde,
            sum(case when bandera_de_anomaly_score='Amarillo' then valor_contrato end)as Amarillo,
            sum(case when bandera_de_anomaly_score='Naranja' then valor_contrato end)as Naranja,
            sum(case when bandera_de_anomaly_score='Rojo' then valor_contrato end)as Rojo
            from datos where mes='"""+mes+"""'group by mes,clase;"""            
    resultado = pysqldf(query)    
    return resultado

pysqldf = lambda q: sqldf(q, globals())


st.title("Tablero de Transparencia en la Contratación Pública")
st.divider()

#Carga de Datasets iniciales
datos = load_data()
anomaly_data=load_data()

#Filtro Inicial
meses = pd.DataFrame({'mes': anomaly_data.mes.unique()})
clases = pd.DataFrame({'clase': anomaly_data.clase.unique()})

#Carga de datasets secundarios
data=load_data()
evolucio_transparencia = get_evolucion_transparencia()
valor_evolucio_transparencia = get_valor_evolucion_transparencia()

#Subtitulo de las gráficas principales
st.subheader("Evolución de Anomaly Score")

#Impresión de las gráficas principales
col1, col2  = st.columns(2)
with col1:
    fig = px.bar(evolucio_transparencia, x="mes", y=["Verde", "Amarillo", "Naranja", "Rojo"], title='Contratos')
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

with col2:
    fig = px.bar(valor_evolucio_transparencia, x="mes", y=["Verde", "Amarillo", "Naranja", "Rojo"], title='Valor en Riesgo por Categoría')
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

st.divider()

#Metricas Generales
mes = st.selectbox('Selecciona el mes, para ver el reporte', meses['mes'])
anomaly_data =get_data(str(mes))
total_empresas_anomalas=len(datos[datos['bandera_de_anomaly_score']=='Rojo'])
total_empresas_anomalas_mes=len(anomaly_data[anomaly_data['bandera_de_anomaly_score']=='Rojo'])

#Impresión de las Métricas
col1, col2= st.columns(2)
col1.metric("Total empresas con anomalias 2021",total_empresas_anomalas)
col2.metric("Empresas con Anomalias para el mes de: "+mes, total_empresas_anomalas_mes)
st.divider()

#fig = px.histogram(anomaly_data,  x="anomaly_score", marginal="rug", hover_data="anomaly_score",nbins=40, title='Distribución #de Contratos Según su Score por Mes')
#st.plotly_chart(fig, theme="streamlit", use_container_width=True)

clases_valor_evolucio_transparencia = get_valor_clase_evolucion_transparencia(mes)
fig2 = px.pie(clases_valor_evolucio_transparencia,  values="Valor", names='clase' ,title='Distribución de Valor de Contrato por Clase y Mes')
st.plotly_chart(fig2, theme="streamlit", use_container_width=True)

valor_acumulado_clase_mes_2 = get_valor_acumulado_clase_mes_2(mes)
fig4 = px.bar(valor_acumulado_clase_mes_2, x="clase", y=["Verde", "Amarillo", "Naranja", "Rojo"], title='Contratos en riesgo por clase y mes')
st.plotly_chart(fig4, theme="streamlit", use_container_width=True)

#Filtro de clases
clase = st.selectbox('Selecciona la clase, para actualizar la gráfica', clases['clase'])
valor_acumulado_clase_mes = get_valor_acumulado_clase_mes(mes,clase)
fig3 = px.bar(valor_acumulado_clase_mes, x="clase", y=["Verde", "Amarillo", "Naranja", "Rojo"], title='Detalle de contratos en riesgo por clase y mes')
st.plotly_chart(fig3, theme="streamlit", use_container_width=True)
