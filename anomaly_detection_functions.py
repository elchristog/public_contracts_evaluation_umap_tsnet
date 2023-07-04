import numpy as np
import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



def leer_csv(directorio, nombre_archivo, delimitador):
    ruta_archivo = f"{directorio}/{nombre_archivo}"

    datos = pd.read_csv(ruta_archivo, delimiter=delimitador)

    return datos


def read_partitioned_csv(path):
    all_files = []
    
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith(".csv"):
                file_path = os.path.join(dirname, filename)
                all_files.append(file_path)
    
    dfs = [pd.read_csv(file) for file in all_files]
    combined_df = pd.concat(dfs, ignore_index=True)
    
    return combined_df




def preprocess_data(data, numerical_columns, categorical_columns):
    scaler = StandardScaler()
    data_numerical = data[numerical_columns]
    data_numerical = pd.DataFrame(scaler.fit_transform(data_numerical), columns=data_numerical.columns, index=data_numerical.index)
    
    data_categorical = data[categorical_columns]
    data_categorical = pd.get_dummies(data_categorical, columns=categorical_columns)
    
    preprocessed_data = pd.concat([data_numerical, data_categorical], axis=1)
    
    return preprocessed_data



def detect_anomalies(data, cluster_col, features, contamination=0.05):
    unique_clusters = data[cluster_col].unique()
    data['anomaly_label'] = 0
    data['original_index'] = data.index
    data = data.reset_index(drop=True)
    data = data.dropna()

    scaler = MinMaxScaler(feature_range=(0, 1))

    for cluster in unique_clusters:
        cluster_data = data[data[cluster_col] == cluster]
        
        if cluster_data.shape[0] > 0:
            isolation_forest = IsolationForest(contamination=contamination)
            isolation_forest.fit(cluster_data[features])

            anomaly_scores = isolation_forest.decision_function(cluster_data[features])
            anomaly_labels = isolation_forest.predict(cluster_data[features])

            # Invierte y reescala las puntuaciones de anomalía
            rescaled_anomaly_scores = scaler.fit_transform(-anomaly_scores.reshape(-1, 1)).flatten()

            data.loc[cluster_data.index, 'anomaly_score'] = rescaled_anomaly_scores

    data = data.set_index('original_index')
    
    # Aplica PCA para visualizar todas las dimensiones
    pca = PCA(n_components=3)
    pca_data = pca.fit_transform(data[features])
    pca_df = pd.DataFrame(data=pca_data, columns=['PCA1', 'PCA2', 'PCA3'])
    pca_df.index = data.index

    # Crea un nuevo dataframe con los valores anómalos
    anomalies = pca_df[data['anomaly_label'] == 1]

    # Grafica los datos normales en azul y los anómalos en rojo
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pca_df['PCA1'], pca_df['PCA2'], c='b', label='Normal')
    ax.scatter(anomalies['PCA1'], anomalies['PCA2'], c='r', label='Anomalies')
    ax.legend()
    plt.show()

    return data



