import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import plotly.io as plt_io
import plotly.graph_objects as go
from sklearn.manifold import Isomap
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import OPTICS
from sklearn.neighbors import NearestNeighbors



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


def replace_nulls(df):
    df['meses_desde_ultima_multa'] = df['meses_desde_ultima_multa'].fillna(99999999)

    columns_to_replace = df.columns[df.columns != 'meses_desde_ultima_multa']
    df[columns_to_replace] = df[columns_to_replace].fillna(0)

    return df



def apply_standard_scaler(df):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    
    return scaled_df





def apply_umap(df, n_components):
    umap_model = umap.UMAP(n_components=n_components, random_state=42)
    umap_data = umap_model.fit_transform(df)
    
    plt.scatter(umap_data[:, 0], umap_data[:, 1], s=5)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("UMAP Visualization")
    plt.show()
    
    explained_variance = np.var(umap_data, axis=0)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    explained_variance_ratio_cumsum = np.cumsum(explained_variance_ratio)
    
    return umap_data, explained_variance_ratio_cumsum


def apply_pca(df, n_components):
    pca_model = PCA(n_components=n_components, random_state=42)
    pca_data = pca_model.fit_transform(df)
    
    plt.scatter(pca_data[:, 0], pca_data[:, 1], s=5)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("PCA Visualization")
    plt.show()
    
    explained_variance_ratio_cumsum = np.cumsum(pca_model.explained_variance_ratio_)
    
    return pca_data, explained_variance_ratio_cumsum


def apply_tsne(df, n_components, perplexity=30):
    tsne_model = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    tsne_data = tsne_model.fit_transform(df)
    
    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], s=5)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("t-SNE Visualization")
    plt.show()
    
    return tsne_data


def apply_lda(df, labels, n_components):
    lda_model = LinearDiscriminantAnalysis(n_components=n_components)
    lda_data = lda_model.fit_transform(df, labels)
    
    plt.scatter(lda_data[:, 0], lda_data[:, 1], c=labels, s=5, cmap='viridis')
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title("LDA Visualization")
    plt.show()
    
    explained_variance_ratio_cumsum = np.cumsum(lda_model.explained_variance_ratio_)
    
    return lda_data, explained_variance_ratio_cumsum



def plot_3d(component1,component2,component3):
    fig = go.Figure(data=[go.Scatter3d(
        x=component1,
        y=component2,
        z=component3,
        mode='markers',
        marker=dict(
            size=10, #color=y, 
            colorscale='Rainbow', 
            opacity=1,
            line_width=1
        )
    )])
    fig.update_layout(margin=dict(l=50,r=50,b=50,t=50),width=1800,height=1000)
    fig.layout.template = 'plotly_dark'
    
    fig.show()


def plot_3d_clusters(component1,component2,component3,colored):
    fig = go.Figure(data=[go.Scatter3d(
        x=component1,
        y=component2,
        z=component3,
        mode='markers',
        marker=dict(
            size=10, 
            color=colored, 
            colorscale='Rainbow', 
            opacity=1,
            line_width=1
        )
    )])
    fig.update_layout(margin=dict(l=50,r=50,b=50,t=50),width=1800,height=1000)
    fig.layout.template = 'plotly_dark'
    
    fig.show()


def find_optimal_clusters(data, max_clusters=20):
    wcss = []
    silhouette = []
    clusters_range = range(2, max_clusters+1)

    for n_clusters in clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        silhouette.append(silhouette_score(data, kmeans.labels_))
    
    plt.plot(clusters_range, wcss, 'bo-')
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.title("Elbow Plot")
    plt.show()

    # plt.plot(clusters_range, silhouette, 'bo-')
    # plt.xlabel("Number of Clusters")
    # plt.ylabel("Silhouette Score")
    # plt.title("Silhouette Plot")
    # plt.show()

    # plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=5)
    # plt.xlabel("Component 1")
    # plt.ylabel("Component 2")
    # plt.title("UMAP Components Colored by Cluster")
    # plt.show()



def perform_kmeans(df, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=42)
    kmeans.fit(df)
    labels = kmeans.labels_
    return labels


def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)

def perform_agglomerative_clustering_and_plot_dendrogram(df, n_clusters):
    model = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward', compute_distances=True)
    model = model.fit(df)
    labels = model.labels_

    plt.title('Dendrogram')
    plot_dendrogram(model, truncate_mode='level', p=4)
    plt.xlabel("Number of points in node (or index of point if no parenthesis)")
    plt.show()

    return labels



def perform_optics_clustering(df, k_neighbors=None, min_samples=None, max_eps=None):
    if k_neighbors is None:
        k_neighbors = 5

    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(df)
    distances, _ = nbrs.kneighbors(df)

    # Calcular el valor óptimo de eps
    
    sorted_distances = np.sort(distances[:, -1])
    plt.plot(sorted_distances)
    plt.xlabel('Points')
    plt.ylabel('k-distances')
    plt.title('k-distance Graph')
    plt.show()

    # Calcular el valor óptimo de min_samples
    if min_samples is None:
        min_samples = int(0.01 * len(df))

    # Aplicar OPTICS
    optics = OPTICS(min_samples=min_samples, max_eps=max_eps, cluster_method='xi', metric='minkowski', p=2)
    optics.fit(df)
    labels = optics.labels_
    return labels



def get_cluster_centroids(dataframe: pd.DataFrame, cluster_column: str = 'cluster', exclude_columns: list = None):
    if exclude_columns is None:
        exclude_columns = [cluster_column, "NOMBRE_ENTIDAD"]
    else:
        exclude_columns.extend([cluster_column, "NOMBRE_ENTIDAD"])

    centroid_columns = [col for col in dataframe.columns if col not in exclude_columns]

    centroids = dataframe.groupby(cluster_column)[centroid_columns].mean()

    return centroids







