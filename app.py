import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.stats import multivariate_normal


# Set page configuration
st.set_page_config(
    page_title="Aplikasi Clustering Sederhana",
    page_icon="📊",
)

# Title and description
st.title("Aplikasi Clustering Sederhana")
st.markdown("Unggah dataset CSV, pilih metode clustering, dan lihat hasilnya.")

# File upload
uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

# Main content area
if uploaded_file is not None:
    # Load data
    try:
        data = pd.read_csv(uploaded_file)
        
        # Display data preview
        st.subheader("Preview Dataset")
        st.write(data.head())
        
        # Feature selection
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if not numeric_columns:
            st.error("Dataset tidak memiliki kolom numerik yang dapat digunakan untuk clustering.")
        else:

            Scaled = st.checkbox("Lakukan normalisasi")

            selected_features = st.multiselect(
                "Pilih kolom untuk clustering:",
                options=numeric_columns,
                default=numeric_columns[:min(2, len(numeric_columns))]
            )
            
            if selected_features:
                # Prepare features for clustering
                features = data[selected_features]
                
                # Handle missing values
                if features.isnull().any().any():
                    features = features.fillna(features.mean())
                
                if Scaled :
                    # Scale features
                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(features)
                else:
                    scaled_features = features
                
                # Clustering method selection
                col1, col2 = st.columns(2)
                
                with col1:
                    clustering_method = st.selectbox(
                        "Metode Clustering",
                        ["K-Means Optimaize W PSO", "SOM", "GMM"]
                    )
                
                with col2:
                    if clustering_method == "K-Means Optimaize W PSO":
                        X = data.to_numpy()
                        n_sample, n_fitur = X.shape
                        k = 4
                        n_partikel = 50
                        iterasi = 100
                        dimensi = k*n_fitur
                        np.random.seed(42)

                        posisi = np.random.randn(n_partikel, dimensi)
                        velocity = np.zeros((n_partikel, dimensi))
                        pbest = posisi.copy()
                        value_pbest = np.full(n_partikel, np.inf)

                        for f in range(n_partikel): #nilai fitness awal
                            centroid = posisi[f].reshape(k, n_fitur) #ini buat ngubah jadi 2D soalnya asalnya itu 1d
                            label = np.zeros(n_sample, dtype=int)

                            for e in range(n_sample): #jarak tiap data
                                jarak = np.sum((centroid - X[e])**2, axis=1) #nyari jarak tiap data ke centroid terdekat untuk masuk label mana
                                label[e] = np.argmin(jarak)
                            jarak_total = 0

                            for c in range(k): #hitung jarak tiap data ke centroidnya masing"
                                cluster = X[label == c]
                                if len(cluster) > 0:
                                    jarak_total += np.sum((cluster - centroid[c])**2) #buat ngukut bagus/ga pembagian clusternya

                            value_pbest[f] = jarak_total
                            pbest[f] = posisi[f].copy()

                        index_gbest = np.argmin(value_pbest)
                        gbest = posisi[index_gbest].copy()
                        value_gbest= value_pbest[index_gbest]

                    elif clustering_method == "SOM":
                        data_pca = data.to_numpy()

                        grid = (2, 2)
                        dimensi = 16
                        np.random.seed(0)
                        bobot = np.random.rand(grid[0], grid[1], dimensi)
                        alpha = 0.2
                        epochs = 150
                        radius = max(grid) / 2
                        bobot_lama = bobot.copy()
                        tol = 1e-4

                    elif clustering_method == "GMM":
                        X = data.to_numpy()
                        n, d = X.shape

                        K = 4  # jumlah komponen (cluster)
                        np.random.seed(42)

                        # Inisialisasi parameter
                        pi = np.ones(K) / K       # bobot campuran

                        mu = X[np.random.choice(n, K, replace=False)]  # mean awal

                        Sigma = np.array([np.eye(d)] * K)         # kovarian awal (identitas)
                        max_iter = 100
                        tol = 1e-4
                        log_likelihoods = []

                
                if st.button("Jalankan Clustering"):
                    with st.spinner("Melakukan clustering..."):
                        if clustering_method == "K-Means Optimaize W PSO":
                            note_cost = []

                            for i in range(iterasi):
                                for p in range(n_partikel):
                                    r1 = np.random.rand()
                                    r2 = np.random.rand()
                                    w = 0.7
                                    c1 = 2
                                    c2 = 2
                                    velocity[p] = (w * velocity[p] + c1 * r1 * (pbest[p] - posisi[p]) + c2 * r2 * (gbest - posisi[p]))
                                    posisi[p] += velocity[p]
                                    centroid = posisi[p].reshape(k, n_fitur) #direshape lagi soalnya setiap partikel itu bergerak terus jadi biar bisa menghitung ulagnposisi centroid yg baru dgn benar
                                    label = np.zeros(n_sample, dtype=int)
                                    for e in range(n_sample):
                                        jarak = np.sum((centroid - X[e])**2, axis=1)
                                        label[e] = np.argmin(jarak)
                                    jarak_total = 0
                                    for c in range(k):
                                        cluster = X[label == c]
                                    if len(cluster) > 0:
                                        jarak_total += np.sum((cluster - centroid[c])**2)

                                    cost = jarak_total
                                    if cost < value_pbest[p]:
                                        value_pbest[p] = cost
                                        pbest[p] = posisi[p].copy()
                                    if cost < value_gbest:
                                        value_gbest = cost
                                        gbest = posisi[p].copy()
                                note_cost.append(value_gbest)
                        elif clustering_method == "SOM":
                            for iter in range(epochs):
                                alpha = alpha * (1-(iter/epochs))
                                radius = radius * np.exp(-iter/epochs)
                                for d in data_pca:
                                    jarak_minimal = np.linalg.norm(np.array(d) - bobot[0, 0])
                                    indeks_best = (0, 0)
                                    for i in range(grid[0]):
                                        for j in range(grid[1]):
                                            jarak_neuron_terdekat = np.linalg.norm(np.array(d) - bobot[i, j])
                                            if jarak_neuron_terdekat < jarak_minimal:
                                                jarak_minimal = jarak_neuron_terdekat
                                                indeks_best = (i, j)
                                    for i in range(grid[0]):
                                        for j in range(grid[1]):
                                            jarak_bmu = np.linalg.norm(np.array([i, j]) - np.array(indeks_best))
                                            if jarak_bmu <= radius:
                                                bobot[i,j] += alpha * (np.array(d) - bobot[i,j])

                                perubahan = np.linalg.norm(bobot - bobot_lama)
                                if perubahan < tol:
                                    print(f'Berhenti pada iterasi ke-{iter} karena bobot sudah konvergen dengan perubahan {perubahan:.6f}')
                                    break
                        elif clustering_method == "GMM":

                            def e_step(X, pi, mu, Sigma):
                                """E-step: hitung responsibility gamma_ik"""
                                gamma = np.zeros((n, K))
                                for k in range(K):
                                    gamma[:, k] = pi[k] * multivariate_normal.pdf(X, mean=mu[k], cov=Sigma[k], allow_singular=True)
                                gamma /= np.sum(gamma, axis=1, keepdims=True)
                                return gamma

                            def m_step(X, gamma):
                                N_k = np.sum(gamma, axis=0)
                                pi = N_k / n
                                mu = np.dot(gamma.T, X) / N_k[:, np.newaxis]

                                Sigma = np.zeros((K, d, d))
                                epsilon = 1e-6  # nilai regularisasi

                                for k in range(K):
                                    X_centered = X - mu[k]
                                    gamma_k = gamma[:, k][:, np.newaxis]
                                    Sigma[k] = (gamma_k * X_centered).T @ X_centered / N_k[k]
                                    Sigma[k] += np.eye(d) * epsilon  # tambahkan regularisasi

                                return pi, mu, Sigma

                            for iteration in range(max_iter):
                                # E-step
                                gamma = e_step(X, pi, mu, Sigma)

                                # Log-likelihood
                                ll = np.sum(np.log(np.sum([
                                    pi[k] * multivariate_normal.pdf(X, mean=mu[k], cov=Sigma[k], allow_singular=True) for k in range(K)], axis=0)))
                                log_likelihoods.append(ll)

                                # M-step
                                pi, mu, Sigma = m_step(X, gamma)

                                # Cek konvergensi
                                if iteration > 0 and abs(log_likelihoods[-1] - log_likelihoods[-2]) < tol:
                                    break

                            print(f'Converged at iteration {iteration}')
                        
                        # Fit model and get labels
                        if clustering_method == "GMM":
                            cluster_labels = np.argmax(gamma, axis=1)
                        elif clustering_method == "K-Means Optimaize W PSO" :
                            best_centroid = gbest.reshape(k, n_fitur)
                            cluster_labels = np.zeros(n_sample, dtype=int)
                            for i in range(n_sample): # nge fix in masuk label mana
                                jarak = np.sum((best_centroid - X[i])**2, axis=1)
                                cluster_labels[i] = np.argmin(jarak)
                        elif clustering_method == "SOM":
                            cluster_labels = []
                            for x in data_pca:
                                jarak_minimal = np.linalg.norm(np.array(x) - bobot[0, 0])
                                indeks_dekat = (0, 0)
                                for i in range(grid[0]):
                                    for j in range(grid[1]):
                                        jarak_neuron_terdekat = np.linalg.norm(np.array(x) - bobot[i, j])
                                        if jarak_neuron_terdekat < jarak_minimal:
                                            jarak_minimal = jarak_neuron_terdekat
                                            indeks_dekat = (i, j)
                                cluster_id = indeks_dekat[0] * grid[1] + indeks_dekat[1]
                                cluster_labels.append(cluster_id)  
                        
                        # Add cluster labels to the original data
                        result_df = data.copy()
                        result_df['Cluster'] = cluster_labels
                        
                        # Display results
                        st.subheader("Hasil Clustering")
                        
                        # Display cluster distribution
                        st.write("Distribusi Cluster:")
                        cluster_counts = result_df['Cluster'].value_counts().sort_index()
                        
                        # Handle noise points in DBSCAN
                        if -1 in cluster_counts.index:
                            cluster_counts = cluster_counts.rename({-1: 'Noise'})
                        
                        fig = px.bar(
                            x=cluster_counts.index.astype(str),
                            y=cluster_counts.values,
                            labels={'x': 'Cluster', 'y': 'Jumlah Data'},
                        )
                        st.plotly_chart(fig)
                        
                        # Display data with cluster labels
                        st.write("Data dengan Label Cluster:")
                        st.dataframe(result_df)
                        
                        # Visualize clusters
                        st.subheader("Visualisasi Cluster")
                        
                        # PCA for dimensionality reduction if more than 2 features
                        if scaled_features.shape[1] > 2:
                            pca = PCA(n_components=2)
                            pca_result = pca.fit_transform(scaled_features)
                            viz_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
                            viz_df['Cluster'] = cluster_labels
                            
                            fig = px.scatter(
                                viz_df, x='PC1', y='PC2', color='Cluster',
                                title="Visualisasi Cluster", labels={"x" : "Feature 1", "y" : "Feature 2"}
                            )
                            st.plotly_chart(fig)
                            
                        # Direct visualization for 2D data
                        elif scaled_features.shape[1] == 2:
                            viz_df = pd.DataFrame(scaled_features, columns=selected_features)
                            viz_df['Cluster'] = cluster_labels
                            
                            fig = px.scatter(
                                viz_df, x=selected_features[0], y=selected_features[1], color='Cluster',
                                title=f"Visualisasi Cluster: {selected_features[0]} vs {selected_features[1]}"
                            )
                            st.plotly_chart(fig)
                        
                        # Calculate and display silhouette score if applicable
                        unique_labels = np.unique(cluster_labels)
                        if len(unique_labels) > 1 and not (len(unique_labels) == 2 and -1 in unique_labels):
                            # For DBSCAN, exclude noise points
                            if clustering_method == "DBSCAN" and -1 in unique_labels:
                                mask = cluster_labels != -1
                                if np.sum(mask) > 1:  # Need at least 2 points
                                    silhouette = silhouette_score(scaled_features[mask], cluster_labels[mask])
                                    st.write(f"Silhouette Score: {silhouette:.3f}")
                            else:
                                silhouette = silhouette_score(scaled_features, cluster_labels)
                                st.write(f"Silhouette Score: {silhouette:.3f}")
            else:
                st.warning("Silakan pilih minimal satu fitur untuk clustering.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
else:
    # Display instructions when no file is uploaded
    st.info("Silakan unggah file CSV untuk memulai analisis clustering.")
    
    # Example dataset section
    st.subheader("Contoh Dataset")
    st.markdown("""
    Anda dapat menggunakan dataset contoh berikut:
    
    1. [Iris Dataset](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv)
    2. [Mall Customer Segmentation Data](https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2025%20-%20Hierarchical%20Clustering/Mall_Customers.csv)
    """)
