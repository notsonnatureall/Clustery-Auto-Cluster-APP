# 🧠 Customer Segmentation using GMM, SOM, and K-Means PSO

This project focuses on customer segmentation using three unsupervised learning algorithms: **Gaussian Mixture Model (GMM)**, **Self-Organizing Maps (SOM)**, and **K-Means with Particle Swarm Optimization (PSO)**. The segmentation helps businesses understand different customer profiles and target them more effectively.

This project was developed as the final assignment for the **Machine Learning course** at Universitas Negeri Surabaya (UNESA).

---

## 📌 Project Objectives

- Analyze customer behavior patterns using clustering techniques.
- Identify specific needs and spending habits of different customer groups.
- Compare and evaluate clustering performance from three algorithms.

---

## ⚙️ Technologies and Methods

- **Python** with `numpy`, `pandas`, `matplotlib`, `sklearn`
- **Preprocessing**: Label encoding, missing value handling, feature engineering (e.g., age), and scaling
- **Dimensionality Reduction**: PCA (Principal Component Analysis) — retains 90% variance
- **Clustering Algorithms**:
  - Gaussian Mixture Model (GMM)
  - Self-Organizing Maps (SOM)
  - K-Means optimized with Particle Swarm Optimization (PSO)
- **Evaluation Metric**: Silhouette Score

---

## 📊 Model Performance Comparison

| Algorithm                        | Silhouette Score | Notes                                                                 |
| -------------------------------- | ---------------- | --------------------------------------------------------------------- |
| **Gaussian Mixture Model (GMM)** | 0.0604           | Poor separation; overlapping clusters; covariance regularization used |
| **Self-Organizing Maps (SOM)**   | **0.4053**       | Best performance; clear cluster boundaries; consistent interpretation |
| **K-Means + PSO**                | 0.3400           | Moderate performance; some overlap observed among clusters            |

---

## 🌐 Live Application

You can try the interactive customer segmentation app via the link below:

👉 [🧪 Launch the App on Streamlit](https://clustery-auto-cluster-app.streamlit.app/)

The application allows users to upload their dataset (CSV), automatically preprocess the data, and perform clustering using GMM, SOM, or K-Means PSO. It also provides visualizations and evaluation metrics such as Silhouette Score.
