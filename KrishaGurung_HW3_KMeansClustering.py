# ============================================================
# K-Means Clustering on Iris Dataset
# ============================================================
# This file:
# 1. Loads a dataset (default: iris.xlsx)
# 2. Runs K-Means for k = 2–6
# 3. Saves silhouette scores and pairwise scatter plots
# 4. Runs the elbow method (k = 1–10)
# 5. Saves all results into organized output folders
# ============================================================

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------ #
# Settings / Configuration
# ------------------------------ #
INPUT_FILE = "iris.xlsx"        # dataset
SHEET_NAME = 0
OUTPUT_FILE = "kmeans_outputs"
KS = [2, 3, 4, 5, 6]            # k values for clustering
PAIR_INDX = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
RANDOM_STATE = 42

# Create main output folder if it doesn’t exist
os.makedirs(OUTPUT_FILE, exist_ok=True)

# ------------------------------ #
# 1) Load dataset
# ------------------------------ #
print("Loading data......", INPUT_FILE)
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

# Select the first 4 columns for clustering (Iris features)
X = df.iloc[:, :4]
features = list(df.columns[:4])
print("Features detected:", features)
print("Data Shape:", X.shape)

# ------------------------------ #
# 2) Run K-Means for each k in KS
# ------------------------------ #
sil_scores = {}

for k in KS:
    print(f"\nRunning KMeans with k = {k} ...")

    # Create subfolder for this k
    outdir_k = os.path.join(OUTPUT_FILE, f"k_{k}")
    os.makedirs(outdir_k, exist_ok=True)

    # Run KMeans
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X)

    # Compute silhouette score (if valid)
    try:
        s = silhouette_score(X, labels)
    except Exception:
        s = float('nan')
    sil_scores[k] = s
    print(f"Silhouette score (k={k}): {s:.4f}")

    # ---- Save 6 pairwise scatter plots ---- #
    for (i, j) in PAIR_INDX:
        plt.figure(figsize=(6, 4))
        plt.scatter(X.iloc[:, i], X.iloc[:, j], c=labels, cmap='tab10', s=40, edgecolor='k')
        plt.scatter(kmeans.cluster_centers_[:, i], kmeans.cluster_centers_[:, j],
                    marker='X', s=120, c='black', label='centroids')

        plt.xlabel(features[i])
        plt.ylabel(features[j])
        plt.title(f"k={k} | {features[i]} vs {features[j]}\nSilhouette={s:.4f}")
        plt.legend(loc='best', fontsize='small')

        # Save plot
        fname = os.path.join(outdir_k, f"k{k}_pair_{i}_{j}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()

    # Save cluster labels as CSV
    pd.DataFrame({'label': labels}).to_csv(
        os.path.join(outdir_k, f"k{k}_labels.csv"), index=False
    )

# ------------------------------ #
# 3) Save silhouette scores table
# ------------------------------ #
sil_df = pd.DataFrame(list(sil_scores.items()), columns=['k', 'silhouette_score']).sort_values('k')
sil_path = os.path.join(OUTPUT_FILE, "silhouette_scores.csv")
sil_df.to_csv(sil_path, index=False)

# Also save a simple bar plot for silhouette scores
plt.figure(figsize=(6, 4))
sns.barplot(x='k', y='silhouette_score', data=sil_df, palette='pastel')
plt.title("Silhouette Scores by Number of Clusters (k)")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FILE, "silhouette_scores.png"), dpi=150)
plt.close()
print("Saved silhouette score table and barplot.")

# ------------------------------ #
# 4) Elbow Method (k = 1..10)
# ------------------------------ #
print("\nRunning elbow method (k=1..10)...")
distortions = []
K_ELBOW = range(1, 11)

for k in K_ELBOW:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(X)

    # Average min distance to cluster centers
    D = np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)
    distortions.append(np.mean(D))

# Save elbow plot
elbow_path = os.path.join(OUTPUT_FILE, "elbow_plot.png")
plt.figure(figsize=(6, 4))
plt.plot(list(K_ELBOW), distortions, marker='o')
plt.xlabel('k')
plt.ylabel('Average distance to cluster center (distortion)')
plt.title('Elbow Method (k=1..10)')
plt.xticks(list(K_ELBOW))
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(elbow_path, dpi=150)
plt.close()
print("Saved elbow plot to", elbow_path)

# ------------------------------ #
# Done!
# ------------------------------ #
print("\nAll done! Outputs saved in folder:", OUTPUT_FILE)
print("Files organized by k in subfolders: k_2 ... k_6")
print("Example: kmeans_outputs/k_3/k3_pair_0_1.png")
print("Remember to take screenshots of the terminal outputs and selected plots for your report.")
