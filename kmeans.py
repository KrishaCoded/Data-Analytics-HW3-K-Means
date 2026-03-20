import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns

# settings
INPUT_FILE = "iris.xlsx"        # dataset
SHEET_NAME = 0
OUTPUT_FILE = "kmeans_outputs"
KS = [2, 3, 4, 5, 6]
PAIR_INDX = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
RANDOM_STATE = 42
# ------------------------------

os.makedirs(OUTPUT_FILE, exist_ok=True)

# (1) Load dataset
print ("Loading data......", INPUT_FILE)
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
X = df.iloc[:, :4]
features = list(df.columns[:4])

print("Features detected:", features)
print("Data Shape:", X.shape)

# (2) run kMeans for ks & get silhouette
sil_scores = {}
for k in KS:
    print(f"\nRunning KMeans with k = {k} ...")
    outdir_k = os.path.join(OUTPUT_FILE, f"k_{k}")
    os.makedirs(outdir_k, exist_ok=True)

    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X)
    # Only compute silhouette if k > 1 and min cluster size > 1
    try:
        s = silhouette_score(X, labels)
    except Exception as e:
        s = float('nan')
    sil_scores[k] = s
    print(f"Silhouette score (k={k}): {s:.4f}")

# Save 6 pairwise scatter plots
    for idx, (i, j) in enumerate(PAIR_INDX, start=1):
        plt.figure(figsize=(6,4))
        plt.scatter(X[:, i], X[:, j], c=labels, cmap='tab10', s=40, edgecolor='k')
        plt.scatter(kmeans.cluster_centers_[:, i], kmeans.cluster_centers_[:, j],
                    marker='X', s=120, c='black', label='centroids')
        plt.xlabel(features[i])
        plt.ylabel(features[j])
        plt.title(f"k={k}  |  {features[i]} vs {features[j]}\nSilhouette={s:.4f}")
        plt.legend(loc='best', fontsize='small')
        fname = os.path.join(outdir_k, f"k{k}_pair_{i}_{j}.png")
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()

    # Also save labels for possible screenshot table
    pd.DataFrame({'label': labels}).to_csv(os.path.join(outdir_k, f"k{k}_labels.csv"), index=False)


# 3) Save silhouette score table (csv + image)
sil_df = pd.DataFrame(list(sil_scores.items()), columns=['k', 'silhouette_score']).sort_vals('k)')

# 4) Elbow method: distortions for k=1..10
print("\nRunning elbow method (k=1..10)...")
distortions = []
K_ELBOW = range(1, 11)
for k in K_ELBOW:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(X)

    # compute average min distance to cluster center
    D = np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)
    distortions.append(np.mean(D))
elbow_path = os.path.join(OUTPUT_FILE, "elbow_plot.png")
plt.figure(figsize=(6,4))
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

print("\nAll done. Outputs saved in folder:", OUTPUT_FILE)
print("Files organized by k in subfolders k_2 ... k_6. Example: kmeans_outputs/k_3/k3_pair_0_1.png")
print("Remember to take screenshots of the terminal outputs and selected plots for your run examples.")