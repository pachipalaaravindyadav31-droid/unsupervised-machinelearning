import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from IPython.display import display
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering
)

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)



RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_PATH = r"C:\Users\aravi\Downloads\placement_predict_50k Dataset.csv"

# Silhouette score on all 50,000 rows is slow.
# A representative sample makes the calculation faster.
SILHOUETTE_SAMPLE_SIZE = 5000

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 100)




df = pd.read_csv(DATA_PATH)

print("Dataset path:", DATA_PATH)
print("Shape:", df.shape)

display(df.head())

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

df.info()


# ============================================================
# 3. DATASET INFORMATION
# ============================================================

numeric_cols_all = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

print("\nNumerical columns:")
print(numeric_cols_all)

print("\nCategorical columns:")
print(categorical_cols)

print("\nMissing values:")
display(
    df.isna()
    .sum()
    .sort_values(ascending=False)
    .to_frame("missing_count")
    .head(20)
)


# ============================================================
# 4. SELECT MODELING FEATURES
# ============================================================

FEATURES = [
    "SGPA_Sem1", "SGPA_Sem2", "SGPA_Sem3", "SGPA_Sem4",
    "SGPA_Sem5", "SGPA_Sem6", "SGPA_Sem7", "SGPA_Sem8",
    "CGPA", "AttendancePercent",
    "Internships", "Projects", "Workshops", "Certifications",
    "Publications", "AptitudeTestScore", "SoftSkillsRating",
    "CodingTestScore", "MockInterviewScore", "ExtraCurricular"
]

FEATURES = [column for column in FEATURES if column in df.columns]

print("\nNumber of modeling features:", len(FEATURES))
print(FEATURES)

X_raw = df[FEATURES].copy()

display(X_raw.describe().T)



preprocess = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_scaled_array = preprocess.fit_transform(X_raw)

X_scaled = pd.DataFrame(
    X_scaled_array,
    columns=FEATURES,
    index=df.index
)

print("Processed matrix shape:", X_scaled.shape)
display(X_scaled.head())

print("\nMeans after standardisation:")
display(X_scaled.mean().round(3).to_frame("mean"))

print("\nStandard deviations after standardisation:")
display(X_scaled.std(ddof=0).round(3).to_frame("std"))



def fast_silhouette_score(X, labels, sample_size=SILHOUETTE_SAMPLE_SIZE):
    """
    Calculates silhouette score using a fixed random sample.
    This avoids the very slow full-data calculation.
    """

    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return np.nan

    sample_size = min(sample_size, len(X))

    return silhouette_score(
        X,
        labels,
        sample_size=sample_size,
        random_state=RANDOM_STATE
    )




pca_vis = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca2 = pca_vis.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(
    X_pca2[:, 0],
    X_pca2[:, 1],
    s=8,
    alpha=0.35
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Student Data Projected onto First Two Principal Components")
plt.show()

print(
    "Variance explained by PC1 + PC2:",
    round(pca_vis.explained_variance_ratio_.sum() * 100, 2),
    "%"
)



K_VALUES = list(range(2, 9))
results_kmeans = []

for k in K_VALUES:

    print(f"\nProcessing K = {k}")

    km_random = KMeans(
        n_clusters=k,
        init="random",
        n_init=10,
        random_state=RANDOM_STATE
    )

    labels_random = km_random.fit_predict(X_scaled)

    km_plus = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=RANDOM_STATE
    )

    labels_plus = km_plus.fit_predict(X_scaled)

    results_kmeans.append({
        "K": k,
        "Random_Inertia": km_random.inertia_,
        "Random_Silhouette": fast_silhouette_score(
            X_scaled,
            labels_random
        ),
        "KMeans++_Inertia": km_plus.inertia_,
        "KMeans++_Silhouette": fast_silhouette_score(
            X_scaled,
            labels_plus
        )
    })

kmeans_results = pd.DataFrame(results_kmeans)

print("\nK-Means results:")
display(kmeans_results.round(4))




fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    kmeans_results["K"],
    kmeans_results["Random_Inertia"],
    marker="o",
    label="Random init"
)

ax.plot(
    kmeans_results["K"],
    kmeans_results["KMeans++_Inertia"],
    marker="o",
    label="K-Means++"
)

ax.set_xlabel("Number of clusters (K)")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Method: Inertia vs K")
ax.legend()

plt.show()



fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    kmeans_results["K"],
    kmeans_results["Random_Silhouette"],
    marker="o",
    label="Random init"
)

ax.plot(
    kmeans_results["K"],
    kmeans_results["KMeans++_Silhouette"],
    marker="o",
    label="K-Means++"
)

ax.set_xlabel("Number of clusters (K)")
ax.set_ylabel("Silhouette score")
ax.set_title("Silhouette Score vs K")
ax.legend()

plt.show()




best_k_silhouette = int(
    kmeans_results.loc[
        kmeans_results["KMeans++_Silhouette"].idxmax(),
        "K"
    ]
)

print(
    "\nK with highest K-Means++ silhouette:",
    best_k_silhouette
)

K_BEST = best_k_silhouette

km_random_final = KMeans(
    n_clusters=K_BEST,
    init="random",
    n_init=10,
    random_state=RANDOM_STATE
)

labels_random_final = km_random_final.fit_predict(X_scaled)

km_plus_final = KMeans(
    n_clusters=K_BEST,
    init="k-means++",
    n_init=10,
    random_state=RANDOM_STATE
)

labels_plus_final = km_plus_final.fit_predict(X_scaled)



comparison = pd.DataFrame({
    "Model": ["K-Means random", "K-Means++"],
    "K": [K_BEST, K_BEST],
    "Inertia": [
        km_random_final.inertia_,
        km_plus_final.inertia_
    ],
    "Silhouette": [
        fast_silhouette_score(X_scaled, labels_random_final),
        fast_silhouette_score(X_scaled, labels_plus_final)
    ],
    "Davies_Bouldin": [
        davies_bouldin_score(X_scaled, labels_random_final),
        davies_bouldin_score(X_scaled, labels_plus_final)
    ]
})

print("\nFinal K-Means comparison:")
display(comparison.round(4))


# ============================================================
# 13. K-MEANS PCA GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

plt.scatter(
    X_pca2[:, 0],
    X_pca2[:, 1],
    c=labels_plus_final,
    s=10,
    alpha=0.45,
    cmap="tab10"
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title(
    f"K-Means++ Clusters (K={K_BEST}) shown in PCA 2-D space"
)
plt.colorbar(label="Cluster")

plt.show()


# ============================================================
# 14. CLUSTER PROFILE
# ============================================================

profile_df = df[FEATURES].copy()
profile_df["Cluster"] = labels_plus_final

cluster_profile = (
    profile_df
    .groupby("Cluster")[FEATURES]
    .mean()
    .round(2)
)

cluster_sizes = (
    pd.Series(labels_plus_final)
    .value_counts()
    .sort_index()
    .rename("Student_Count")
)

print("\nCluster profile:")
display(pd.concat([cluster_sizes, cluster_profile], axis=1))




heatmap_data = (
    cluster_profile - cluster_profile.mean()
) / cluster_profile.std(ddof=0).replace(0, 1)

plt.figure(figsize=(14, 7))

sns.heatmap(
    heatmap_data.T,
    center=0,
    cmap="coolwarm"
)

plt.xlabel("Cluster")
plt.ylabel("Feature")
plt.title("Cluster Profiles — Relative Strength by Feature")

plt.show()


# ============================================================
# 16. DBSCAN SAMPLE
# ============================================================

DBSCAN_SAMPLE_SIZE = min(10000, len(X_scaled))

dbscan_rng = np.random.default_rng(RANDOM_STATE)

dbscan_idx = dbscan_rng.choice(
    len(X_scaled),
    size=DBSCAN_SAMPLE_SIZE,
    replace=False
)

X_db_sample = X_scaled.iloc[dbscan_idx].to_numpy()

print(
    "\nDBSCAN parameter-search sample:",
    X_db_sample.shape
)


# ============================================================
# 17. DBSCAN PARAMETER SEARCH
# ============================================================

EPS_VALUES = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
MIN_SAMPLES_VALUES = [5, 10, 20]

dbscan_results = []

for eps in EPS_VALUES:

    for min_samples in MIN_SAMPLES_VALUES:

        print(
            f"DBSCAN: eps={eps}, min_samples={min_samples}"
        )

        db = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            n_jobs=-1
        )

        labels = db.fit_predict(X_db_sample)

        n_clusters = len(set(labels)) - (
            1 if -1 in labels else 0
        )

        noise_pct = (labels == -1).mean() * 100

        non_noise = labels != -1

        if n_clusters >= 2 and non_noise.sum() > 1:

            sil = silhouette_score(
                X_db_sample[non_noise],
                labels[non_noise],
                sample_size=min(
                    SILHOUETTE_SAMPLE_SIZE,
                    non_noise.sum()
                ),
                random_state=RANDOM_STATE
            )

            dbi = davies_bouldin_score(
                X_db_sample[non_noise],
                labels[non_noise]
            )

        else:

            sil = np.nan
            dbi = np.nan

        dbscan_results.append({
            "eps": eps,
            "min_samples": min_samples,
            "clusters": n_clusters,
            "noise_pct": noise_pct,
            "silhouette_non_noise": sil,
            "davies_bouldin_non_noise": dbi
        })

dbscan_results = pd.DataFrame(dbscan_results)

print("\nDBSCAN results:")

display(
    dbscan_results
    .sort_values(
        ["silhouette_non_noise", "noise_pct"],
        ascending=[False, True]
    )
    .round(4)
)


# ============================================================
# 18. DBSCAN NOISE GRAPH
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

for m in MIN_SAMPLES_VALUES:

    sub = dbscan_results[
        dbscan_results["min_samples"] == m
    ]

    ax.plot(
        sub["eps"],
        sub["noise_pct"],
        marker="o",
        label=f"min_samples={m}"
    )

ax.set_xlabel("eps")
ax.set_ylabel("Noise percentage")
ax.set_title("DBSCAN Sensitivity — Noise Percentage")
ax.legend()

plt.show()


# ============================================================
# 19. SELECT DBSCAN PARAMETERS
# ============================================================

valid_db = dbscan_results[
    dbscan_results["silhouette_non_noise"].notna() &
    (dbscan_results["noise_pct"] < 50)
].copy()

if len(valid_db):

    best_db_row = valid_db.sort_values(
        ["silhouette_non_noise", "noise_pct"],
        ascending=[False, True]
    ).iloc[0]

    BEST_EPS = float(best_db_row["eps"])
    BEST_MIN_SAMPLES = int(best_db_row["min_samples"])

else:

    BEST_EPS = 1.2
    BEST_MIN_SAMPLES = 10

print("\nCandidate DBSCAN parameters:")
print("eps =", BEST_EPS)
print("min_samples =", BEST_MIN_SAMPLES)


# ============================================================
# 20. FINAL DBSCAN GRAPH
# ============================================================

db_final_sample = DBSCAN(
    eps=BEST_EPS,
    min_samples=BEST_MIN_SAMPLES,
    n_jobs=-1
).fit_predict(X_db_sample)

db_pca2 = PCA(
    n_components=2,
    random_state=RANDOM_STATE
).fit_transform(X_db_sample)

plt.figure(figsize=(9, 6))

plt.scatter(
    db_pca2[:, 0],
    db_pca2[:, 1],
    c=db_final_sample,
    s=10,
    alpha=0.5,
    cmap="tab10"
)

plt.xlabel("PCA PC1")
plt.ylabel("PCA PC2")

plt.title(
    f"DBSCAN on Exploration Sample: "
    f"eps={BEST_EPS}, min_samples={BEST_MIN_SAMPLES}"
)

plt.colorbar(label="Cluster / Noise")

plt.show()


# ============================================================
# 21. DBSCAN FINAL OUTPUT
# ============================================================

print(
    "\nClusters:",
    len(set(db_final_sample)) -
    (1 if -1 in db_final_sample else 0)
)

print(
    "Noise points:",
    int((db_final_sample == -1).sum())
)

print(
    "Noise percentage:",
    round(
        (db_final_sample == -1).mean() * 100,
        2
    ),
    "%"
)

print("\nExecution completed successfully.")



HIER_SAMPLE_SIZE = min(2000, len(X_scaled))
hier_idx = dbscan_rng.choice(len(X_scaled), size=HIER_SAMPLE_SIZE, replace=False)
X_hier_sample = X_scaled.iloc[hier_idx].to_numpy()

print("Hierarchical sample:", X_hier_sample.shape)

# Compare the four common linkage strategies using the same sample.
linkage_methods = ["single", "complete", "average", "ward"]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

Z_by_linkage = {}

for ax, method in zip(axes, linkage_methods):
    Z = linkage(X_hier_sample, method=method)
    Z_by_linkage[method] = Z

    dendrogram(
        Z,
        truncate_mode="lastp",
        p=30,
        no_labels=True,
        ax=ax
    )
    ax.set_title(f"Dendrogram — {method.title()} linkage")
    ax.set_xlabel("Merged groups")
    ax.set_ylabel("Distance")

plt.tight_layout()
plt.show()

CUT_K_VALUES = [2, 3, 4, 5, 6]

hierarchical_results = []

for method in linkage_methods:
    for k in CUT_K_VALUES:
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage=method
        )
        labels = model.fit_predict(X_hier_sample)

        hierarchical_results.append({
            "linkage": method,
            "K": k,
            "silhouette": silhouette_score(X_hier_sample, labels),
            "davies_bouldin": davies_bouldin_score(X_hier_sample, labels),
            "calinski_harabasz": calinski_harabasz_score(X_hier_sample, labels)
        })

hierarchical_results = pd.DataFrame(hierarchical_results)
display(hierarchical_results.round(4))


fig, ax = plt.subplots(figsize=(10, 6))

for method in linkage_methods:
    sub = hierarchical_results[hierarchical_results["linkage"] == method]
    ax.plot(sub["K"], sub["silhouette"], marker="o", label=method.title())

ax.set_xlabel("Number of clusters / cut level")
ax.set_ylabel("Silhouette score")
ax.set_title("Hierarchical Clustering — Silhouette vs Cut Level")
ax.legend()
plt.show()

# Visualize one linkage (Ward) at multiple cut levels in PCA space.
pca_hier = PCA(n_components=2, random_state=RANDOM_STATE)
X_hier_pca = pca_hier.fit_transform(X_hier_sample)

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
axes = axes.ravel()

for ax, k in zip(axes, CUT_K_VALUES):
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X_hier_sample)

    ax.scatter(X_hier_pca[:, 0], X_hier_pca[:, 1],
               c=labels, s=8, alpha=0.45, cmap="tab10")
    ax.set_title(f"Ward linkage — K={k}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

# Hide unused axis
if len(axes) > len(CUT_K_VALUES):
    axes[-1].axis("off")

plt.tight_layout()
plt.show()

# Visualize one linkage (Ward) at multiple cut levels in PCA space.
pca_hier = PCA(n_components=2, random_state=RANDOM_STATE)
X_hier_pca = pca_hier.fit_transform(X_hier_sample)

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
axes = axes.ravel()

for ax, k in zip(axes, CUT_K_VALUES):
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = model.fit_predict(X_hier_sample)

    ax.scatter(X_hier_pca[:, 0], X_hier_pca[:, 1],
               c=labels, s=8, alpha=0.45, cmap="tab10")
    ax.set_title(f"Ward linkage — K={k}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

# Hide unused axis
if len(axes) > len(CUT_K_VALUES):
    axes[-1].axis("off")

plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(9, 5))
linkage_compare = pd.DataFrame(hierarchical_results)
ax.set_xlabel("Linkage")
ax.set_ylabel("Silhouette score")
COMPARE_K = 3
ax.set_title(f"Effect of Linkage at K={COMPARE_K}")
plt.show()

pca_full = PCA(random_state=RANDOM_STATE)
X_pca_full = pca_full.fit_transform(X_scaled)

explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

pca_summary = pd.DataFrame({
    "PC": np.arange(1, len(explained) + 1),
    "Explained_Variance_Ratio": explained,
    "Cumulative_Explained_Variance": cumulative
})

display(pca_summary.head(15).round(4))

plt.figure(figsize=(10, 5))
plt.plot(
    np.arange(1, len(explained) + 1),
    cumulative * 100,
    marker="o"
)
plt.axhline(90, linestyle="--", linewidth=1, label="90%")
plt.axhline(95, linestyle="--", linewidth=1, label="95%")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative explained variance (%)")
plt.title("PCA — Cumulative Explained Variance")
plt.legend()
plt.show()

n_90 = np.argmax(cumulative >= 0.90) + 1
n_95 = np.argmax(cumulative >= 0.95) + 1

print("Components needed for at least 90%:", n_90)
print("Components needed for at least 95%:", n_95)

# PCA 2-D visualization
plt.figure(figsize=(9, 6))
plt.scatter(
    X_pca_full[:, 0],
    X_pca_full[:, 1],
    s=8,
    alpha=0.35
)
plt.xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
plt.title("PlacementPredict Students in PCA 2-D Space")
plt.show()

n_components_to_show = min(5, X_pca_full.shape[1])

loadings = pd.DataFrame(
    pca_full.components_[:n_components_to_show].T,
    index=FEATURES,
    columns=[f"PC{i}" for i in range(1, n_components_to_show + 1)]
)

display(loadings.round(3))

plt.figure(figsize=(12, 8))
sns.heatmap(loadings, center=0, cmap="coolwarm")
plt.title("PCA Component Loadings")
plt.xlabel("Principal Component")
plt.ylabel("Original Feature")
plt.show()

# Median-imputed but UNSTANDARDIZED matrix
imputer_only = SimpleImputer(strategy="median")
X_imputed = imputer_only.fit_transform(X_raw)

pca_raw = PCA(n_components=2, random_state=RANDOM_STATE)
pca_raw.fit(X_imputed)

pca_scaled = PCA(n_components=2, random_state=RANDOM_STATE)
pca_scaled.fit(X_scaled)

raw_loadings = pd.Series(pca_raw.components_[0], index=FEATURES, name="Raw_PC1")
scaled_loadings = pd.Series(pca_scaled.components_[0], index=FEATURES, name="Scaled_PC1")

pca_loading_compare = pd.concat([raw_loadings, scaled_loadings], axis=1)
pca_loading_compare["abs_raw"] = pca_loading_compare["Raw_PC1"].abs()
pca_loading_compare["abs_scaled"] = pca_loading_compare["Scaled_PC1"].abs()

display(
    pca_loading_compare.sort_values("abs_scaled", ascending=False)
    .drop(columns=["abs_raw", "abs_scaled"])
    .round(3)
)


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

raw_sorted = raw_loadings.abs().sort_values(ascending=False).head(10)
scaled_sorted = scaled_loadings.abs().sort_values(ascending=False).head(10)

axes[0].barh(raw_sorted.index[::-1], raw_sorted.values[::-1])
axes[0].set_title("Top |PC1 loadings| — Unscaled PCA")
axes[0].set_xlabel("Absolute loading")

axes[1].barh(scaled_sorted.index[::-1], scaled_sorted.values[::-1])
axes[1].set_title("Top |PC1 loadings| — Standardized PCA")
axes[1].set_xlabel("Absolute loading")

plt.tight_layout()
plt.show()


# Compact comparison using the selected configurations.
summary_rows = []

# K-Means++
summary_rows.append({
    "Method": "K-Means++",
    "Configuration": f"K={K_BEST}",
    "Silhouette": silhouette_score(X_scaled, labels_plus_final),
    "Davies_Bouldin": davies_bouldin_score(X_scaled, labels_plus_final),
    "Clusters": len(np.unique(labels_plus_final))
})

# DBSCAN exploration result
if len(valid_db):
    summary_rows.append({
        "Method": "DBSCAN (sample)",
        "Configuration": f"eps={BEST_EPS}, min_samples={BEST_MIN_SAMPLES}",
        "Silhouette": float(best_db_row["silhouette_non_noise"]),
        "Davies_Bouldin": float(best_db_row["davies_bouldin_non_noise"]),
        "Clusters": int(best_db_row["clusters"])
    })

# Best hierarchical configuration by silhouette
best_hier = hierarchical_results.sort_values("silhouette", ascending=False).iloc[0]
summary_rows.append({
    "Method": f"Hierarchical ({best_hier['linkage']})",
    "Configuration": f"K={int(best_hier['K'])}",
    "Silhouette": float(best_hier["silhouette"]),
    "Davies_Bouldin": float(best_hier["davies_bouldin"]),
    "Clusters": int(best_hier["K"])
})

method_summary = pd.DataFrame(summary_rows)
display(method_summary.round(4))

plt.figure(figsize=(10, 5))
sns.barplot(data=method_summary, x="Method", y="Silhouette")
plt.xticks(rotation=20)
plt.title("Clustering Method Comparison — Silhouette")
plt.ylabel("Silhouette score")
plt.show()

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.base import clone

try:
    from xgboost import XGBClassifier
    MODEL_NAME = "XGBoost"
    clf = XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    MODEL_NAME = "HistGradientBoostingClassifier (XGBoost unavailable)"
    clf = HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE
    )

print("Final supervised model:", MODEL_NAME)




