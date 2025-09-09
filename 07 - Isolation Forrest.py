import numpy as np
import pandas as pd
import warnings

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score

df_raw = pd.read_parquet(".parquet")
df_pc = pd.read_parquet(".parquet")
print(df_raw.head())
print(df_pc.head())

pc_cols = [col for col in df_pc.columns if col.startswith("PC")]

# Scale the PC columns
scaler = StandardScaler()
df_pc_scaled = df_pc.copy()
df_pc_scaled[pc_cols] = scaler.fit_transform(df_pc[pc_cols])

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest


warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated and will be removed in a future version")

# Setup
pc_cols = [col for col in df_pc_scaled.columns if col.startswith("PC")]
X_scaled = df_pc_scaled[pc_cols]
n_samples = X_scaled.shape[0]

contamination_values = [0.03, 0.05, 0.1]
n_estimators_values = [100, 200, 300, 400, 500]

results = []
score_matrix = {}
global_min = float('inf')
global_max = float('-inf')

# Fit Isolation Forest models
for n_estimators in n_estimators_values:
    model = IsolationForest(
        contamination='auto',
        n_estimators=n_estimators,
        max_samples=n_samples,
        random_state=42
    )
    model.fit(X_scaled)

    scores = pd.Series(model.decision_function(X_scaled), index=X_scaled.index)
    scores_clean = scores.replace([np.inf, -np.inf], np.nan).dropna()

    global_min = min(global_min, scores_clean.min())
    global_max = max(global_max, scores_clean.max())

    for contamination in contamination_values:
        outlier_count = (scores_clean < scores_clean.quantile(contamination)).sum()
        results.append({
            'contamination': contamination,
            'n_estimators': n_estimators,
            'outliers_flagged': outlier_count,
            'score_min': scores_clean.min(),
            'score_max': scores_clean.max(),
            'score_median': scores_clean.median(),
            'score_std': scores_clean.std()
        })

        score_matrix[(n_estimators, contamination)] = scores_clean

# Summary DataFrame
results_df = pd.DataFrame(results)

# Plot grid
rows = len(n_estimators_values)
cols = 3
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(22, 34))
axes = axes.flatten()

for idx, (n_estimators, contamination) in enumerate([(n, c) for n in n_estimators_values for c in contamination_values]):
    ax = axes[idx]
    scores_clean = score_matrix[(n_estimators, contamination)]

    sns.histplot(scores_clean, bins=30, kde=False, color='#7f7f7f', ax=ax)
    cutoff = scores_clean.quantile(contamination)
    ax.axvline(x=cutoff, color='red', linestyle='--')

    ax.set_xlim(global_min, global_max)
    ax.set_xlabel("Anomaly Score", fontsize=21)
    ax.set_ylabel("Frequency", fontsize=21)
    ax.set_title(f"n = {n_estimators}, contamination = {contamination}", fontsize=22)
    ax.tick_params(axis='both', labelsize=19)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)

# Remove unused subplots
for i in range(len(n_estimators_values) * len(contamination_values), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout(pad=4.0, w_pad=3.0, h_pad=4.0)
plt.show()


# Config
contamination_levels = [0.03, 0.05]
estimator_range = [100, 200, 300, 400, 500, 600]

# Setup input data
pc_cols = [col for col in df_pc_scaled.columns if col.startswith("PC")]
X_scaled = df_pc_scaled[pc_cols]

# Plot setup
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for idx, contamination in enumerate(contamination_levels):
    outlier_counts = []
    for n_estimators in estimator_range:
        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=len(X_scaled),
            random_state=42
        )
        model.fit(X_scaled)
        preds = model.predict(X_scaled)
        n_outliers = (preds == -1).sum()
        outlier_counts.append(n_outliers)

 
    ax = axes[idx]
    ax.plot(estimator_range, outlier_counts, marker='o', linestyle='-', color='#1f4e79')  # dark blue
    ax.set_title(f'Contamination = {contamination}')
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('Outliers Flagged')
    ax.set_yticks(range(min(outlier_counts), max(outlier_counts) + 1))
    ax.grid(True)

plt.tight_layout()
plt.show()


import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score

# Parameters
contamination_levels = [0.03, 0.05]
n_estimators = 400
seeds = list(range(10, 110, 10))
sample_frac = 0.8

results = []

for contamination in contamination_levels:
    preds_dict = {}

    for seed in seeds:
        X_sampled = X_scaled.sample(frac=sample_frac, random_state=seed)

        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=len(X_sampled),
            random_state=seed
        )
        model.fit(X_sampled)
        preds = pd.Series(model.predict(X_scaled), index=X_scaled.index)
        preds_dict[seed] = preds

    for (s1, s2) in itertools.combinations(seeds, 2):
        labels_1 = preds_dict[s1]
        labels_2 = preds_dict[s2]

        ari = adjusted_rand_score(labels_1, labels_2)

        results.append({
            'contamination': contamination,
            'seed_pair': f"{s1}-{s2}",
            'ARI': ari
        })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Plot only ARI
plt.figure(figsize=(6, 6))
sns.boxplot(
    data=results_df,
    x="contamination", y="ARI",
    width=0.5,
    color='#7f7f7f',
    boxprops=dict(facecolor='#7f7f7f', edgecolor='none')
)
plt.xlabel("Contamination Level", fontsize=15)
plt.ylabel("Adjusted Rand Index", fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.grid(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

# Output median ARI values
median_ari = results_df.groupby('contamination')['ARI'].median()
print("\n📌 Median ARI by Contamination Level:")
print(median_ari.to_string(float_format="%.4f"))


# Final IsolationForest config
model = IsolationForest(
    contamination=0.05,
    n_estimators=400,
    max_samples=len(X_scaled),
    random_state=42
)

# Fit the model
model.fit(X_scaled)

# Predict: -1 = outlier, 1 = inlier
preds = pd.Series(model.predict(X_scaled), index=X_scaled.index)

# Get outlier indices
outlier_indices = preds[preds == -1].index

# Output
print(f"Total outliers flagged: {len(outlier_indices)}")
print("Outlier indices:")
print(list(outlier_indices))


metadata_cols = ['Material Code', 'Material Batch Nr', 'Component Code', 'Component Batch Nr']
cma_cols = [col for col in df_raw.columns if col not in metadata_cols]

# Compute IQR-based whiskers for full dataset
Q1 = df_raw[cma_cols].quantile(0.25)
Q3 = df_raw[cma_cols].quantile(0.75)
IQR = Q3 - Q1
lower_whisker = Q1 - 1.5 * IQR
upper_whisker = Q3 + 1.5 * IQR

# Retrieve raw feature values for outliers
df_outliers_iso = df_raw.loc[outlier_indices].copy()

# Check which values lie within whiskers
within_whiskers = df_outliers_iso[cma_cols].apply(
    lambda row: (row >= lower_whisker) & (row <= upper_whisker),
    axis=1
)

# Classify deviation type (same logic as DBSCAN case)
last_cma = cma_cols[-1]

def classify_deviation(row_mask):
    if row_mask.all():
        return 'Multivariate deviation'
    out_of_bounds = ~row_mask
    if out_of_bounds.sum() == 1 and out_of_bounds.index[out_of_bounds.argmax()] == last_cma:
        return 'Multivariate deviation'
    return 'Univariate deviation'

df_outliers_iso['Deviation Type'] = within_whiskers.apply(classify_deviation, axis=1)

# 6. Create final summary table
iso_deviation_summary = df_outliers_iso[metadata_cols + ['Deviation Type']]

print(iso_deviation_summary)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Copy PC dataframe and annotate status
df_plot = df_pc_scaled.copy()
df_plot['status'] = 'Core'

# metadata index to map deviation types
df_plot = df_plot.reset_index()
df_plot = df_plot.merge(
    iso_deviation_summary,
    on=['Material Code', 'Material Batch Nr', 'Component Code', 'Component Batch Nr'],
    how='left'
)
df_plot['status'] = df_plot['Deviation Type'].fillna('Core')

#jitter
jitter_strength = 0.03
df_plot['PC1_jittered'] = df_plot['PC1'] + np.random.normal(0, jitter_strength, size=len(df_plot))
df_plot['PC2_jittered'] = df_plot['PC2'] + np.random.normal(0, jitter_strength, size=len(df_plot))

color_map = {
    'Core': '#7f7f7f',
    'Univariate deviation': 'red',
    'Multivariate deviation': 'gold'
}
colors = df_plot['status'].map(color_map)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(
    df_plot['PC1_jittered'],
    df_plot['PC2_jittered'],
    c=colors,
    edgecolor='black',
    alpha=0.8,
    s=60
)

plt.xlabel("PC1", fontsize=15)
plt.ylabel("PC2", fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)


legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Core',
           markerfacecolor='#7f7f7f', markeredgecolor='black', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Univariate Outlier',
           markerfacecolor='red', markeredgecolor='black', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Multivariate Outlier',
           markerfacecolor='gold', markeredgecolor='black', markersize=10)
]
plt.legend(handles=legend_elements, loc='upper left', fontsize=13, frameon=True)

# Visual cleanup
ax = plt.gca()
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


df_plot_3d = df_pc_scaled.copy()

deviation_map = df_raw.loc[outlier_indices].index.to_series().map(
    iso_deviation_summary.set_index(df_raw.loc[outlier_indices].index)['Deviation Type']
)
df_plot_3d['status'] = 'Core'
df_plot_3d.loc[outlier_indices, 'status'] = deviation_map.values

jitter_strength = 0.03
df_plot_3d['PC1_jittered'] = df_plot_3d['PC1'] + np.random.normal(0, jitter_strength, size=len(df_plot_3d))
df_plot_3d['PC2_jittered'] = df_plot_3d['PC2'] + np.random.normal(0, jitter_strength, size=len(df_plot_3d))
df_plot_3d['PC3_jittered'] = df_plot_3d['PC3'] + np.random.normal(0, jitter_strength, size=len(df_plot_3d))


x = df_plot_3d['PC1_jittered']
y = df_plot_3d['PC2_jittered']
z = df_plot_3d['PC3_jittered']


color_map = {
    'Core': '#7f7f7f',
    'Univariate deviation': 'red',
    'Multivariate deviation': 'gold'
}
colors = df_plot_3d['status'].map(color_map)

# Plot setup
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# Manual box shift to expose Z label
box = ax.get_position()
ax.set_position([box.x0 - 0.05, box.y0, box.width * 0.95, box.height])

# Scatter
ax.scatter(x, y, z, c=colors, edgecolor='black', s=50, alpha=0.6)


ax.set_xlabel("PC1", labelpad=15, fontsize=15)
ax.set_ylabel("PC2", labelpad=15, fontsize=15)
ax.set_zlabel("PC3", labelpad=20, fontsize=15)
ax.tick_params(axis='both', labelsize=13)

# View angle
ax.view_init(elev=30, azim=100)


legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Core',
           markerfacecolor='#7f7f7f', markeredgecolor='black', markersize=8, linewidth=0),
    Line2D([0], [0], marker='o', color='w', label='Univariate Outlier',
           markerfacecolor='red', markeredgecolor='black', markersize=8, linewidth=0),
    Line2D([0], [0], marker='o', color='w', label='Multivariate Outlier',
           markerfacecolor='gold', markeredgecolor='black', markersize=8, linewidth=0)
]
ax.legend(
    handles=legend_elements,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.93),
    fontsize=13,
    frameon=True,
    ncol=3,
    title=None
)

plt.show()


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


df_plot_3d = df_pc_scaled.copy()


deviation_map = df_raw.loc[outlier_indices].index.to_series().map(
    iso_deviation_summary.set_index(df_raw.loc[outlier_indices].index)['Deviation Type']
)
df_plot_3d['status'] = 'Core'
df_plot_3d.loc[outlier_indices, 'status'] = deviation_map.values

# jitter to PCs 5–7
jitter_strength = 0.03
df_plot_3d['PC5_jittered'] = df_plot_3d['PC5'] + np.random.normal(0, jitter_strength, size=len(df_plot_3d))
df_plot_3d['PC6_jittered'] = df_plot_3d['PC6'] + np.random.normal(0, jitter_strength, size=len(df_plot_3d))
df_plot_3d['PC7_jittered'] = df_plot_3d['PC7'] + np.random.normal(0, jitter_strength, size=len(df_plot_3d))

x = df_plot_3d['PC5_jittered']
y = df_plot_3d['PC6_jittered']
z = df_plot_3d['PC7_jittered']

color_map = {
    'Core': '#7f7f7f',
    'Univariate deviation': 'red',
    'Multivariate deviation': 'gold'
}
colors = df_plot_3d['status'].map(color_map)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')


box = ax.get_position()
ax.set_position([box.x0 - 0.05, box.y0, box.width * 0.95, box.height])


ax.scatter(x, y, z, c=colors, edgecolor='black', s=50, alpha=0.6)


ax.set_xlabel("PC5", labelpad=15, fontsize=15)
ax.set_ylabel("PC6", labelpad=15, fontsize=15)
ax.set_zlabel("PC7", labelpad=20, fontsize=15)
ax.tick_params(axis='both', labelsize=13)

# View angle
ax.view_init(elev=30, azim=100)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Core',
           markerfacecolor='#7f7f7f', markeredgecolor='black', markersize=8, linewidth=0),
    Line2D([0], [0], marker='o', color='w', label='Univariate Outlier',
           markerfacecolor='red', markeredgecolor='black', markersize=8, linewidth=0),
    Line2D([0], [0], marker='o', color='w', label='Multivariate Outlier',
           markerfacecolor='gold', markeredgecolor='black', markersize=8, linewidth=0)
]
ax.legend(
    handles=legend_elements,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.93),
    fontsize=13,
    frameon=True,
    ncol=3,
    title=None
)

plt.show()

metadata_cols = ['Material Code', 'Material Batch Nr', 'Component Code', 'Component Batch Nr']
cma_cols = [col for col in df_raw.columns if col not in metadata_cols]

# Compute IQR-based whisker bounds from full dataset
Q1 = df_raw[cma_cols].quantile(0.25)
Q3 = df_raw[cma_cols].quantile(0.75)
IQR = Q3 - Q1
lower_whisker = Q1 - 1.5 * IQR
upper_whisker = Q3 + 1.5 * IQR

# Filter IsolationForest outliers from original raw dataset
df_outliers = df_raw.loc[outlier_indices].copy()


within_whiskers = df_outliers[cma_cols].apply(
    lambda row: (row >= lower_whisker) & (row <= upper_whisker),
    axis=1
)

# 5. Classify each outlier
last_cma = cma_cols[-1]

def classify_deviation(row_mask):
    if row_mask.all():
        return 'Multivariate deviation'
    out_of_bounds = ~row_mask
    if out_of_bounds.sum() == 1 and out_of_bounds.index[out_of_bounds.argmax()] == last_cma:
        return 'Multivariate deviation'
    return 'Univariate deviation'

df_outliers['Deviation Type'] = within_whiskers.apply(classify_deviation, axis=1)

isofor_deviation_summary = df_outliers[metadata_cols + ['Deviation Type']]

print(isofor_deviation_summary)


# Define metadata and CMA columns
metadata_cols = ['Material Code', 'Material Batch Nr', 'Component Code', 'Component Batch Nr']
cma_cols = [col for col in df_raw.columns if col not in metadata_cols]


Q1 = df_raw[cma_cols].quantile(0.25)
Q3 = df_raw[cma_cols].quantile(0.75)
IQR = Q3 - Q1
lower_whisker = Q1 - 1.5 * IQR
upper_whisker = Q3 + 1.5 * IQR


df_outliers = df_raw.loc[outlier_indices].copy()

# boolean mask for each CMA: True if univariate outlier
flagged_cmas = df_outliers[cma_cols].apply(
    lambda row: (row < lower_whisker) | (row > upper_whisker),
    axis=1
)

cma_flag_counts = flagged_cmas.sum()


cma_flag_counts = cma_flag_counts[cma_flag_counts > 0].sort_values(ascending=False)


cma_deviation_summary = pd.DataFrame({
    'CMA': cma_flag_counts.index,
    'Times Flagged': cma_flag_counts.values
})

# 8. Preview
print(cma_deviation_summary)

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

# top 3 CMAs (excluding most-flagged)
top_cmas = cma_deviation_summary['CMA'].iloc[[1, 2, 3]].tolist()

# deviation types
plot_data = df_raw.copy()
plot_data['status'] = 'Core'
plot_data = plot_data.set_index(metadata_cols)
plot_data.loc[
    isofor_deviation_summary.set_index(metadata_cols).index,
    'status'
] = isofor_deviation_summary.set_index(metadata_cols)['Deviation Type']
plot_data = plot_data.reset_index()

outlier_rows = plot_data[plot_data['status'] != 'Core'].index.tolist()
core_rows = plot_data[plot_data['status'] == 'Core'].index.tolist()
subset_plot = plot_data.loc[outlier_rows + core_rows, top_cmas + ['status']].copy()

jitter_strength = 0.06
for cma in top_cmas:
    subset_plot[cma] += np.random.normal(0, jitter_strength, size=len(subset_plot))


x = subset_plot[top_cmas[0]]
y = subset_plot[top_cmas[1]]
z = subset_plot[top_cmas[2]]


color_map = {
    'Core': '#7f7f7f',
    'Univariate deviation': 'red',
    'Multivariate deviation': 'gold'
}
colors = subset_plot['status'].map(color_map)


fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')


box = ax.get_position()
ax.set_position([box.x0 - 0.05, box.y0, box.width * 0.95, box.height])

# Scatter
ax.scatter(
    x, y, z,
    c=colors,
    edgecolor='black',
    s=50,
    alpha=0.6
)


ax.set_xlabel(top_cmas[0], labelpad=15, fontsize=15)
ax.set_ylabel(top_cmas[1], labelpad=15, fontsize=15)
ax.set_zlabel(top_cmas[2], labelpad=20, fontsize=15)
ax.tick_params(axis='both', labelsize=13)

# View angle
ax.view_init(elev=30, azim=100)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Core',
           markerfacecolor='#7f7f7f', markeredgecolor='black', markersize=8, linewidth=0),
    Line2D([0], [0], marker='o', color='w', label='Univariate Outlier',
           markerfacecolor='red', markeredgecolor='black', markersize=8, linewidth=0),
    Line2D([0], [0], marker='o', color='w', label='Multivariate Outlier',
           markerfacecolor='gold', markeredgecolor='black', markersize=8, linewidth=0)
]
ax.legend(
    handles=legend_elements,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.93),
    fontsize=13,
    frameon=True,
    ncol=3,
    title=None
)

plt.show()


# metadata for IsoFor-flagged samples
isofor_outlier_meta = df_raw.loc[outlier_indices, metadata_cols]

# Match against df_raw to get full feature rows
isofor_outlier_rows = isofor_outlier_meta.merge(df_raw, on=metadata_cols, how='left')

# identifier for source model
isofor_outlier_rows.insert(0, "Flagged By", "IsolationForest")


print(isofor_outlier_rows)


# dbfs_path = ".parquet" 
# isofor_outlier_rows.to_parquet(dbfs_path, index=False)

# # Hash each row's metadata as a string
# raw_hashes = df_raw[metadata_cols].astype(str).agg('|'.join, axis=1)
# pca_hashes = df_pc_scaled[metadata_cols].astype(str).agg('|'.join, axis=1)

# # Compare
# aligned = raw_hashes.equals(pca_hashes)
# print("Are row orders aligned?", aligned)


