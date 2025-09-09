%pip install --upgrade matplotlib

dbutils.library.restartPython()

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

parquet_path = ".parquet"

# Load the data
df = pd.read_parquet(parquet_path)

print("Shape:", df.shape)
df.head()



metadata_cols = ['Material Code', 'Material Batch Nr', 'Component Code', 'Component Batch Nr']


cma_cols = [col for col in df.columns if col not in metadata_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[cma_cols])

df_scaled = pd.DataFrame(X_scaled, columns=cma_cols, index=df.index)
df_scaled.head()

pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(df_scaled)
pca_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
df_pca = pd.DataFrame(X_pca, columns=pca_cols, index=df.index)

# metadata for visualization
df_pca_full = pd.concat([df[metadata_cols], df_pca], axis=1)

# Explained variance ratio 
explained_var = pca.explained_variance_ratio_

# Loadings matrix (for heatmap and feature contributions)
loadings = pd.DataFrame(
    pca.components_.T, 
    columns=pca_cols, 
    index=cma_cols
)

print("Number of components retained:", pca.n_components_)

import matplotlib.pyplot as plt
import numpy as np

# Cumulative and individual variance explained
cum_var = np.cumsum(explained_var)
ind_var = explained_var


color_cumulative = '#1f4e79'  
color_individual = 'lightgrey' 
color_threshold = 'red'  

# Plot
plt.figure(figsize=(10, 7.5))
plt.plot(
    range(1, len(ind_var) + 1),
    ind_var,
    marker='o',
    label='Individual Explained Variance',
    color=color_individual
)
plt.plot(
    range(1, len(cum_var) + 1),
    cum_var,
    marker='s',
    linestyle='--',
    label='Cumulative Explained Variance',
    color=color_cumulative
)
plt.axhline(
    y=0.95,
    color=color_threshold,
    linestyle='--',
    label='95% Threshold'
)


plt.xlabel('Principal Component', fontsize=15)
plt.ylabel('Explained Variance Ratio', fontsize=15)
plt.xticks(range(1, len(ind_var) + 1), fontsize=13)
plt.yticks(fontsize=13)


plt.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.1),
    ncol=3,
    frameon=False,
    fontsize=13
)


ax = plt.gca()
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()




# loadings for PC1
pc1_loadings = loadings["PC1"].sort_values(ascending=False)

# original variance of CMA columns
cma_variances = df[cma_cols].var()

# PC1 deep dive dataframe
pc1_df = pd.DataFrame({
    "Loading": pc1_loadings
})
pc1_df["Abs Loading"] = pc1_df["Loading"].abs()
pc1_df = pc1_df.sort_values(by="Abs Loading", ascending=False)

# full PC1 table (not truncated)
from IPython.display import display
display(pc1_df)

# loadings for PC2
pc2_loadings = loadings["PC2"].sort_values(ascending=False)

# PC2 deep dive dataframe
pc2_df = pd.DataFrame({
    "Loading": pc2_loadings
})
pc2_df["Abs Loading"] = pc2_df["Loading"].abs()
pc2_df = pc2_df.sort_values(by="Abs Loading", ascending=False)

#Display full PC2 table (not truncated)
from IPython.display import display
display(pc2_df)

import matplotlib.pyplot as plt
import seaborn as sns

# order features by total absolute contribution
sorted_features = (loadings.abs().sum(axis=1)
                   .sort_values(ascending=False)
                   .index)

# Subset and reorder loadings matrix
heatmap_data = loadings.loc[sorted_features]


plt.figure(figsize=(10, 8)) 
sns.heatmap(
    heatmap_data,
    cmap="coolwarm",
    center=0,
    annot=False,
    linewidths=0.5,
    cbar_kws={"label": "Loading"}
)


plt.xlabel("Principal Components", fontsize=15)
plt.ylabel("CMA Features", fontsize=15)
plt.xticks(rotation=45, ha='right', fontsize=13)
plt.yticks(fontsize=13)

colorbar = plt.gca().collections[0].colorbar
colorbar.ax.yaxis.label.set_size(13)


ax = plt.gca()
ax.set_title("")


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()





squared_loadings = loadings ** 2  

# Weight each column by explained variance
weighted_contrib = squared_loadings.mul(explained_var, axis=1)

#Sum across all components to get total contribution per feature
global_contrib = weighted_contrib.sum(axis=1)

# Sort descending to see which features dominate
global_contrib = global_contrib.sort_values(ascending=False)

# top contributors
print(global_contrib)


import matplotlib.pyplot as plt
import numpy as np

# Jitter setup
jitter_strength = 0.03
x_jittered = df_pca["PC1"] + np.random.normal(0, jitter_strength, size=len(df_pca))
y_jittered = df_pca["PC2"] + np.random.normal(0, jitter_strength, size=len(df_pca))


pc1_var = explained_var[0] * 100 
pc2_var = explained_var[1] * 100
x_label = f"PC 1 ({pc1_var:.1f}%)"
y_label = f"PC 2 ({pc2_var:.1f}%)"

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(
    x_jittered,
    y_jittered,
    color="#7f7f7f",     
    edgecolor="black",
    alpha=0.6,
    s=50
)


plt.xlabel(x_label, fontsize=15)
plt.ylabel(y_label, fontsize=15)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)


ax = plt.gca()
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


jitter_strength = 0.03
x = df_pca['PC1'] + np.random.normal(0, jitter_strength, size=len(df_pca))
y = df_pca['PC2'] + np.random.normal(0, jitter_strength, size=len(df_pca))
z = df_pca['PC3'] + np.random.normal(0, jitter_strength, size=len(df_pca))

# Variance percentages
pc1_var = explained_var[0] * 100
pc2_var = explained_var[1] * 100
pc3_var = explained_var[2] * 100


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

box = ax.get_position()
ax.set_position([box.x0 - 0.05, box.y0, box.width * 0.95, box.height])

# Scatter plot
ax.scatter(
    x, y, z,
    color="#7f7f7f",  
    edgecolor="black",
    alpha=0.6,
    s=50
)


ax.set_xlabel(f"PC1 ({pc1_var:.1f}%)", labelpad=15, fontsize=12)
ax.set_ylabel(f"PC2 ({pc2_var:.1f}%)", labelpad=15, fontsize=12)
ax.set_zlabel(f"PC3 ({pc3_var:.1f}%)", labelpad=20, fontsize=12)

# View angle
ax.view_init(elev=30, azim=100)

plt.tight_layout()
plt.show()


print(df_pca_full.head())
print(df_pca_full.shape)

# Save masked version
# df_pca_full.to_parquet('.parquett', index=False)
# print("masked dataset saved successfully!")

# # Save unmasked version
# df_pca_full.to_parquet('.parquet', index=False)
# print("unmasked dataset saved successfully!")

