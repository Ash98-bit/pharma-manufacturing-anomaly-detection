import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)


pd.set_option('display.max_colwidth', None)

df_raw = pd.read_parquet(".parquet")
df_pc = pd.read_parquet(".parquet")
print(df_raw.head())
print(df_pc.head())
print(df_raw.shape)
print(df_pc.shape)

# Same CMA columns as before
cma_columns = ['CMA2', 'CMA3', 'CMA4', 'CMA12', 'CMA14', 'CMA15', 'CMA16', 'CMA17']


profile_counts = df_raw[cma_columns].value_counts().reset_index(name='frequency')

# group by the 'frequency' column, count how many profiles exist at each frequency
frequency_distribution = profile_counts['frequency'].value_counts().reset_index()
frequency_distribution.columns = ['frequency', 'num_profiles']

# Sort
frequency_distribution = frequency_distribution.sort_values(by='frequency', ascending=False).reset_index(drop=True)

print(frequency_distribution)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

cma_columns = ['CMA2', 'CMA3', 'CMA4', 'CMA12', 'CMA14', 'CMA15', 'CMA16', 'CMA17']
cma_only = df_raw[cma_columns]

# Count unique CMA patterns
profile_counts = cma_only.value_counts().reset_index(name='frequency')
profile_counts['CMA_Profile_ID'] = ['CMA_Profile_' + str(i) for i in range(len(profile_counts))]

# Merge with original DataFrame
df_with_profiles = df_raw.merge(profile_counts, on=cma_columns, how='left')

# Grouping
df_with_profiles['Profile_Group'] = np.where(
    df_with_profiles['frequency'] > 5,
    'Frequent',
    'Rare'
)

# PCs and jitter
np.random.seed(42)
jitter_scale = 0.1
df_with_profiles['PC1'] = df_pc['PC1'] + np.random.normal(0, jitter_scale, size=len(df_pc))
df_with_profiles['PC2'] = df_pc['PC2'] + np.random.normal(0, jitter_scale, size=len(df_pc))
df_with_profiles['PC3'] = df_pc['PC3'] + np.random.normal(0, jitter_scale, size=len(df_pc))


color_map = {'Frequent': '#7f7f7f', 'Rare': '#1f4e79'}
colors = df_with_profiles['Profile_Group'].map(color_map)

# 3D Plot setup
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

# expose Z label
box = ax.get_position()
ax.set_position([box.x0 - 0.05, box.y0, box.width * 0.95, box.height])

# Scatter plot
ax.scatter(
    df_with_profiles['PC1'],
    df_with_profiles['PC2'],
    df_with_profiles['PC3'],
    c=colors,
    alpha=0.8,
    s=50,
    edgecolor='none'
)


ax.set_xlabel('PC1', labelpad=15, fontsize=15)
ax.set_ylabel('PC2', labelpad=15, fontsize=15)
ax.set_zlabel('PC3', labelpad=20, fontsize=15)
ax.tick_params(axis='both', labelsize=13)
ax.view_init(elev=30, azim=100)


from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Rare Profiles',
           markerfacecolor='#1f4e79', markeredgecolor='none', markersize=8, linewidth=0),
    Line2D([0], [0], marker='o', color='w', label='Frequent Profiles',
           markerfacecolor='#7f7f7f', markeredgecolor='none', markersize=8, linewidth=0)
]
ax.legend(
    handles=legend_elements,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.93),
    fontsize=13,
    frameon=True,
    ncol=2,
    title=None
)

plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


cma_columns = ['CMA2', 'CMA3', 'CMA4', 'CMA5', 'CMA12', 'CMA14', 'CMA15', 'CMA16', 'CMA17', 'CMA18']
cma_only = df_raw[cma_columns]

# frequency of each CMA pattern
profile_counts = cma_only.value_counts().reset_index(name='frequency')
profile_counts['CMA_Profile_ID'] = ['CMA_Profile_' + str(i) for i in range(len(profile_counts))]

# Merge profile info back to df_raw
df_with_profiles = df_raw.merge(profile_counts, on=cma_columns, how='left')

# rare profiles
df_with_profiles['Profile_Group'] = df_with_profiles['CMA_Profile_ID']
df_with_profiles.loc[df_with_profiles['frequency'] == 1, 'Profile_Group'] = 'Rare_Profile'

# PC1, PC2, PC3 from df_pc
df_with_profiles[['PC1', 'PC2', 'PC3']] = df_pc[['PC1', 'PC2', 'PC3']].values

# Map profile groups to integer colors
unique_profiles = df_with_profiles['Profile_Group'].unique()
profile_to_color = {profile: idx for idx, profile in enumerate(unique_profiles)}
colors = df_with_profiles['Profile_Group'].map(profile_to_color)


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    df_with_profiles['PC1'],
    df_with_profiles['PC2'],
    df_with_profiles['PC3'],
    c=colors,
    cmap='tab20',
    alpha=0.7,
    s=40
)

ax.set_title('3D PCA: Colored by CMA Profile Group')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')


plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


cma_columns = ['CMA2', 'CMA3', 'CMA4', 'CMA5', 'CMA12', 'CMA14', 'CMA15', 'CMA16', 'CMA17', 'CMA18']
cma_only = df_raw[cma_columns]


profile_counts = cma_only.value_counts().reset_index(name='frequency')
profile_counts['CMA_Profile_ID'] = ['CMA_Profile_' + str(i) for i in range(len(profile_counts))]


df_with_profiles = df_raw.merge(profile_counts, on=cma_columns, how='left')

# rare profiles
df_with_profiles['Profile_Group'] = df_with_profiles['CMA_Profile_ID']
df_with_profiles.loc[df_with_profiles['frequency'] == 1, 'Profile_Group'] = 'Rare_Profile'


df_with_profiles[['PC4', 'PC5', 'PC6', 'PC7', 'PC8']] = df_pc[['PC4', 'PC5', 'PC6', 'PC7', 'PC8']].values


unique_profiles = df_with_profiles['Profile_Group'].unique()
profile_to_color = {profile: idx for idx, profile in enumerate(unique_profiles)}
colors = df_with_profiles['Profile_Group'].map(profile_to_color)

# 3D projections
fig = plt.figure(figsize=(16, 7))

# ---- PCA 4-5-6 ----
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(
    df_with_profiles['PC4'],
    df_with_profiles['PC5'],
    df_with_profiles['PC6'],
    c=colors,
    cmap='tab20',
    alpha=0.75,
    s=35
)
ax1.set_title('PCA Space: PC4 vs PC5 vs PC6')
ax1.set_xlabel('PC4')
ax1.set_ylabel('PC5')
ax1.set_zlabel('PC6')

# ---- PCA 6-7-8 ----
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(
    df_with_profiles['PC6'],
    df_with_profiles['PC7'],
    df_with_profiles['PC8'],
    c=colors,
    cmap='tab20',
    alpha=0.75,
    s=35
)
ax2.set_title('PCA Space: PC6 vs PC7 vs PC8')
ax2.set_xlabel('PC6')
ax2.set_ylabel('PC7')
ax2.set_zlabel('PC8')

plt.tight_layout()
plt.show()




