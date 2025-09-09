%pip install openpyxl
dbutils.library.restartPython()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, to_date, create_map, isnan, when, lit, sum as spark_sum
from pyspark.sql.types import StringType, DateType, FloatType, DoubleType

from functools import reduce
from itertools import chain
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from decimal import Decimal, ROUND_HALF_UP, getcontext
import openpyxl

#Read data
df = spark.read.option("header", True).option("inferSchema", True).csv("")


num_rows = df.count()
num_cols = len(df.columns)
print(f"Shape of the data: ({num_rows}, {num_cols})")
display(df.limit(10))

from pyspark.sql.functions import col, count, sum, when


df = df.withColumnRenamed("Batch Nr", "Material Batch Nr")

#Drop unnecessary columns
df = df.drop("Plant", "Manufacturing Date")

#Find the most manufactured Material Code
material_counts = df.groupBy("Material Code").agg(count("*").alias("batch_count"))
material_counts = material_counts.orderBy(col("batch_count").desc())

top_material_row = material_counts.first()
top_material_code = top_material_row["Material Code"]
top_material_batch_count = top_material_row["batch_count"]

print(f"Material Code with most batches: {top_material_code} ({top_material_batch_count} batches)")

#Find the most used Component Code for the top Material Code
filtered_material_df = df.filter(col("Material Code") == top_material_code)

component_counts = filtered_material_df.groupBy("Component Code").agg(count("*").alias("batch_count"))
component_counts = component_counts.orderBy(col("batch_count").desc())

top_component_row = component_counts.first()
top_component_code = top_component_row["Component Code"]
top_component_batch_count = top_component_row["batch_count"]

print(f"📦 Most frequent Component Code in {top_material_code}: {top_component_code} ({top_component_batch_count} batches)")


df_filtered = df.filter(
    (col("Material Code") == top_material_code) & 
    (col("Component Code") == top_component_code)
)

#Drop columns that are completely null in the filtered dataframe
non_null_summary = df_filtered.select([
    sum(when(col(c).isNotNull(), 1).otherwise(0)).alias(c)
    for c in df_filtered.columns
])

non_null_counts = non_null_summary.collect()[0].asDict()

non_null_cols = [col_name for col_name, non_null_count in non_null_counts.items() if non_null_count > 0]

df_filtered = df_filtered.select(*non_null_cols)


print(f"Filtered and cleaned Data Shape: ({df_filtered.count()}, {len(df_filtered.columns)})")
# display(df_filtered.limit(10))

# tests per component batch, redefining the meta cols to include this parameter too 

# Define metadata columns
metadata_cols = ["Material Code", "Component Code", "Material Batch Nr", "Component Batch Nr"]


cma_test_cols = [c for c in df_filtered.columns if c not in metadata_cols]


test_count_expr = reduce(
    lambda a, b: a + b,
    [when(col(c).isNotNull(), lit(1)).otherwise(lit(0)) for c in cma_test_cols]
)

# Add new column
df_filtered = df_filtered.withColumn("Number of CMA Tests Per Batch", test_count_expr)


metadata_cols = [
    "Material Code", 
    "Component Code", 
    "Material Batch Nr", 
    "Component Batch Nr", 
    "Number of CMA Tests Per Batch" 
]


all_columns = metadata_cols + [c for c in df_filtered.columns if c not in metadata_cols]


df_filtered = df_filtered.select(*all_columns)

# Display to verify
print(f"Final Data Shape after column reordering: ({df_filtered.count()}, {len(df_filtered.columns)})")
display(df_filtered.limit(10))


from pyspark.sql.types import StringType, FloatType, IntegerType


string_cols = ["Material Code", "Component Code", "Material Batch Nr", "Component Batch Nr"]
count_col = "Number of CMA Tests Per Batch"
cma_test_cols = [c for c in df_filtered.columns if c not in string_cols + [count_col]]


for col_name in string_cols:
    df_filtered = df_filtered.withColumn(col_name, col(col_name).cast(StringType()))

df_filtered = df_filtered.withColumn(count_col, col(count_col).cast(IntegerType()))

for col_name in cma_test_cols:
    df_filtered = df_filtered.withColumn(col_name, col(col_name).cast(FloatType()))


df_filtered.printSchema()

#function

def mask_data(df, apply_masking_1=True):
    if not apply_masking_1:
        return df, {}

    from pyspark.sql.functions import lit, create_map, col
    from pyspark.sql import functions as F
    from itertools import chain

 
    metadata_cols = [
        "Material Code",
        "Component Code",
        "Material Batch Nr",
        "Component Batch Nr",
        "Number of CMA Tests Per Batch"
    ]

    # Create the mappings
    unique_materials = df.select("Material Code").distinct().rdd.flatMap(lambda x: x).collect()
    material_mapping = {mat: f"M{i+1}" for i, mat in enumerate(unique_materials)}

    unique_components = df.select("Component Code").distinct().rdd.flatMap(lambda x: x).collect()
    component_mapping = {comp: f"C{i+1}" for i, comp in enumerate(unique_components)}

    unique_batches = df.select("Material Batch Nr").distinct().rdd.flatMap(lambda x: x).collect()
    batch_mapping = {bn: f"MB{i+1}" for i, bn in enumerate(unique_batches)}

    unique_comp_batches = df.select("Component Batch Nr").distinct().rdd.flatMap(lambda x: x).collect()
    comp_batch_mapping = {cbn: f"CB{i+1}" for i, cbn in enumerate(unique_comp_batches)}

    # Replace function
    def replace_column(df, colname, mapping_dict):
        mapping_expr = create_map([lit(x) for x in chain(*mapping_dict.items())])
        return df.withColumn(colname, mapping_expr.getItem(col(colname)))

    
    df = replace_column(df, "Material Code", material_mapping)
    df = replace_column(df, "Component Code", component_mapping)
    df = replace_column(df, "Material Batch Nr", batch_mapping)
    df = replace_column(df, "Component Batch Nr", comp_batch_mapping)

    # Mask CMA test column names
    cma_cols = [c for c in df.columns if c not in metadata_cols]

    cma_col_mapping = {old: f"CMA{i+1}" for i, old in enumerate(cma_cols)}

    for old_name, new_name in cma_col_mapping.items():
        df = df.withColumnRenamed(old_name, new_name)

    
    mappings = {
        "material_mapping": material_mapping,
        "component_mapping": component_mapping,
        "batch_mapping": batch_mapping,
        "comp_batch_mapping": comp_batch_mapping,
        "cma_col_mapping": cma_col_mapping
    }

    return df, mappings

# BIG GREEN MASKING BUTTON

apply_masking_1 = 

df_masked, mappings = mask_data(df_filtered, apply_masking_1)
display(df_masked.limit(10))

save_path = "./"  

# Save each mapping to CSV 

# for mapping_name, mapping_dict in mappings.items():
#     df_mapping = pd.DataFrame(list(mapping_dict.items()), columns=["Original", "Masked"])
#     df_mapping.to_csv(os.path.join(save_path, f"{mapping_name}.csv"), index=False)


if apply_masking_1:
    working_df, mappings = mask_data(df_filtered, apply_masking_1)
else:
    working_df = df_filtered

# display(working_df.limit(10))

import matplotlib.pyplot as plt

metadata_cols = [
    "Material Code",
    "Component Code",
    "Material Batch Nr",
    "Component Batch Nr",
    "Number of CMA Tests Per Batch"
]


cma_cols = [c for c in working_df.columns if c not in metadata_cols]

# Convert Spark to Pandas
pdf_working = working_df.toPandas()

total_cma_tests = len(cma_cols)

pdf_working["Original Tests"] = pdf_working["Number of CMA Tests Per Batch"]
pdf_working["Imputed Tests"] = total_cma_tests - pdf_working["Original Tests"]

# Sort by original tests (ascending for better barh visual)
# pdf_working_sorted = pdf_working.sort_values(by="Original Tests", ascending=True).reset_index(drop=True)


# plt.figure(figsize=(14, 8))
# bar_width = 0.6
# indices = range(len(pdf_working_sorted))

# # Plot original tests - DARK BLUE
# plt.barh(indices,
#          pdf_working_sorted["Original Tests"],
#          height=bar_width,
#          label="Original",
#          color="#1f4e79")  # dark blue

# # Plot imputed tests - GREY
# plt.barh(indices,
#          pdf_working_sorted["Imputed Tests"],
#          height=bar_width,
#          left=pdf_working_sorted["Original Tests"],
#          label="Imputed",
#          color="lightgrey")

# # Labels and styling
# plt.xlabel('Number of CMA Tests', fontsize=18)
# plt.ylabel('Component Batches (Sorted by # Original Tests)', fontsize=18)
# # plt.title('Original vs Imputed CMA Tests per Component Batch', fontsize=14, weight='bold')


# plt.legend(loc='upper right', fontsize=14)


# plt.grid(axis='x', linestyle='--', alpha=0.7)


# total_batches = len(pdf_working_sorted)
# yticks_pos = [0]
# yticks_labels = [f"{total_batches}"]
# plt.yticks(yticks_pos, yticks_labels, fontsize=12)

# plt.tight_layout()
# plt.show()
import matplotlib.pyplot as plt
import pandas as pd


pdf_working["Original Tests"] = pdf_working["Number of CMA Tests Per Batch"]
pdf_working["Imputed Tests"] = total_cma_tests - pdf_working["Original Tests"]


grouped = pdf_working.groupby("Original Tests").size().reset_index(name="Batch Count")


grouped["% Original"] = grouped["Original Tests"] / total_cma_tests
grouped["% Imputed"] = 1 - grouped["% Original"]


grouped = grouped.sort_values(by="Batch Count", ascending=False).reset_index(drop=True)

# Prepare data for plotting
x_labels = grouped["Batch Count"].astype(str).tolist()
bottom_vals = grouped["% Original"] * total_cma_tests
top_vals = grouped["% Imputed"] * total_cma_tests

fig, ax = plt.subplots(figsize=(9, 6))
bars1 = ax.bar(x_labels, bottom_vals, color="#1f4e79", label="Original Tests")
bars2 = ax.bar(x_labels, top_vals, bottom=bottom_vals, color="grey", label="Imputed Tests")

ax.set_yticks([total_cma_tests])
ax.set_yticklabels([str(total_cma_tests)], fontsize=13)

ax.set_xlabel("Number of Batch Instances with same number of Original Test", fontsize=15)
ax.set_ylabel("Total Number of CMA Tests", fontsize=15)
ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels, fontsize=13)
ax.tick_params(axis='y', labelsize=13)


ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    fontsize=13,
    frameon=True,
    title=None
)


ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()



df = pd.read_excel('/dbfs/FileStore/T_N_CMAs_post_mapping.xlsx', sheet_name='List1')
df = df.rename(columns={'Batch Nr': 'Material Batch Nr'})
df = df.rename(columns={"Total Tests for the Comp Batch": "Original Tests"})

print(f"Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")



#  Metadata columns
metadata_cols = [
    "Material Code",
    "Component Code",
    "Material Batch Nr",
    "Component Batch Nr",
    "Original Tests"
]


cma_cols = [c for c in df_masked.columns if c not in metadata_cols]

# Print the number of CMA columns
print(f" Number of CMA tests (columns) after pruning = {len(cma_cols)}")


#MASKING LOGIC

def mask_data_pandas(df, apply_masking_2=True):
    if not apply_masking_2:
        return df, {}
    
    import pandas as pd
    import os
    
    # Define metadata columns and prefixes
    metadata_columns = {
        'Material Code': 'M',
        'Component Code': 'C',
        'Material Batch Nr': 'MB',
        'Component Batch Nr': 'CB'
    }

    # Mask metadata columns
    metadata_mappings = {}
    
    for col_name, prefix in metadata_columns.items():
        unique_vals = df[col_name].dropna().unique()
        mapping_dict = {val: f"{prefix}{i+1}" for i, val in enumerate(unique_vals)}
        
        # Save mapping
        metadata_mappings[col_name] = mapping_dict
        
        # Apply mapping
        df[col_name] = df[col_name].map(mapping_dict)

    # Mask CMA test columns
    non_cma_cols = list(metadata_columns.keys()) + ["Original Tests"]
    cma_columns = [c for c in df.columns if c not in non_cma_cols]

    cma_col_mapping = {orig: f"CMA{i+1}" for i, orig in enumerate(cma_columns)}

    df.rename(columns=cma_col_mapping, inplace=True)

    # Save mappings 
    mappings = {
        "material_mapping": metadata_mappings.get("Material Code", {}),
        "component_mapping": metadata_mappings.get("Component Code", {}),
        "material_batch_mapping": metadata_mappings.get("Material Batch Nr", {}),
        "component_batch_mapping": metadata_mappings.get("Component Batch Nr", {}),
        "cma_col_mapping": cma_col_mapping
    }

    return df, mappings

# BIG GREEN MASKING BUTTON

apply_masking_2 =   

df_masked, mappings = mask_data_pandas(df, apply_masking_2=apply_masking_2)
# Step 4: Optional rename based on flags
print(f"Masking applied: {apply_masking_2}")
print(df_masked.columns.tolist())
print(df_masked.head())

from decimal import Decimal

# Metadata columns
metadata_cols = [
    "Material Code",
    "Component Code",
    "Material Batch Nr",
    "Component Batch Nr",
    "Original Tests"
]

# Identify CMA columns
cma_cols = [c for c in df_masked.columns if c not in metadata_cols]

# Cast metadata columns (except 'Original Tests') to string
for col in metadata_cols[:-1]:
    df_masked[col] = df_masked[col].astype(str)

# Cast 'Original Tests' to integer
df_masked["Original Tests"] = df_masked["Original Tests"].astype(int)

# Cast CMA columns to Decimal
# Replace blanks with NaN
for col in cma_cols:
    df_masked[col] = df_masked[col].replace(' ', np.nan)  
    df_masked[col] = df_masked[col].dropna().apply(lambda x: Decimal(str(x)))

print("Type casting complete (with blank space handling)!")
print("Typecasting to Decimal (without float conversion) complete!")

print(df_masked.dtypes)
print(df_masked.head())

# Validation – check that all non-null values in CMA columns are Decimal
invalid_cols = {}

for col in cma_cols:
    bad_vals = df_masked[col].dropna().map(type) != Decimal
    if bad_vals.any():
        n_bad = bad_vals.sum()
        examples = df_masked[col][bad_vals].unique().tolist()
        invalid_cols[col] = (n_bad, examples)

if not invalid_cols:
    print("All non-null values in all CMA columns are correctly typed as Decimal.")
else:
    print("Type mismatch found in the following CMA columns:")
    for col, (n_bad, examples) in invalid_cols.items():
        print(f"  - {col}: {n_bad} non-Decimal values (e.g., {examples[:3]})")


test_counts = df["Original Tests"].value_counts().sort_index()


y_labels = (test_counts.index / len(cma_cols) * 100).round().astype(int).astype(str) + "%"
x = test_counts.values


plt.figure(figsize=(10, 6))
plt.barh(y_labels, x, color="#7f7f7f") 


plt.ylabel("% of Total CMA Tests", fontsize=15)
plt.xlabel("Number of Batch Instances", fontsize=15)
plt.xticks([], []) 
# Add text labels next to each bar
for i, val in enumerate(x):
    plt.text(val + 0.5, i, str(val), va='center', fontsize=13)
plt.yticks(fontsize=13)
plt.grid(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.show()


max_tests = df_masked["Original Tests"].max()


threshold = 0.6 * max_tests
print(f"Max number of CMA tests observed: {max_tests}")
print(f"Threshold for pruning (60% of max): {threshold}")


start_count = df_masked.shape[0]

df_masked_pruned = df_masked[df_masked["Original Tests"] >= threshold].reset_index(drop=True)

end_count = df_masked_pruned.shape[0]

#Report
print(f"Batches before pruning: {start_count}")
print(f"Batches after pruning: {end_count}")
print(f"Batches dropped: {start_count - end_count}")


from decimal import Decimal
from statistics import median

#Median Imputation
for col in cma_cols:
    df_masked_pruned[col] = df_masked_pruned[col].replace(r'^\s*$', np.nan, regex=True)  

    
    df_masked_pruned[col] = df_masked_pruned[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else x)

    non_na_vals = [x for x in df_masked_pruned[col] if isinstance(x, Decimal)]
    if not non_na_vals:
        print(f"Skipping {col}: no valid values to compute median.")
        continue

    median_val = median(non_na_vals)

    df_masked_pruned[col] = df_masked_pruned[col].apply(lambda x: median_val if pd.isna(x) else x)

print(f"Median imputation complete for {len(cma_cols)} CMA tests.")
print(df_masked_pruned.columns.tolist())
print(df_masked_pruned.head())

null_counts = df_masked_pruned.isnull().sum()

null_counts = null_counts[null_counts > 0]

if null_counts.empty:
    print("No nulls found in the DataFrame. Imputation successful!")
else:
    print("Warning: Some columns still have nulls:")
    print(null_counts)


metadata_cols = [
    "Material Code",
    "Component Code",
    "Material Batch Nr",
    "Component Batch Nr",
    "Original Tests"   


cma_cols = [c for c in df_masked_pruned.columns if c not in metadata_cols]


total_cma_tests = len(cma_cols)


df_masked_pruned["Imputed Tests"] = total_cma_tests - df_masked_pruned["Original Tests"]

print("'Imputed Tests' column created successfully!")

# Sort df_masked_pruned by Original Tests descending
df_masked_pruned = df_masked_pruned.sort_values(by="Original Tests", ascending=True).reset_index(drop=True)

# print("df_masked_pruned sorted by Original Tests (highest first).")
print(df_masked_pruned.dtypes)
display(df_masked_pruned.head())


import matplotlib.pyplot as plt
import pandas as pd

metadata_cols = [
    "Material Code",
    "Component Code",
    "Material Batch Nr",
    "Component Batch Nr",
    "Original Tests"
]
cma_cols = [c for c in df_masked_pruned.columns if c not in metadata_cols + ["Imputed Tests"]]


total_cma_tests = len(cma_cols)


grouped = df_masked_pruned.groupby("Original Tests").size().reset_index(name="Batch Count")


grouped["% Original"] = grouped["Original Tests"] / total_cma_tests
grouped["% Imputed"] = 1 - grouped["% Original"]


grouped = grouped.sort_values(by="Batch Count", ascending=False).reset_index(drop=True)


x_labels = grouped["Batch Count"].astype(str).tolist()
bottom_vals = grouped["% Original"] * total_cma_tests
top_vals = grouped["% Imputed"] * total_cma_tests


fig, ax = plt.subplots(figsize=(9, 6))
ax.bar(x_labels, bottom_vals, color="#1f4e79", label="Original Tests")  # Dark blue
ax.bar(x_labels, top_vals, bottom=bottom_vals, color="#7f7f7f", label="Imputed Tests")  # Grey

# Step 8: Axis and legend styling
ax.set_yticks([total_cma_tests])
ax.set_yticklabels([str(total_cma_tests)], fontsize=13)
ax.set_xlabel("Number of Batch Instances with same number of Original Test", fontsize=15)
ax.set_ylabel("Total Number of CMA Tests", fontsize=15)
ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels, fontsize=13)
ax.tick_params(axis='y', labelsize=13)

ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=2,
    fontsize=13,
    frameon=True
)

#Clean up visuals
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

one_cma_col = cma_cols[16]  
print(f"Analyzing dominant decimal precision for column: {one_cma_col}")


vals = df_masked_pruned[one_cma_col].dropna().values


decimal_counts = []
for val in vals:
    dec = Decimal(str(val)).normalize()
    exponent = dec.as_tuple().exponent
    decimal_places = abs(exponent) if exponent < 0 else 0
    decimal_counts.append(decimal_places)


decimal_df = pd.DataFrame({
    'Value': vals,
    'Decimal Places': decimal_counts
})

#Find the most common number of decimal places
if not decimal_df.empty:
    most_common_decimals = decimal_df['Decimal Places'].mode()[0]
else:
    most_common_decimals = None

#Bulletproofing: Enforce minimum 1 decimal place
if most_common_decimals == 0:
    print("Detected 0 decimal places - enforcing minimum precision of 1 decimal place.")
    most_common_decimals = 1

# Output
print(decimal_df.head(10))  
print(f"Most common decimal places across ALL observations = {most_common_decimals}")


def analyze_and_round_column(df, column_name):
    print(f"Working on column: {column_name}")
    
    
    vals = df[column_name].dropna().values

    
    decimal_counts = []
    for val in vals:
        dec = Decimal(str(val)).normalize()
        exponent = dec.as_tuple().exponent
        decimal_places = abs(exponent) if exponent < 0 else 0
        decimal_counts.append(decimal_places)

    
    if decimal_counts:
        most_common_decimals = pd.Series(decimal_counts).mode()[0]
    else:
        most_common_decimals = None

    
    if most_common_decimals == 0:
        print(f"{column_name}: Detected 0 decimal places, enforcing minimum 1 decimal place.")
        most_common_decimals = 1

    
    quantize_format = f'1.{"0" * most_common_decimals}'

    print(f"{column_name}: Final precision = {most_common_decimals} decimal places.")

    #Define rounding function
    def round_and_format(val):
        if pd.isnull(val):
            return val
        dec_val = Decimal(str(val)).quantize(Decimal(quantize_format), rounding=ROUND_HALF_UP)
        return dec_val

    #Apply rounding and formatting
    df[column_name] = df[column_name].apply(round_and_format)

    print(f"{column_name}: All values now rounded/formatted to {most_common_decimals} decimal places.\n")

    return most_common_decimals


# list to collect summary results
precision_summary = []


for cma_col in cma_cols:
    final_precision = analyze_and_round_column(df_masked_pruned, cma_col)
    
    # Save into summary
    precision_summary.append({
        'CMA Test Name': cma_col,
        'Final Decimal Precision': final_precision
    })

# summary DataFrame
precision_summary_df = pd.DataFrame(precision_summary)

# final precision summary
print("Rounding Logic Accounted")
print("Final Decimal Precision Summary across all CMA tests:")
print(precision_summary_df)


# Final float conversion before modeling
for col in cma_cols:
    df_masked_pruned[col] = df_masked_pruned[col].astype(float)

# #Dropping count columns
df_masked_pruned.drop(columns=["Original Tests", "Imputed Tests"], inplace=True)
print(df_masked_pruned.head(5))

# # # Save masked version
# df_masked_pruned.to_parquet('.parquet', index=False)
# print("masked dataset saved successfully!")

# Save unmasked version
df_masked_pruned.to_parquet('.parquet', index=False)
print("unmasked dataset saved successfully!")

