# Import required libraries for data manipulation, visualization, and statistical analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Set seaborn theme for consistent plot styling (white background with grid, deep color palette)
# Configure default figure size and font size for readability
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 6)

# Load the real estate dataset, disabling low-memory mode to handle mixed data types
df = pd.read_csv("C:\\Users\\ASUS\\Downloads\\Real_Estate_Sales_2001-2022_GL.csv", low_memory=False)

# Display the first few rows and missing value counts to understand data structure
print(df.head())
print(df.isnull().sum())

# Convert 'Date Recorded' to datetime and extract year and month for temporal analysis
df["Date Recorded"] = pd.to_datetime(df["Date Recorded"], errors='coerce')
df["Year Recorded"] = df["Date Recorded"].dt.year
df["Month Recorded"] = df["Date Recorded"].dt.month

# Fill missing values in categorical and numeric columns to prevent errors in analysis
# Use 'Unknown' for categorical fields and median for numeric fields to preserve distribution
df["Town"] = df["Town"].fillna("Unknown")
df["Property Type"] = df["Property Type"].fillna("Unknown")
df["Residential Type"] = df["Residential Type"].fillna("Unknown")
df["Sale Amount"] = df["Sale Amount"].fillna(df["Sale Amount"].median())
df["Assessed Value"] = df["Assessed Value"].fillna(df["Assessed Value"].median())
df["Sales Ratio"] = df["Sales Ratio"].replace([np.inf, -np.inf], 0).fillna(0)
df["Address"] = df["Address"].fillna("Unknown")
df["Non Use Code"] = df["Non Use Code"].fillna("Unknown")
df["Assessor Remarks"] = df["Assessor Remarks"].fillna("Unknown")
df["OPM remarks"] = df["OPM remarks"].fillna("Unknown")
df["Location"] = df["Location"].fillna("Unknown")

# Remove outliers in 'Sale Amount' and 'Assessed Value' using Interquartile Range (IQR) method
# This reduces the impact of extreme values on visualizations and analysis
for col in ["Sale Amount", "Assessed Value"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

# Create a random sample of 5000 rows to improve performance in visualizations
df_sample = df.sample(n=5000, random_state=42)

# Plot 1: Annual sales trend with a 3-year rolling average
# Group sales by year and calculate rolling average to smooth trends
sales_year = df_sample.groupby("Year Recorded").size()
sales_year_roll = sales_year.rolling(window=3, min_periods=1).mean()
plt.figure(figsize=(10, 6))
sns.lineplot(x=sales_year.index, y=sales_year.values, label="Annual Sales", color="skyblue")
sns.lineplot(x=sales_year_roll.index, y=sales_year_roll.values, label="3-Year Rolling Average", color="orange", linestyle="--", marker='o')
plt.title("Annual Real Estate Sales Trend (2001-2022)")
plt.xlabel("Year")
plt.ylabel("Number of Sales")
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.show()

# Plot 2: Top 10 towns by number of sales
# Count sales per town and visualize the top 10 using a bar plot
top_towns_counts = df_sample["Town"].value_counts().head(10).reset_index()
top_towns_counts.columns = ["Town", "Count"]
plt.figure(figsize=(10, 6))
sns.barplot(data=top_towns_counts, x="Town", y="Count", hue="Town", palette="viridis", legend=False)
plt.title("Top 10 Towns by Number of Sales")
plt.xlabel("Town")
plt.ylabel("Number of Sales")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Plot 3: Distribution of log-transformed sale amounts
# Use log transformation to handle skewed data and highlight distribution shape
plt.figure(figsize=(10, 6))
sns.histplot(np.log1p(df_sample["Sale Amount"]), bins=30, kde=True, color="lightcoral")
mean_log_sale = np.log1p(df_sample["Sale Amount"]).mean()
plt.axvline(mean_log_sale, color="red", linestyle="--", label=f"Mean Log(Sale): {mean_log_sale:.2f}")
plt.title("Distribution of Log-Transformed Sale Amount")
plt.xlabel("Log(Sale Amount + 1)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# Plot 4: Assessed Value vs. Sale Amount with regression line
# Scatter plot with color-coded property types and a regression line to show relationship
plt.figure(figsize=(10, 6))
sns.scatterplot(x="Assessed Value", y="Sale Amount", data=df_sample, alpha=0.3, hue="Property Type", size=10)
slope, intercept, r_value, p_value, std_err = stats.linregress(df_sample["Assessed Value"], df_sample["Sale Amount"])
plt.plot(df_sample["Assessed Value"], slope * df_sample["Assessed Value"] + intercept, color="black", linestyle="--", label=f"Regression Line (R² = {r_value**2:.2f})")
plt.title("Assessed Value vs. Sale Amount")
plt.xlabel("Assessed Value")
plt.ylabel("Sale Amount")
plt.legend(title="Property Type", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# Plot 5: Sales Ratio distribution over time by property type
# Box plot to compare sales ratio trends across years and property types
plt.figure(figsize=(12, 7))
sns.boxplot(x="Year Recorded", y="Sales Ratio", hue="Property Type", data=df_sample, palette="Set2")
plt.title("Sales Ratio Distribution Over Time by Property Type")
plt.xlabel("Year Recorded")
plt.ylabel("Sales Ratio")
plt.xticks(rotation=45, ha="right")
plt.ylim(0, df_sample["Sales Ratio"].quantile(0.95))
plt.legend(title="Property Type", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# Plot 6: Distribution of property types
# Bar plot to show the frequency of different property types
prop_counts = df_sample["Property Type"].value_counts().reset_index()
prop_counts.columns = ["Property Type", "Count"]
plt.figure(figsize=(10, 6))
sns.barplot(data=prop_counts, x="Property Type", y="Count", hue="Property Type", palette="magma", legend=False)
plt.title("Distribution of Property Types")
plt.xlabel("Property Type")
plt.ylabel("Number of Sales")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Plot 7: Top 10 Non Use Codes
# Bar plot to visualize the most common non-use codes in the dataset
non_use_counts = df_sample["Non Use Code"].value_counts().head(10).reset_index()
non_use_counts.columns = ["Non Use Code", "Count"]
plt.figure(figsize=(10, 6))
sns.barplot(data=non_use_counts, x="Non Use Code", y="Count", hue="Non Use Code", palette="Set3", legend=False)
plt.title("Top 10 Non Use Codes")
plt.xlabel("Non Use Code")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Plot 8: Correlation matrix of numeric features
# Heatmap to explore relationships between numeric columns
numeric_cols = df_sample[["Sale Amount", "Assessed Value", "Sales Ratio", "Year Recorded"]]
plt.figure(figsize=(8, 7))
correlation_matrix = numeric_cols.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'})
plt.title("Correlation Matrix of Numeric Features")
plt.tight_layout()
plt.show()

# Plot 9: Kernel Density Estimate of Sale Amount
# KDE plot to show the smoothed distribution of sale amounts
plt.figure(figsize=(10, 6))
sns.kdeplot(df_sample["Sale Amount"], fill=True, color='skyblue')
mean_sale = df_sample["Sale Amount"].mean()
plt.axvline(mean_sale, color="red", linestyle="--", label=f"Mean: ${mean_sale:,.0f}")
plt.title("Kernel Density Estimate of Sale Amount")
plt.xlabel("Sale Amount")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

# Plot 10: Kernel Density Estimate of Log-Transformed Sale Amount
# KDE plot for log-transformed sale amounts to handle skewness
plt.figure(figsize=(10, 6))
sns.kdeplot(np.log1p(df_sample["Sale Amount"]), fill=True, color='lightgreen')
mean_log_sale = np.log1p(df_sample["Sale Amount"]).mean()
plt.axvline(mean_log_sale, color="red", linestyle="--", label=f"Mean Log(Sale): {mean_log_sale:.2f}")
plt.title("Kernel Density Estimate of Log-Transformed Sale Amount")
plt.xlabel("Log(Sale Amount + 1)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()
