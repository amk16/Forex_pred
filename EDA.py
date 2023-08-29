import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('your_dataset.csv')

# Display the first few rows
print(df.head())

# Summary of data types and non-null counts
print(df.info())

# Summary statistics for numerical columns
print(df.describe())

#UNIVARIATE ANALYSIS

#1. Numerical Data
df['column_name'].hist()
plt.show()

#2. Boxplots
sns.boxplot(x='column_name', data=df)
plt.show()

#3. Bar Charts
sns.countplot(x='categorical_column', data=df)
plt.show()

