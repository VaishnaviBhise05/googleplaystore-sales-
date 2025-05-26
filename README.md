# googleplaystore-sales-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

df = pd.read_csv('googleplaystore.csv')
df

print(df.shape)
df.head()
df.info()
df.describe()

if 10472 in df.index:
    df.drop(10472, inplace=True)
    print("Dropped row 10472 due to known issue.")
else:
    print("Row 10472 not found. Proceeding without drop.")

df = df[df['Rating'] <= 5]
def convert_size(x):
    if 'M' in x:
        return float(x.replace('M','')) * 1024 * 1024
    elif 'k' in x:
        return float(x.replace('k','')) * 1024
    elif x == 'Varies with device':
        return np.nan
    return x

df['Size_in_bytes'] = df['Size'].apply(convert_size)
df['Size_MB'] = df['Size_in_bytes'] / (1024 * 1024)



print("Original number of rows:", df.shape[0])

if 10472 in df.index:
    df.drop(10472, inplace=True)
    print("Dropped row 10472 due to known issue.")
else:
    print("Row 10472 not found. Proceeding without drop.")

invalid_ratings = df[df['Rating'] > 5]
print("Number of rows with invalid ratings (>5):", invalid_ratings.shape[0])

df = df[df['Rating'] <= 5]

print("Final number of rows after cleaning:", df.shape[0])


missing_before = df['Rating'].isnull().sum()
print(f"Missing 'Rating' values before filling: {missing_before}")

group_means = df.groupby('Installs_category')['Rating'].mean()

for category, value in group_means.items():
    mask = (df['Installs_category'] == category) & (df['Rating'].isnull())
    count = mask.sum()
    df.loc[mask, 'Rating'] = value
    print(f"Filled {count} missing ratings in '{category}' category with value {value:.2f}")

missing_after = df['Rating'].isnull().sum()
print(f"Missing 'Rating' values after filling: {missing_after}")

df.info()
df.isnull().sum()
df_cleaned = df.dropna(subset=['Rating'])
df_cleaned.isnull().sum()


df_cleaned = df.dropna(subset=['Rating'])

df_cleaned = df_cleaned[df_cleaned['Installs'].str.match(r'^[\d,]+$', na=False)]
df_cleaned['Installs'] = df_cleaned['Installs'].str.replace(',', '').astype(int)

df_cleaned = df_cleaned[df_cleaned['Price'] != 'Everyone']

df_cleaned['Price'] = df_cleaned['Price'].str.replace('$', '', regex=False)
df_cleaned['Price'] = pd.to_numeric(df_cleaned['Price'], errors='coerce')

df_cleaned = df_cleaned[df_cleaned['Reviews'].str.isnumeric()]
df_cleaned['Reviews'] = df_cleaned['Reviews'].astype(int)

df_cleaned = df_cleaned.dropna(subset=['Price'])


df_cleaned.head()


top_installed = df_cleaned.sort_values(by='Installs', ascending=False)[['App', 'Installs']].head()
print(top_installed)

avg_rating_per_category = df_cleaned.groupby('Category')['Rating'].mean().sort_values(ascending=False)
print(avg_rating_per_category.head())


plt.figure(figsize=(12, 6))
top_categories = df_cleaned['Category'].value_counts().head(10)
sns.barplot(x=top_categories.index, y=top_categories.values, palette="viridis")
plt.title("Top 10 Categories by Number of Apps", fontsize=16)
plt.xlabel("Category")
plt.ylabel("Number of Apps")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
avg_rating_top_categories = df_cleaned.groupby('Category')['Rating'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=avg_rating_top_categories.index, y=avg_rating_top_categories.values, palette="magma")
plt.title("Average Rating per Top 10 Categories", fontsize=16)
plt.xlabel("Category")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.ylim(0, 5)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(df_cleaned['Rating'], bins=30, kde=True, color='skyblue')
plt.title("Distribution of App Ratings", fontsize=16)
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


top_reviewed_apps = df_cleaned.sort_values(by='Reviews', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_reviewed_apps['App'], y=top_reviewed_apps['Reviews'], palette='plasma')
plt.title('Top 10 Most Reviewed Apps')
plt.xticks(rotation=45, ha='right')
plt.xlabel('App')
plt.ylabel('Number of Reviews')
plt.tight_layout()
plt.show()


paid_apps = df_cleaned[df_cleaned['Price'] > 0]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=paid_apps, x='Price', y='Rating', alpha=0.7, color='green')
plt.title('Price vs Rating (Paid Apps)')
plt.xlabel('Price ($)')
plt.ylabel('Rating')
plt.tight_layout()
plt.show()

print(df.describe())
print(df['Category'].value_counts())

import matplotlib.pyplot as plt
top_categories = df['Category'].value_counts().head(10)
plt.figure(figsize=(10, 6))
top_categories.plot(kind='bar', color='skyblue')
plt.title('Top 10 Categories by Number of Apps')
plt.xlabel('Category')
plt.ylabel('Number of Apps')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
df['Rating'].dropna().plot(kind='hist', bins=20, color='orange', edgecolor='black')
plt.title('Distribution of App Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

df['Type'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
plt.title('Distribution of Free vs Paid Apps')
plt.ylabel('')
plt.tight_layout()
plt.show()



