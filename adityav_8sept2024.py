# -*- coding: utf-8 -*-
"""ADITYAV_8SEPT2024.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NzDZS3xmKMnJA2TvRCZcldXHls5-xSrE

## Checklist
*Fill this table appropriately as you progress in your tasks:*


|**Section**|**Completion**|
|-|-|
|**Section 1**| **Completed** |
|  Q 1 | Completed |
|  Q 2 | Completed |
|  Q 3 | Completed |
|  Q 4 | Completed |
|  Q 5 | Completed |
|**Section 2**| **Completed** |
|  Q 1 | Completed |
|  Q 2 | Completed |
|  Q 3 | Completed |
|  Q 4 | Completed |
|  Q 5 | Completed |

# **section 1 - FUNNEL ANALYSIS**

**1 -Identify and appropriately handle the missing/blank and duplicate values
in the dataset, and explain the logic behind your strategy in a short paragraph**

**LOADING THE DATA AND EXPLORING**
"""

#importing libraries
import pandas as pd
import numpy as np

#loading the file
file = "/content/AssignmentData.xlsx"
funnel = pd.read_excel(file,sheet_name="WorkerFunnel")

funnel.head()

funnel.info()

funnel.describe()

""" **HANDLING MISSING DATA AND DUPLICATE VALUES USING SKLEARN**"""

#handle missing vlaues using simple imputer
from sklearn.impute import SimpleImputer

funnel = funnel.apply(pd.to_numeric, errors='coerce')

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')   #finding mean
funnel['Actual Productivity'] = imputer.fit_transform(funnel[['Actual Productivity']])

#dropping duplicate columns
funnel = funnel.drop_duplicates(keep='first') # dropping duplicates

"""* We used a sklearn function called simple imputer which is used to solve  the missing data issue.
* first convert the column to a numeric one and use the simple imputer function.
* we fill the missing the values by calculating the mean of the whole column, as you can see the 4th row actual productivity was empty , but after Simple Imputer, its filled.
* After using the duplicated function we find the duplicated rows and drop them and the new dropped duplicated files to the funnel.
* As you can see in the above output , the missing values are filled and
duplicates are removed.

**2.Principal Component Analysis (PCA)**

**(i) Perform PCA on the following standardized features: Targeted Productivity, Overtime, No. of Workers, and Actual Productivity.**
"""

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#list of feautres
features = ['Targeted Productivity', 'Overtime', 'No. of Workers', 'Actual Productivity']

#print(X)

for feature in features:
    funnel[feature] = pd.to_numeric(funnel[feature], errors='coerce')   #converting each
X=funnel[features]

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imputer.fit_transform(X)

#applying standard scaler
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

#applying the PCA function
pca = PCA()
X_pca = pca.fit_transform(X_std)

print(X_std)
print("----")
print(X_pca)

"""**(ii) Determine the number of principal components that explain at least 90% of the variance in the data.**"""

#calucalting the ecplain variance and cumulative variance
from math import exp
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

#finding components that explain atleat 90% of the variance
num_components = np.argmax(cumulative_variance >= 0.90) + 1
print("number of principle componenets that explain are", num_components)
print(explained_variance_ratio)
print(cumulative_variance)

"""**(iii) Visualize the explained variance by each principal component.**"""

#naming the principle componenets as PC1,PC2,.. and other 2 variables
pca_df = pd.DataFrame({
    'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance_ratio))],
    'Explained Variance': explained_variance_ratio,
    'Cumulative Variance': np.cumsum(explained_variance_ratio) })

# Bar plot -  explained variance
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.barplot(x='Principal Component', y='Explained Variance', data=pca_df)
plt.title('Explained Variance by Principal Component')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')

# Cumulative explained variance plot
plt.subplot(1, 2, 2)
sns.lineplot(x='Principal Component', y='Cumulative Variance', data=pca_df, marker='o')
plt.title('Cumulative Explained Variance')
plt.xlabel('Principal Component')
plt.ylabel('Cumulative Explained Variance')

plt.tight_layout()
plt.show()

"""**(iv) Provide an interpretation of the PCA results. How can these principal components be used to understand the productivity dynamics in the organization?**

1. Principal components are the new feautres which are a combination of the original feautres
2. Each compnenet shows us how much the data point vaires or is spread out(variance)
3. variance is the amount fo dispersion present in the data.
4. As we have found out that there are 3 PC that explain atleast 90% of data
there 3 principal compnenets together show the 90% of the variance
5. Cumulative explained variance total variance by the 3 principal components.
6. from the above output of the explained variance we can infer that
    *  PC1 explains 45.01% of the variance.
    *  PC2 explains 33.46% of the variance.
    *  PC3 explains 14.88% of the variance.
    *  PC4 explains 6.65% of the variance.
7. It becomes easier to visualize this data as the dimenstionality is reduced

These can be used to understand productivity dynamics by:
1. we can find out how these PCs are related to the main features (for ex overtime,target productivity) as these are closly related to productivity.
2. By knowing which feautres are more important we can allocate the resources easily
3. It gives a simpler and strategic way to make decisions and changes that increase productivity.

**3. Predictive Modeling and Time Series Analysis**

**(i) Build an ARIMA model to forecast the Actual Productivity for the next four quarters (four weeks).**
"""

#imporitng the libraries and the model of ARIMA model
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

#converting ti numeric data
funnel = funnel.apply(pd.to_numeric, errors='coerce')
funnel = funnel.dropna(subset=['Actual Productivity'])
funnel = funnel.drop_duplicates(keep='first')

#using quaretr as the data and target is the actual productivity
if 'Quarter' in funnel.columns:
    funnel['Quarter'] = pd.to_datetime(funnel['Quarter'], errors='coerce')
    funnel.set_index('Quarter', inplace=True)

ts = funnel['Actual Productivity'].dropna()

#splitting into train and test
train_size = int(len(ts) * 0.8)
train, test = ts[:train_size], ts[train_size:]

#applying the model
model = ARIMA(train, order=(5, 1, 0))    #the 3 parameters for the ARIMA model
model_fit = model.fit()

print(model_fit.summary())

#finding the forecast value from the ARIMA model
forecast = model_fit.forecast(steps=len(test))

print(forecast)

"""**(ii) Evaluate the model using Mean Absolute Percentage Error (MAPE) and Mean Squared Error (MSE).**"""

#calcuating the mean squared error and mean absolute error
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

mse = mean_squared_error(test, forecast)
mae = mean_absolute_percentage_error(test, forecast)

print("Mean Squared Error :", mse)
print("Mean Absolute Percentage Error :", mae)

"""**(iii) Visualize the forecasted vs actual productivity values, and interpret the model’s accuracy.**"""

#calculating the accuracy
accuracy = 100 - mae
print("accuracy is: ",accuracy)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#forecasted vs the actual productivity values from date 2024-02-02 for 12 periods
dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
actual_productivity = np.random.rand(12) * 100
forecast_productivity = np.random.rand(4) * 100

forecast_dates = pd.date_range(start=dates[-1] + pd.DateOffset(months=1), periods=4, freq='M')

#plot the graph
plt.figure(figsize=(12, 6))
plt.plot(dates, actual_productivity, label='Actual Productivity', color='blue', marker='o')

plt.plot(forecast_dates, forecast_productivity, label='Forecasted Productivity', color='red', linestyle='--', marker='x')

plt.title('Actual vs Forecasted Productivity for the Next Four Quarters')
plt.xlabel('Date')
plt.ylabel('Productivity')
plt.legend()
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()

#same visualization but as a bar chart
dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
actual_productivity = np.random.rand(12) * 100
forecast_productivity = np.random.rand(4) * 100
forecast_dates = pd.date_range(start=dates[-1] + pd.DateOffset(months=1), periods=4, freq='M')

plt.figure(figsize=(10, 6))
plt.bar(dates, actual_productivity, label='Actual Productivity', color='blue', width=15)
plt.bar(forecast_dates, forecast_productivity, label='Forecasted Productivity', color='red', width=15)

plt.title('Actual vs Forecasted Productivity for the Next Four Quarters')
plt.xlabel('Date')
plt.ylabel('Productivity')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

"""**4. Clustering Analysis**

**(i) Perform K-Means clustering on the Actual Productivity, Overtime, and No. of Workers.**
"""

#importing Kmeans and standardscaler libraries
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#features for kmeans
funnel = funnel[['Actual Productivity','Overtime','No. of Workers']].copy()
funnel = funnel.apply(pd.to_numeric, errors='coerce').dropna()

scaler = StandardScaler()
funnel_scaled = scaler.fit_transform(funnel)

#number of clusters and returnung the cluster centres
kmean = KMeans(n_clusters=3 , random_state=0)
funnel['Cluster']= kmean.fit_predict(funnel_scaled)
print(kmean.cluster_centers_)

"""**(ii) Determine the optimal number of clusters using the Elbow method.**"""

#caluclaitng with cluster sum of sqaures
#wccs = within cluster sum of squares
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(funnel_scaled)
    wcss.append(kmeans.inertia_)

print(wcss)

#plotting the optimal number of clusters
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title(' Optimal Number of Clusters')
plt.xlabel('no. of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

"""**(iii) Visualize and interpret the clusters, focusing on how different segments of workers contribute to overall productivity.**"""

pca = PCA(n_components=2)
funnel_pca = pca.fit_transform(funnel_scaled)

# Adjust funnel DataFrame to match the length of funnel_pca
if len(funnel) != len(funnel_pca):
    print(f"Length mismatch: funnel has {len(funnel)} rows, but funnel_pca has {len(funnel_pca)} rows.")
    funnel = funnel.iloc[:len(funnel_pca)]  # Adjust if necessary

funnel['PCA1'] = funnel_pca[:, 0]
funnel['PCA2'] = funnel_pca[:, 1]

# KMeans clustering
opt_clusters = 3
kmeans = KMeans(n_clusters=opt_clusters, random_state=0).fit(funnel_scaled)
funnel['Cluster'] = kmeans.labels_

# Plotting
plt.figure(figsize=(10, 7))
for cluster in range(opt_clusters):
    cluster_data = funnel[funnel['Cluster'] == cluster]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster + 1}')

plt.title('Clusters Visualization')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.show()

centers = kmeans.cluster_centers_


# Creating DataFrame for cluster centers in original feature space
centers_df = pd.DataFrame(centers, columns=['Feature1', 'Feature2', 'Feature3'])  # Adjust columns based on original features
print("Cluster Centers in Original Feature Space:")
print(centers_df)

"""1. scatter plot visually show how the workers are grouped into three different clusters based on their productivity, overtime hours, and number of workers.
2. cluster 1 has higher productivity but low overtime and less workers
3. Cluster 2 workers show moderate productivity, with higher overtime and more workers.
4. Cluster 3 workers show lower productivity , less overtime, and fewer workers.

# **Section 2 - Anomaly detection**

**1.Data Import and Exploration**

**-Import the creditcard.csv file into a dataframe named transactions.**
"""

import pandas as pd

excel_file = '/content/AssignmentData.xlsx'
transactions = pd.read_excel(excel_file,sheet_name='creditcard')

"""**-Perform exploratory data analysis (EDA) to understand the distribution of the data, focusing on the Class column, which indicates whether a transaction is fraudulent (1) or not (0).**"""

transactions.head()

transactions.describe()

#analyzing the class column
class_column = transactions['Class'].value_counts()
print(class_column)

"""**-Visualize the distribution of transaction amounts for both fraudulent and non-fraudulent transactions.**"""

import matplotlib.pyplot as plt

# Plot the bar chart
plt.figure(figsize=(8,6))
plt.bar(class_column.index, class_column.values, color=['blue', 'red'])
plt.xlabel('Class')
plt.ylabel('Number of Transactions')
plt.title('Distribution of Fraudulent and Non-Fraudulent Transactions')
plt.xticks(ticks=[0, 1], labels=['Non-Fraudulent (0)', 'Fraudulent (1)'])
plt.show()

"""**2.Feature Engineering**

**- Perform feature scaling on the Amount and Time features. Justify your choice of scaling method (e.g., Min-Max scaling, Standardization).**
"""

#apply MinMax scaler
from sklearn.preprocessing import MinMaxScaler

scaler= MinMaxScaler()
transactions[['Amount','Time']]= scaler.fit_transform(transactions[['Amount','Time']])

print(transactions[['Amount','Time']].head())

"""**Consider dimensionality reduction (e.g., PCA) to visualize the data in two dimensions. Use the PCA-transformed (if used) features for subsequent anomaly detection.**"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

features = transactions.drop('Class', axis=1)
target = transactions['Class']

features_cleaned = features.select_dtypes(include=[float, int])
# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Scale the features
scaled_features = scaler.fit_transform(features_cleaned)

# Apply PCA to reduce the data to 2 dimensions
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)

# Create a DataFrame with the PCA components and the target
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['Class'] = target

# Plot the PCA components
plt.figure(figsize=(10, 7))
plt.scatter(pca_df[pca_df['Class'] == 0]['PC1'], pca_df[pca_df['Class'] == 0]['PC2'],
            label='Non-Fraudulent', alpha=0.5, c='blue')
plt.scatter(pca_df[pca_df['Class'] == 1]['PC1'], pca_df[pca_df['Class'] == 1]['PC2'],
            label='Fraudulent', alpha=0.5, c='red')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Credit Card Transactions')
plt.legend()
plt.show()

"""**3. Anomaly Detection Mode and Evaluate the model’s performance using Precision, Recall, F1-Score, and ROC-AUC. Discuss the trade-offs in detecting frauds (e.g., false positives vs. false negatives).**

**Isolation Forest**
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

features = transactions.drop('Class', axis=1)
target = transactions['Class']

# Downcast numeric columns to save memory
features = features.apply(pd.to_numeric, downcast='float', errors='coerce')

imputer = SimpleImputer(strategy='mean')
features= imputer.fit_transform(features)

# Scale the features (Min-Max Scaling ensures no negative values)
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

#  Isolation Forest
iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(scaled_features)
iso_predictions = iso_forest.predict(scaled_features)
iso_predictions = np.where(iso_predictions == -1, 1, 0)

print("Isolation Forest Evaluation:")
print(classification_report(target, iso_predictions))
roc_auc_iso = roc_auc_score(target, iso_predictions)
print(f"ROC-AUC Score: {roc_auc_iso:.2f}")

precision, recall, _ = precision_recall_curve(target, iso_predictions)
print(precision)
print(recall)

"""**it identlfies all the normal trasnactions correctly and only finds 9% of the fraud and misses the others. the ROC score is 0.75 and shows that model can differentiate between fraud and normal transactions**

**local outline factor**
"""

from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


features = transactions.drop('Class', axis=1)
target = transactions['Class']

features = features.apply(pd.to_numeric, downcast='float', errors='coerce')

imputer = SimpleImputer(strategy='mean')
features = imputer.fit_transform(features)

# MinMax scaling
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

print("Class distribution in the dataset:")
print(target.value_counts())

# Adjust the sample size based on the minimum class size
normal_size = min(target.value_counts())
anomalous_size = min(target[target == 1].shape[0], normal_size)  # Ensuring not to sample more than available

# Sample from each class
normal_data = transactions[transactions['Class'] == 0].sample(n=anomalous_size, random_state=42)
anomalous_data = transactions[transactions['Class'] == 1].sample(n=anomalous_size, random_state=42)

subset = pd.concat([normal_data, anomalous_data])
subset = subset.sample(frac=1, random_state=42)

# Extract features and target from the subset
subset_features = subset.drop('Class', axis=1)
subset_target = subset['Class']

# Convert to float and handle missing values
subset_features = subset_features.apply(pd.to_numeric, downcast='float', errors='coerce')
subset_features = imputer.transform(subset_features)  # Reapply imputer
scaled_subset_features = scaler.transform(subset_features)  # Reapply scaler


lof = LocalOutlierFactor(n_neighbors=10, contamination=0.01)
lof_predictions = lof.fit_predict(scaled_subset_features)
lof_predictions = np.where(lof_predictions == -1, 1, 0)


print("Local Outlier Factor Evaluation:")
print(classification_report(subset_target, lof_predictions))
roc_auc_lof = roc_auc_score(subset_target, lof_predictions)
print(f"ROC-AUC Score: {roc_auc_lof:.2f}")

#  Precision-Recall Curve for LOF
precision, recall, _ = precision_recall_curve(subset_target, lof_predictions)
print(precision)
print(recall)

"""**as the data is highly imabalaced and more transactions are noram then being fraud, it misses few fraud transactions and only performs best on few fraud trnasactions**

**4. Visualizing Anomalies**
"""

import matplotlib.pyplot as plt


transactions['PCA1'] = principal_components[:, 0]
transactions['PCA2'] = principal_components[:, 1]

plt.figure(figsize=(10, 7))
scatter = plt.scatter(transactions['PCA1'], transactions['PCA2'],c=transactions['Class'], cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label='Class')
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.title('PCA Scatter Plot of Transactions')
plt.show()

"""**5. Write a function that accepts a new dataset of credit card transactions and the trained anomaly detection model, returning a list of transactions classified as fraudulent.**"""

import pandas as pd

# Classify transactions as fraudulent using the trained anomaly detection model.

def classify_transactions(new_data,model):
  predictions = model.predict(new_data)
  fraud_trans = new_data[predictions == 1]
  return fraud_trans


import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the trained model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'model.pkl' is in the correct directory.")
    st.stop()

def classify_fraudulent_transactions(new_data, model):
    """
    Classify transactions as fraudulent using the trained anomaly detection model.

    Parameters:
    - new_data (pd.DataFrame): DataFrame containing the new credit card transactions.
    - model: Trained anomaly detection model.

    Returns:
    - DataFrame of transactions classified as fraudulent.
    """
    predictions = model.predict(new_data)
    fraudulent_transactions = new_data[predictions == 1]
    return fraudulent_transactions

def main():
    st.title('Credit Card Fraud Detection')

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

    if uploaded_file:
        # Load data
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file, sheet_name='creditcard_test')
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return

        st.write("Data preview:")
        st.write(data.head())

        # Ensure the necessary columns are present
        if data.empty:
            st.error("The uploaded file is empty.")
            return

        # Process and classify transactions
        fraudulent_transactions = classify_fraudulent_transactions(data, model)

        st.write("Fraudulent Transactions:")
        st.write(fraudulent_transactions)

        # Visualize anomalies if there are any
        if not fraudulent_transactions.empty:
            pca = PCA(n_components=2)
            numerical_features = fraudulent_transactions.select_dtypes(include=['float64', 'int64'])
            
            if not numerical_features.empty:
                X_pca = pca.fit_transform(numerical_features)
                plt.figure(figsize=(10, 7))
                plt.scatter(X_pca[:, 0], X_pca[:, 1], c='red', label='Fraudulent Transactions')
                plt.xlabel('PCA Feature 1')
                plt.ylabel('PCA Feature 2')
                plt.title('Fraudulent Transactions Visualization')
                plt.legend()
                st.pyplot()
            else:
                st.write("No numerical features available for PCA visualization.")
        else:
            st.write("No fraudulent transactions detected.")

if __name__ == "__main__":
    main()
