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
