import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Define and train the Isolation Forest model
@st.cache_resource
def train_model(data):
    # Preprocess the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.select_dtypes(include=['float64', 'int64']))

    # Train the Isolation Forest model
    model = IsolationForest(contamination=0.01)  # Adjust contamination as needed
    model.fit(scaled_data)
    return model, scaler

# Function to classify transactions using the Isolation Forest model
def classify_fraudulent_transactions(new_data, model, scaler):
    """
    Classify transactions as fraudulent using the trained Isolation Forest model.
    
    Parameters:
    - new_data (pd.DataFrame): DataFrame containing the new credit card transactions.
    - model: Trained Isolation Forest model.
    - scaler: StandardScaler used to preprocess the data.
    
    Returns:
    - DataFrame of transactions classified as fraudulent.
    """
    # Preprocess the data and make predictions
    scaled_data = scaler.transform(new_data.select_dtypes(include=['float64', 'int64']))
    predictions = model.predict(scaled_data)
    
    # Return transactions that are classified as fraudulent
    fraudulent_transactions = new_data[predictions == -1]
    return fraudulent_transactions

def main():
    st.title('Credit Card Fraud Detection')

    # File uploader for the input data
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

    if uploaded_file:
        # Load the data
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

        # Train the model (this should ideally be done separately and the model should be saved and loaded)
        model, scaler = train_model(data)

        # Process and classify transactions
        fraudulent_transactions = classify_fraudulent_transactions(data, model, scaler)

        st.write("Fraudulent Transactions:")
        st.write(fraudulent_transactions)

        # Visualize the anomalies if they exist
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



