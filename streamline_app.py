import streamlit as st
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the trained PyTorch LSTM model
@st.cache_resource
def load_model():
    try:
        model_path = 'model.pkl'
        model = torch.load(model_path)
        model.eval()  # Set the model to evaluation mode
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'model.pkl' is in the correct directory.")
        st.stop()

# Function to classify transactions using the LSTM model
def classify_fraudulent_transactions(new_data, model):
    """
    Classify transactions as fraudulent using the trained anomaly detection model.
    
    Parameters:
    - new_data (pd.DataFrame): DataFrame containing the new credit card transactions.
    - model: Trained PyTorch LSTM model.
    
    Returns:
    - DataFrame of transactions classified as fraudulent.
    """
    # Preprocess the data and convert it to a tensor
    input_data = torch.tensor(new_data.values).float()
    
    # Perform prediction with the model
    with torch.no_grad():
        predictions = model(input_data)
    
    # Assuming a threshold of 0.5 for anomaly detection
    predicted_labels = (predictions >= 0.5).int().numpy()

    # Return transactions that are classified as fraudulent
    fraudulent_transactions = new_data[predicted_labels == 1]
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

        # Load the trained model
        model = load_model()

        # Process and classify transactions
        fraudulent_transactions = classify_fraudulent_transactions(data, model)

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

