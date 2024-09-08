import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the trained Isolation Forest model
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join('models', 'model.pkl')  # Example if placed in 'models' folder
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found. Please ensure 'model.pkl' is in the correct directory: {model_path}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Function to classify transactions using the Isolation Forest model
def classify_fraudulent_transactions(new_data, model):
    """
    Classify transactions as fraudulent using the trained anomaly detection model.
    
    Parameters:
    - new_data (pd.DataFrame): DataFrame containing the new credit card transactions.
    - model: Trained Isolation Forest model.
    
    Returns:
    - DataFrame of transactions classified as fraudulent.
    """
    # Preprocess the data if needed and perform prediction
    input_data = new_data.select_dtypes(include=['float64', 'int64'])
    
    # Perform prediction with the model
    predictions = model.predict(input_data)
    
    # Assuming -1 indicates anomaly (fraudulent) and 1 indicates normal
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


