import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from joblib import dump

# --- Configuration ---
# Optimal K should be determined via the Elbow Method on real data.
OPTIMAL_K = 3 
# Features used for clustering (based on your Property entity)
NUMERICAL_FEATURES = ['rentPerMonth', 'Total_Rooms', 'SqM', 'latitude', 'longitude']
CATEGORICAL_FEATURES = ['propertyType', 'rentalType']
RANKING_FEATURES = [] # Removed rating

# All features that form the final vector for clustering and ranking
FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + RANKING_FEATURES
# --------------------

def prepare_data(df):
    """
    Sets up the data processing pipeline, transforms data, and saves the preprocessor.
    """
    print("Starting data preparation and feature scaling...")
    
    # 1. Select the relevant features
    X = df[FEATURE_COLUMNS]
    
    # 2. Create a preprocessing pipeline
    # Scaler for numerical data, One-Hot Encoder for categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES),
            # 'rating' is a numerical feature but sometimes it's better to keep it unscaled or just pass it through 
            # if the target scale is already reasonable (e.g., 0-5). Here we include it in numerical scaling.
        ],
        remainder='passthrough'
    )
    
    # 3. Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    
    # 4. Save the preprocessor (scaler and encoder) for use on live user input
    dump(preprocessor, 'models/preprocessor.joblib')
    print("Preprocessor (Scaler + Encoder) saved to models/preprocessor.joblib")
    
    return X_processed, df['Property_ID'].values

def train_and_save_kmeans(X_features, property_ids):
    """
    Trains the K-Means model and saves the model and the feature matrix.
    """
    print(f"Training K-Means with K={OPTIMAL_K}...")
    
    # Initialize and train K-Means model
    kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init='auto')
    kmeans.fit(X_features)
    
    # 1. Get cluster labels for each property
    cluster_labels = kmeans.labels_
    
    # 2. Save the trained K-Means model
    dump(kmeans, 'models/k_means_model.joblib')
    print("K-Means model saved to models/k_means_model.joblib")
    
    # 3. Create the crucial feature matrix for Stage 2 (Cosine Similarity)
    # Matrix format: [Property ID, Cluster ID, Feature Vector...]
    property_ids_reshaped = property_ids.reshape(-1, 1)
    cluster_labels_reshaped = cluster_labels.reshape(-1, 1)
    
    # Combine all components (ID, Cluster Label, Features)
    property_matrix = np.column_stack((property_ids_reshaped, cluster_labels_reshaped, X_features))
    
    # Save the matrix for fast loading by the FastAPI service
    np.save('data/property_feature_matrix.npy', property_matrix)
    print("Property feature matrix and cluster IDs saved to data/property_feature_matrix.npy")
    print(f"Matrix shape: {property_matrix.shape}")


if __name__ == '__main__':
    # Adjust the path to the CSV based on where you run the script
    try:
        data_path = 'data/properties_data.csv'
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find CSV file at {data_path}. Ensure it exists.")
        exit()

    # Create the 'models' directory if it doesn't exist
    import os
    if not os.path.exists('models'):
        os.makedirs('models')
        
    X_features, property_ids = prepare_data(df)
    train_and_save_kmeans(X_features, property_ids)

    print("\n--- Training Complete ---")
    print("Run 'uvicorn app.main:app --reload' to start the FastAPI service.")