from joblib import load
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
from .schemas import UserQuery, Recommendation

# --- Configuration (MUST match training script) ---
# Total number of numerical features used in the ColumnTransformer's 'num' step
NUMERICAL_FEATURES_COUNT = 5 # 'normalized_rent', 'Total_Rooms', 'SqM', 'latitude', 'longitude'
# This list must be ordered exactly as it appears in the UserQuery schema
USER_QUERY_FEATURES = ['target_rent', 'min_total_rooms', 'target_sqft', 'search_latitude', 'search_longitude']
TOP_N = 10 # Number of recommendations to return

# --- Global Variables and Paths ---
K_MEANS_PATH = 'models/k_means_model.joblib'
PREPROCESSOR_PATH = 'models/preprocessor.joblib'
FEATURE_MATRIX_PATH = 'data/property_feature_matrix.npy'

# These will be loaded once when the FastAPI application starts
kmeans_model = None
preprocessor = None
property_matrix = None # Format: [Property ID (str), Cluster ID (float), Feature Vector...]

def load_artifacts():
    """
    Loads the trained K-Means model, scaler/encoder, and property feature matrix into memory.
    This is called once at FastAPI startup.
    """
    global kmeans_model, preprocessor, property_matrix
    try:
        print("Loading ML artifacts...")
        # Load the trained models and artifacts
        kmeans_model = load(K_MEANS_PATH)
        preprocessor = load(PREPROCESSOR_PATH)
        property_matrix = np.load(FEATURE_MATRIX_PATH, allow_pickle=True)
        
        # Verify load success
        print(f"Model loaded. K={kmeans_model.n_clusters}")
        print(f"Property Matrix loaded. Total properties: {property_matrix.shape[0]}")
        return True
    except FileNotFoundError as e:
        print(f"ERROR: Artifact file not found. Have you run 'python scripts/train_model.py'? -> {e}")
        return False
    except Exception as e:
        print(f"ERROR: Failed to load artifacts: {e}")
        return False

def get_recommendations(query: UserQuery) -> List[Recommendation]:
    """
    Executes the two-stage recommendation process.
    Stage 1: Cluster filtering using K-Means.
    Stage 2: Ranking using Cosine Similarity.
    """
    if kmeans_model is None or property_matrix is None:
        return [] # Return empty if models failed to load

    # --- Adjust query based on user lifestyle ---
    adjusted_min_rooms = query.min_total_rooms
    adjusted_sqft = query.target_sqft
    adjusted_property_type = query.preferred_property_type

    # Adjust based on number of people
    if query.number_of_people > 4:
        adjusted_min_rooms = max(adjusted_min_rooms, query.number_of_people - 2)  # Ensure at least rooms for people minus shared
    elif query.number_of_people <= 2:
        adjusted_min_rooms = max(adjusted_min_rooms, 1)  # At least 1 room for small families

    # Adjust sqft based on people (rough estimate: 20-30 sqm per person)
    min_sqft_needed = query.number_of_people * 25
    adjusted_sqft = max(adjusted_sqft, min_sqft_needed)

    # Adjust based on marital status
    if query.is_married:
        adjusted_sqft = max(adjusted_sqft, query.target_sqft + 20)  # Married couples might want more space
        # Optionally adjust property type preference, but keep user's choice

    # --- 1. Prepare User Query Vector ---
    
    # 1.0 Normalize target_rent to monthly equivalent
    normalized_target_rent = (
        query.target_rent if query.preferred_rental_type == 'MONTHLY' else query.target_rent * 30
    )
    
    # 1.1 Convert Pydantic model to a single row DataFrame for the preprocessor
    # Include all features (numerical, categorical placeholders)
    user_data = {
        # Numerical Features (MUST match order in training script: Price, Rooms, SqM, Lat, Lon)
        'normalized_rent': [normalized_target_rent],
        'Total_Rooms': [adjusted_min_rooms],
        'SqM': [adjusted_sqft],
        'latitude': [query.search_latitude],
        'longitude': [query.search_longitude],
        # Categorical Features (must have values to encode)
        'propertyType': [adjusted_property_type],
        'rentalType': [query.preferred_rental_type]
        # Rating removed
    }
    user_df = pd.DataFrame(user_data)

    # 1.2 Transform the user data using the saved preprocessor
    # This scales numerical features and one-hot encodes categorical features
    user_vector = preprocessor.transform(user_df)

    # --- 2. Stage 1: Cluster Filtering (K-Means) ---

    # Predict the cluster ID for the user's request vector
    # Result is an array like [2]
    user_cluster_id = kmeans_model.predict(user_vector)[0]
    
    # Filter the master property matrix to only include properties in the matching cluster
    candidate_properties = property_matrix[property_matrix[:, 1] == user_cluster_id]

    if candidate_properties.shape[0] == 0:
        print(f"No candidates found in cluster {user_cluster_id}. Returning empty list.")
        return []

    # Separate property IDs and the feature vectors for the candidates
    candidate_ids = candidate_properties[:, 0]
    # The feature vectors start from column index 2 (after ID and Cluster ID)
    candidate_vectors = candidate_properties[:, 2:].astype(float)
    
    # --- 3. Stage 2: Ranking (Cosine Similarity) ---
    
    # 3.1 Calculate similarity between the user vector and all candidate vectors
    # user_vector is (1, N_features), candidate_vectors is (N_candidates, N_features)
    similarity_scores = cosine_similarity(user_vector, candidate_vectors)[0]

    # 3.2 Combine IDs and scores, then sort
    recommendation_results = list(zip(candidate_ids, similarity_scores))
    recommendation_results.sort(key=lambda x: x[1], reverse=True)

    # --- 4. Final Output ---
    
    # Map the top N results to the final Pydantic schema
    final_recommendations = [
        Recommendation(property_id=str(prop_id), similarity_score=float(score))
        for prop_id, score in recommendation_results[:TOP_N]
    ]
    
    return final_recommendations