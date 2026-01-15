# Property Recommendation Engine

## Project Structure

```
requirements.txt          # Python dependencies and package requirements
README.md
.gitignore
model/                    # Main application package
├── __init__.py          # Package initialization
├── __pycache__/         # Compiled Python bytecode (auto-generated)
└── app/                 # Application logic and services
    ├── __init__.py      # Package initialization
    ├── main.py          # Main application entry point and API routes
    ├── model_service.py # Service layer for model operations and recommendations
    ├── schemas.py       # Data models and validation schemas
    ├── train_model.py   # Script for training the recommendation model
    └── __pycache__/     # Compiled Python bytecode (auto-generated)
    data/                     # Data storage and processed datasets
    ├── properties_data.csv  # Raw property data in CSV format
    └── property_feature_matrix.npy  # Preprocessed feature matrix for model input
    models/                   # Trained machine learning models and preprocessors
    ├── k_means_model.pkl    # Trained K-means clustering model for recommendations
    └── preprocessor.pkl     # Data preprocessing pipeline (scaling, encoding, etc.)
```