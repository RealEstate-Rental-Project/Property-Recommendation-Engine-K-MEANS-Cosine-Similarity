from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import Dict, Any
from .schemas import UserQuery, RecommendationResponse
from .model_service import load_artifacts, get_recommendations

# Global state to hold artifacts or initialization info
app_state: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup hook for loading ML models and artifacts once.
    """
    if load_artifacts():
        app_state["models_ready"] = True
    else:
        app_state["models_ready"] = False
        print("FATAL: Recommendation service models are not ready.")
    yield
    # Shutdown logic (if any cleanup is needed)

app = FastAPI(
    title="Property Recommendation AI Microservice",
    description="Uses K-Means Clustering for segmentation and Cosine Similarity for ranking property recommendations.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def health_check():
    """Endpoint for basic health check."""
    return {"status": "ok", "service": "property-recommendation-ai", "models_ready": app_state.get("models_ready", False)}

@app.post("/recommend", response_model=RecommendationResponse)
def recommend_properties(query: UserQuery):
    """
    Accepts a user query (filters + profile data) and returns a ranked list 
    of property IDs using K-Means clustering and Cosine Similarity.
    """
    if not app_state.get("models_ready"):
        raise HTTPException(status_code=503, detail="AI models are not yet loaded or failed to initialize.")
        
    try:
        recommendations = get_recommendations(query)
        return RecommendationResponse(recommendations=recommendations)
    except Exception as e:
        # Log the error internally and return a generic error message
        print(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail="Internal error during recommendation calculation.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)