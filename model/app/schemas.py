from pydantic import BaseModel, Field
from typing import List, Optional

# --- Input Schema: What the user sends to the API (Filters + Profile) ---
class UserQuery(BaseModel):
    """
    Schema for the user's input/query vector features.
    These features define the user's intent and will be converted into a vector.
    """
    # Numerical Features (from user filters)
    target_rent: float = Field(..., description="The user's maximum desired rent amount (in the unit matching preferred_rental_type, e.g., 1500 for monthly or 50 for daily).")
    min_total_rooms: int = Field(..., description="The user's minimum desired number of rooms.")
    target_sqft: float = Field(..., description="The user's minimum desired square meters.")
    
    # Location (from search area)
    search_latitude: float = Field(..., description="The latitude of the target search area.")
    search_longitude: float = Field(..., description="The longitude of the target search area.")
    
    # Categorical Feature (from user filter)
    # Using simple strings; model_service will handle One-Hot Encoding
    preferred_property_type: str = Field(..., description="Preferred property type (e.g., APARTMENT, HOUSE, VILLA, STUDIO).")
    
    # New rental type
    preferred_rental_type: str = Field(..., description="Preferred rental type (e.g., MONTHLY, DAILY).")
    
    # User Profile/Contextual Features
    number_of_people: int = Field(..., description="Number of people in the user's family/household.")
    is_married: bool = Field(..., description="Whether the user is married or not.")

# --- Output Schema: What the API returns ---
class Recommendation(BaseModel):
    """
    Schema for a single recommended property result.
    """
    property_id: str = Field(..., description="The unique ID of the recommended property.")
    similarity_score: float = Field(..., description="The cosine similarity score (0 to 1) indicating relevance.")

class RecommendationResponse(BaseModel):
    """
    Schema for the final list of recommended properties.
    """
    recommendations: List[Recommendation] = Field(..., description="A ranked list of property recommendations.")