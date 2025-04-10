from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from phi.api import API
import os

# FastAPI app
app = FastAPI(
    title="Financial Assistant API",
    description="API for the AI-powered financial assistant",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create Phidata API
api = API(app=app, prefix="/api")

# ----- Models -----
class StockPredictionRequest(BaseModel):
    symbol: str
    days: int = 30

class StockPredictionResponse(BaseModel):
    symbol: str
    predictions: List[Dict[str, Any]]
    
class EmotionAnalysisRequest(BaseModel):
    text: str

class EmotionAnalysisResponse(BaseModel):
    emotion: str
    confidence: float
    
class InvestmentAdviceRequest(BaseModel):
    risk_tolerance: str
    investment_goal: str
    investment_horizon: int
    initial_amount: float
    emotion: Optional[str] = None

class ComicStripRequest(BaseModel):
    investment_type: str
    amount: float
    years: int
    expected_return: float

# ----- Routes -----
@app.get("/")
async def root():
    return {"message": "Financial Assistant API"}

@app.post("/predict-stock", response_model=StockPredictionResponse)
async def predict_stock(request: StockPredictionRequest):
    """Predict stock prices for the given symbol"""
    # This will be implemented with our ML model
    # For now, return dummy data
    return {
        "symbol": request.symbol,
        "predictions": [
            {"date": "2023-01-01", "price": 100, "confidence": 0.9},
            {"date": "2023-01-02", "price": 105, "confidence": 0.85},
        ]
    }

@app.post("/analyze-emotion", response_model=EmotionAnalysisResponse)
async def analyze_emotion(request: EmotionAnalysisRequest):
    """Analyze the emotion in the provided text"""
    # This will be implemented with our emotion analysis model
    # For now, return dummy data
    return {
        "emotion": "neutral",
        "confidence": 0.8
    }

@app.post("/generate-comic")
async def generate_comic(request: ComicStripRequest):
    """Generate a comic strip based on investment choices"""
    # This will be implemented with our comic generator
    # For now, return dummy data
    return {
        "comic_url": "https://example.com/comic.png",
        "description": f"Comic showing {request.investment_type} investment of ${request.amount} over {request.years} years",
    }

@app.post("/get-investment-advice")
async def get_investment_advice(request: InvestmentAdviceRequest):
    """Get personalized investment advice"""
    # This will be implemented with our advice generator
    # For now, return dummy data
    return {
        "advice": "Based on your risk tolerance and goals, consider a balanced portfolio of stocks and bonds.",
        "recommended_allocation": {
            "stocks": 60,
            "bonds": 30,
            "cash": 10
        }
    } 