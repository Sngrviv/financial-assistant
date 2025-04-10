from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from phi.api import API
from app.agents.crew_agents import FinancialCrewAgents
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

# Initialize CrewAI agents
crew_agents = FinancialCrewAgents()

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

class StockOpportunityRequest(BaseModel):
    symbol: str
    investment_amount: float
    time_horizon: int

class CrewStrategyRequest(BaseModel):
    risk_tolerance: str
    investment_goal: str
    investment_horizon: int
    initial_amount: float
    user_text: Optional[str] = None

# ----- Routes -----
@app.get("/")
async def root():
    return {"message": "Financial Assistant API"}

@app.post("/predict-stock", response_model=StockPredictionResponse)
async def predict_stock(request: StockPredictionRequest):
    """Predict stock prices for the given symbol using the Market Analyst agent"""
    try:
        result = crew_agents.analyze_stock_opportunity(
            symbol=request.symbol,
            investment_amount=10000,  # Default amount for analysis
            time_horizon=request.days // 365 + 1  # Convert days to years
        )
        return {
            "symbol": request.symbol,
            "predictions": result["technical_analysis"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/analyze-emotion", response_model=EmotionAnalysisResponse)
async def analyze_emotion(request: EmotionAnalysisRequest):
    """Analyze the emotion in the provided text using the Emotional Intelligence agent"""
    try:
        # Create a simple strategy request to utilize the emotion agent
        result = crew_agents.create_investment_strategy(
            risk_tolerance="moderate",
            investment_goal="analysis",
            investment_horizon=1,
            initial_amount=0,
            user_text=request.text
        )
        return result["emotional_analysis"]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/generate-comic")
async def generate_comic(request: ComicStripRequest):
    """Generate a comic strip based on investment choices using the Financial Educator agent"""
    try:
        # Create a strategy that focuses on educational content
        result = crew_agents.create_investment_strategy(
            risk_tolerance="moderate",
            investment_goal=request.investment_type,
            investment_horizon=request.years,
            initial_amount=request.amount,
            user_text=None
        )
        return result["educational_materials"]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/get-investment-advice")
async def get_investment_advice(request: InvestmentAdviceRequest):
    """Get personalized investment advice using all agents"""
    try:
        result = crew_agents.create_investment_strategy(
            risk_tolerance=request.risk_tolerance,
            investment_goal=request.investment_goal,
            investment_horizon=request.investment_horizon,
            initial_amount=request.initial_amount,
            user_text=request.emotion
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/analyze-stock-opportunity")
async def analyze_stock_opportunity(request: StockOpportunityRequest):
    """Analyze a specific stock investment opportunity using all agents"""
    try:
        result = crew_agents.analyze_stock_opportunity(
            symbol=request.symbol,
            investment_amount=request.investment_amount,
            time_horizon=request.time_horizon
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/create-investment-strategy")
async def create_investment_strategy(request: CrewStrategyRequest):
    """Create a comprehensive investment strategy using all agents"""
    try:
        result = crew_agents.create_investment_strategy(
            risk_tolerance=request.risk_tolerance,
            investment_goal=request.investment_goal,
            investment_horizon=request.investment_horizon,
            initial_amount=request.initial_amount,
            user_text=request.user_text
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 