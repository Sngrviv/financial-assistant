from phidata.agent import Agent
from app.models.stock_predictor import StockPredictor
from app.models.emotion_analyzer import EmotionAnalyzer
from app.models.comic_generator import ComicGenerator
from app.models.voice_assistant import VoiceAssistant
import os
import logging
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialAgent:
    """
    An agent that combines stock prediction, emotion analysis, 
    comic strip generation, and voice assistance to provide
    personalized financial guidance.
    """
    
    def __init__(
        self,
        models_dir: str = "data/models",
        use_voice: bool = True,
        preferred_language: str = "English",
        model_type: str = "lstm"  # or "transformer"
    ):
        """
        Initialize the financial agent
        
        Args:
            models_dir: Directory to store models
            use_voice: Whether to enable voice assistant
            preferred_language: Default language for voice assistant
            model_type: Type of stock prediction model to use
        """
        self.models_dir = models_dir
        self.use_voice = use_voice
        self.preferred_language = preferred_language
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize components
        logger.info("Initializing stock predictor...")
        self.stock_predictor = StockPredictor(model_type=model_type)
        
        logger.info("Initializing emotion analyzer...")
        self.emotion_analyzer = EmotionAnalyzer(use_transformer=False)  # Use rule-based for simplicity
        
        logger.info("Initializing comic generator...")
        self.comic_generator = ComicGenerator()
        
        if use_voice:
            logger.info(f"Initializing voice assistant with language: {preferred_language}...")
            self.voice_assistant = VoiceAssistant(
                preferred_language=preferred_language,
                use_emotion_analysis=True
            )
        
        logger.info("Financial agent initialized successfully")
    
    def predict_stock(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Predict stock prices for the given symbol
        
        Args:
            symbol: Stock symbol (e.g., AAPL, MSFT)
            days: Number of days to predict
            
        Returns:
            dict: Prediction results
        """
        try:
            logger.info(f"Predicting stock prices for {symbol} for next {days} days")
            predictions = self.stock_predictor.predict(symbol, days)
            
            # Convert predictions to the expected format
            return {
                "symbol": symbol,
                "predictions": [
                    {
                        "date": row["Date"].strftime("%Y-%m-%d"),
                        "price": float(row["Price"]),
                        "confidence": float(row["Confidence"])
                    }
                    for _, row in predictions.iterrows()
                ]
            }
        except Exception as e:
            logger.error(f"Error predicting stock prices: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "predictions": []
            }
    
    def analyze_emotion(self, text: str) -> Dict[str, Any]:
        """
        Analyze the emotion in the given text
        
        Args:
            text: Text to analyze
            
        Returns:
            dict: Emotion analysis results
        """
        try:
            logger.info(f"Analyzing emotion in text: '{text[:50]}...' (truncated)")
            emotion, confidence = self.emotion_analyzer.analyze(text)
            
            return {
                "emotion": emotion,
                "confidence": confidence
            }
        except Exception as e:
            logger.error(f"Error analyzing emotion: {e}")
            return {
                "emotion": "neutral",
                "confidence": 0.5,
                "error": str(e)
            }
    
    def generate_comic(
        self,
        investment_type: str,
        amount: float,
        years: int,
        expected_return: float
    ) -> Dict[str, Any]:
        """
        Generate a comic strip based on investment choices
        
        Args:
            investment_type: Type of investment (e.g., Stocks, Bonds)
            amount: Investment amount
            years: Investment period in years
            expected_return: Expected annual return (decimal)
            
        Returns:
            dict: Comic generation results
        """
        try:
            logger.info(f"Generating comic for {investment_type} investment of ${amount:.2f} over {years} years")
            comic_result = self.comic_generator.generate_comic_strip(
                investment_type, amount, years, expected_return
            )
            
            # Also generate alternative scenarios for comparison
            alternatives = self.comic_generator.generate_alternative_scenarios(
                investment_type, amount, years, expected_return
            )
            
            return {
                "comic_path": comic_result["comic_path"],
                "description": comic_result["description"],
                "investment_type": investment_type,
                "amount": amount,
                "years": years,
                "expected_return": expected_return,
                "final_value": comic_result["final_value"],
                "gain_percentage": comic_result["gain_percentage"],
                "alternatives": alternatives
            }
        except Exception as e:
            logger.error(f"Error generating comic: {e}")
            return {
                "error": str(e),
                "investment_type": investment_type,
                "amount": amount,
                "years": years,
                "expected_return": expected_return
            }
    
    def get_investment_advice(
        self,
        risk_tolerance: str,
        investment_goal: str,
        investment_horizon: int,
        initial_amount: float,
        emotion: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get personalized investment advice
        
        Args:
            risk_tolerance: User's risk tolerance
            investment_goal: User's investment goal
            investment_horizon: Investment horizon in years
            initial_amount: Initial investment amount
            emotion: Detected user emotion (optional)
            
        Returns:
            dict: Investment advice
        """
        try:
            logger.info(f"Generating investment advice for {risk_tolerance} risk tolerance, "
                      f"{investment_goal} goal, {investment_horizon} year horizon")
            
            # If emotion is not provided, use neutral
            if not emotion:
                emotion = "neutral"
            
            # Get investment advice based on emotion and other parameters
            advice_text = self.emotion_analyzer.get_investment_advice(
                emotion, risk_tolerance, investment_goal
            )
            
            # Determine asset allocation based on risk tolerance
            if risk_tolerance in ["Very Low", "Low"]:
                allocation = {
                    "stocks": 30,
                    "bonds": 60,
                    "cash": 10
                }
                expected_return = 0.04  # 4% annual return
            elif risk_tolerance == "Medium":
                allocation = {
                    "stocks": 60,
                    "bonds": 30,
                    "cash": 10
                }
                expected_return = 0.06  # 6% annual return
            else:  # High or Very High
                allocation = {
                    "stocks": 80,
                    "bonds": 15,
                    "cash": 5
                }
                expected_return = 0.08  # 8% annual return
            
            # Calculate expected future value
            future_value = initial_amount * (1 + expected_return) ** investment_horizon
            
            return {
                "advice": advice_text,
                "recommended_allocation": allocation,
                "expected_annual_return": expected_return * 100,  # Convert to percentage
                "initial_amount": initial_amount,
                "investment_horizon": investment_horizon,
                "expected_future_value": future_value,
                "emotion_context": emotion
            }
        except Exception as e:
            logger.error(f"Error generating investment advice: {e}")
            return {
                "error": str(e),
                "advice": "Based on your risk tolerance and goals, I'd recommend consulting with a financial advisor for personalized advice."
            }
    
    def process_voice_query(self, query: Optional[str] = None, audio_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a voice query
        
        Args:
            query: Text query (if None, use audio_file or microphone)
            audio_file: Path to audio file (if None and query is None, use microphone)
            
        Returns:
            dict: Voice assistant response
        """
        if not self.use_voice:
            return {"error": "Voice assistant is disabled"}
        
        try:
            logger.info("Processing voice query")
            return self.voice_assistant.process_query(query, audio_file)
        except Exception as e:
            logger.error(f"Error processing voice query: {e}")
            return {"error": str(e)}
    
    def create_phi_agent(self) -> Agent:
        """
        Create a Phidata agent that integrates all components
        
        Returns:
            Agent: Phidata agent instance
        """
        # Define instructions for the agent
        instructions = """
        You are a sophisticated AI financial assistant who helps users with:
        
        1. Stock price predictions
        2. Personalized investment advice based on their financial goals and emotions
        3. Visualizing investment outcomes through comics
        4. Answering financial questions via voice in multiple languages
        
        Your goal is to make financial concepts accessible and engaging, while providing
        personalized guidance that considers both rational financial factors and the
        user's emotional state.
        """
        
        # Create the agent
        agent = Agent(
            name="Financial Assistant",
            description="AI-powered financial assistant with stock prediction and personalized advice",
            instructions=instructions,
        )
        
        # Define agent tools
        @agent.tool("predict_stock_prices")
        def predict_stock_prices(symbol: str, days: int = 30) -> Dict[str, Any]:
            """Predict stock prices for the given symbol for the specified number of days."""
            return self.predict_stock(symbol, days)
        
        @agent.tool("analyze_user_emotion")
        def analyze_user_emotion(text: str) -> Dict[str, Any]:
            """Analyze the emotion in the given text to personalize financial advice."""
            return self.analyze_emotion(text)
        
        @agent.tool("generate_investment_comic")
        def generate_investment_comic(investment_type: str, amount: float, years: int, expected_return: float) -> Dict[str, Any]:
            """Generate a comic strip visualizing an investment journey and outcomes."""
            return self.generate_comic(investment_type, amount, years, expected_return)
        
        @agent.tool("get_personalized_investment_advice")
        def get_personalized_investment_advice(
            risk_tolerance: str,
            investment_goal: str,
            investment_horizon: int,
            initial_amount: float,
            emotion: Optional[str] = None
        ) -> Dict[str, Any]:
            """Get personalized investment advice based on user's profile and emotional state."""
            return self.get_investment_advice(risk_tolerance, investment_goal, investment_horizon, initial_amount, emotion)
        
        @agent.tool("process_voice_question")
        def process_voice_question(query: Optional[str] = None, audio_file: Optional[str] = None) -> Dict[str, Any]:
            """Process a voice query about finance in multiple languages."""
            if not self.use_voice:
                return {"error": "Voice assistant is disabled"}
            return self.process_voice_query(query, audio_file)
        
        return agent
    
    def train_stock_model(self, symbol: str, epochs: int = 50, save_model: bool = True) -> None:
        """
        Train the stock prediction model
        
        Args:
            symbol: Stock symbol to train on
            epochs: Number of training epochs
            save_model: Whether to save the trained model
        """
        try:
            logger.info(f"Training stock prediction model for {symbol} with {epochs} epochs")
            
            # Get historical data for the symbol
            historical_data = self.stock_predictor._get_stock_data(symbol)
            
            # Prepare data for training
            train_loader, test_loader = self.stock_predictor.prepare_data(historical_data)
            
            # Train the model
            self.stock_predictor.train(train_loader, epochs=epochs)
            
            # Evaluate the model
            test_loss = self.stock_predictor.evaluate(test_loader)
            logger.info(f"Model training completed with test loss: {test_loss:.6f}")
            
            # Save the model if requested
            if save_model:
                model_path = os.path.join(self.models_dir, f"stock_model_{symbol}.pt")
                self.stock_predictor.save_model(model_path)
                logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error training stock model: {e}")
    
    def load_stock_model(self, symbol: str) -> bool:
        """
        Load a previously trained stock prediction model
        
        Args:
            symbol: Stock symbol of the model to load
            
        Returns:
            bool: Whether the model was loaded successfully
        """
        try:
            model_path = os.path.join(self.models_dir, f"stock_model_{symbol}.pt")
            
            if not os.path.exists(model_path):
                logger.warning(f"No model found for {symbol} at {model_path}")
                return False
            
            logger.info(f"Loading stock prediction model for {symbol} from {model_path}")
            self.stock_predictor.load_model(model_path)
            logger.info(f"Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading stock model: {e}")
            return False 