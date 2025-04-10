from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun
from app.models.stock_predictor import StockPredictor
from app.models.emotion_analyzer import EmotionAnalyzer
from app.models.comic_generator import ComicGenerator
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialCrewAgents:
    """
    CrewAI-based implementation of financial agents that work together
    to provide comprehensive financial advice and analysis.
    """
    
    def __init__(self, models_dir: str = "data/models"):
        self.models_dir = models_dir
        self.search_tool = DuckDuckGoSearchRun()
        
        # Initialize models
        self.stock_predictor = StockPredictor()
        self.emotion_analyzer = EmotionAnalyzer()
        self.comic_generator = ComicGenerator()
        
        # Initialize agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize the specialized financial agents"""
        
        # Chief Financial Officer (CFO) Agent
        self.cfo_agent = Agent(
            name="Chief Financial Officer",
            role="Senior Financial Advisor",
            goal="Provide high-level financial strategy and coordinate other agents",
            backstory="""You are an experienced CFO with decades of experience in 
            financial markets and personal finance. You coordinate the efforts of 
            other specialized agents to provide comprehensive financial advice.""",
            tools=[self.search_tool],
            verbose=True
        )
        
        # Market Analyst Agent
        self.market_analyst = Agent(
            name="Market Analyst",
            role="Technical and Fundamental Analyst",
            goal="Analyze market trends and provide accurate stock predictions",
            backstory="""You are an expert in technical and fundamental analysis 
            with a deep understanding of market patterns and economic indicators.""",
            tools=[self.search_tool, self.stock_predictor.predict],
            verbose=True
        )
        
        # Risk Manager Agent
        self.risk_manager = Agent(
            name="Risk Manager",
            role="Risk Assessment Specialist",
            goal="Evaluate and mitigate investment risks",
            backstory="""You specialize in risk assessment and management, 
            ensuring investment strategies align with user risk tolerance.""",
            tools=[self.search_tool],
            verbose=True
        )
        
        # Financial Educator Agent
        self.educator = Agent(
            name="Financial Educator",
            role="Financial Education Specialist",
            goal="Create engaging educational content about finance",
            backstory="""You excel at explaining complex financial concepts 
            through engaging stories and visual content.""",
            tools=[self.search_tool, self.comic_generator.generate_comic_strip],
            verbose=True
        )
        
        # Emotional Intelligence Agent
        self.emotion_agent = Agent(
            name="Emotional Intelligence Advisor",
            role="Behavioral Finance Specialist",
            goal="Analyze and account for emotional factors in financial decisions",
            backstory="""You understand the psychological aspects of investing 
            and help users make rational decisions despite emotional biases.""",
            tools=[self.emotion_analyzer.analyze],
            verbose=True
        )
    
    def create_investment_strategy(
        self,
        risk_tolerance: str,
        investment_goal: str,
        investment_horizon: int,
        initial_amount: float,
        user_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a comprehensive investment strategy using all agents
        
        Args:
            risk_tolerance: User's risk tolerance level
            investment_goal: User's financial goals
            investment_horizon: Investment timeline in years
            initial_amount: Initial investment amount
            user_text: Optional text for emotion analysis
            
        Returns:
            Dict containing the complete investment strategy
        """
        
        # Create tasks for each agent
        emotion_task = Task(
            description=f"Analyze emotional state from text: {user_text}",
            agent=self.emotion_agent
        )
        
        market_analysis_task = Task(
            description=f"""Analyze market conditions and recommend asset allocation 
            for {initial_amount} with {investment_horizon} year horizon""",
            agent=self.market_analyst
        )
        
        risk_assessment_task = Task(
            description=f"""Evaluate risks and provide mitigation strategies for 
            {risk_tolerance} risk tolerance and {investment_goal} goal""",
            agent=self.risk_manager
        )
        
        education_task = Task(
            description=f"""Create educational materials explaining the investment 
            strategy and potential outcomes""",
            agent=self.educator
        )
        
        strategy_task = Task(
            description="""Synthesize all analyses and create final investment 
            strategy recommendations""",
            agent=self.cfo_agent
        )
        
        # Create and run the crew
        crew = Crew(
            agents=[
                self.emotion_agent,
                self.market_analyst,
                self.risk_manager,
                self.educator,
                self.cfo_agent
            ],
            tasks=[
                emotion_task,
                market_analysis_task,
                risk_assessment_task,
                education_task,
                strategy_task
            ],
            verbose=True
        )
        
        result = crew.kickoff()
        
        return {
            "strategy": result,
            "emotional_analysis": emotion_task.output if user_text else None,
            "market_analysis": market_analysis_task.output,
            "risk_assessment": risk_assessment_task.output,
            "educational_materials": education_task.output
        }
    
    def analyze_stock_opportunity(
        self,
        symbol: str,
        investment_amount: float,
        time_horizon: int
    ) -> Dict[str, Any]:
        """
        Analyze a specific stock investment opportunity
        
        Args:
            symbol: Stock symbol to analyze
            investment_amount: Proposed investment amount
            time_horizon: Investment timeline in years
            
        Returns:
            Dict containing the analysis results
        """
        
        # Create specialized tasks for stock analysis
        technical_analysis_task = Task(
            description=f"Perform technical analysis for {symbol}",
            agent=self.market_analyst
        )
        
        risk_analysis_task = Task(
            description=f"""Evaluate specific risks for {symbol} investment 
            of {investment_amount} over {time_horizon} years""",
            agent=self.risk_manager
        )
        
        education_task = Task(
            description=f"""Create educational materials about {symbol} 
            and its industry""",
            agent=self.educator
        )
        
        recommendation_task = Task(
            description=f"""Synthesize analyses and provide final recommendation 
            for {symbol} investment""",
            agent=self.cfo_agent
        )
        
        # Create and run the crew
        crew = Crew(
            agents=[
                self.market_analyst,
                self.risk_manager,
                self.educator,
                self.cfo_agent
            ],
            tasks=[
                technical_analysis_task,
                risk_analysis_task,
                education_task,
                recommendation_task
            ],
            verbose=True
        )
        
        result = crew.kickoff()
        
        return {
            "recommendation": result,
            "technical_analysis": technical_analysis_task.output,
            "risk_analysis": risk_analysis_task.output,
            "educational_materials": education_task.output
        } 