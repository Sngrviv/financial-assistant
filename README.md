# Financial Assistant AI Agent

An agentic AI application built with Phidata that provides personalized financial advice and education through:

1. **Stock Price Prediction**: Using transformers/LSTM models
2. **Voice-Activated Assistant**: With regional language support
3. **Investment Comic Strip Generator**: Visualize investment outcomes through AI-generated comics
4. **Emotion-Aware Investment Coach**: Adapts advice based on user's emotional state

## Features

- **Storytelling Power**: Makes financial education relatable through narratives
- **Interactive Learning**: Choose-your-own-adventure format for learning by doing
- **Personalized Experience**: Tailored advice based on goals and risk tolerance
- **Engagement Gamification**: Rewards to keep users coming back

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env`
4. Run the application: `python -m app.main`

## Project Structure

- `app/`: Main application code
  - `models/`: ML models for stock prediction and emotion analysis
  - `agents/`: Agent definitions and workflows
  - `api/`: FastAPI backend 
  - `ui/`: Streamlit frontend
  - `utils/`: Utility functions
- `data/`: Data storage and processing
- `tests/`: Test cases
- `notebooks/`: Jupyter notebooks for model development 