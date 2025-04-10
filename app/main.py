import os
import uvicorn
from dotenv import load_dotenv
from phi.app import App
from app.api.api import api
from app.ui.ui import ui

# Load environment variables
load_dotenv()

# Create phidata app
app = App(
    title="Financial Assistant",
    description="AI-powered financial assistant with stock prediction and personalized advice",
    api=api,
    ui=ui,
)

def start():
    """Start the application"""
    # Start the API
    uvicorn.run("app.api.api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start() 