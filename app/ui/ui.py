import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import json

# Base URL for API calls
API_BASE_URL = "http://localhost:8000"

def main():
    st.set_page_config(
        page_title="Financial Assistant",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("Financial Assistant 💰")
    st.sidebar.title("Navigation")
    
    # Navigation
    page = st.sidebar.radio(
        "Select a page",
        ["Home", "Stock Prediction", "Investment Advice", "Comic Generator", "Voice Assistant"]
    )
    
    # Display appropriate page
    if page == "Home":
        show_home_page()
    elif page == "Stock Prediction":
        show_stock_prediction_page()
    elif page == "Investment Advice":
        show_investment_advice_page()
    elif page == "Comic Generator":
        show_comic_generator_page()
    elif page == "Voice Assistant":
        show_voice_assistant_page()

def show_home_page():
    st.markdown("""
    ## Welcome to your AI Financial Assistant
    
    This app helps you with your financial journey:
    
    - 📈 **Stock Price Prediction**: Predict future stock prices using AI
    - 💬 **Investment Advice**: Get personalized advice based on your goals and emotions
    - 🎭 **Comic Generator**: Visualize investment outcomes through fun comic strips
    - 🗣️ **Voice Assistant**: Talk to your financial advisor in your preferred language
    
    Use the sidebar to navigate to different features.
    """)
    
    st.info("Start by checking out the Stock Prediction tool!")

def show_stock_prediction_page():
    st.header("Stock Price Prediction 📈")
    
    # User inputs
    symbol = st.text_input("Enter stock symbol (e.g., AAPL, MSFT):", "AAPL")
    days = st.slider("Prediction horizon (days):", 7, 90, 30)
    
    if st.button("Predict"):
        with st.spinner("Predicting stock prices..."):
            # Call prediction API (dummy for now)
            # response = requests.post(
            #     f"{API_BASE_URL}/predict-stock",
            #     json={"symbol": symbol, "days": days}
            # )
            # predictions = response.json()
            
            # Dummy data for now
            start_date = datetime.now()
            dates = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
            
            # Generate some fake prediction data
            import random
            start_price = 150
            prices = [start_price]
            for i in range(1, days):
                prices.append(prices[-1] * (1 + random.uniform(-0.03, 0.04)))
            
            confidence = [random.uniform(0.7, 0.95) for _ in range(days)]
            
            # Create DataFrame
            df = pd.DataFrame({
                "Date": dates,
                "Price": prices,
                "Confidence": confidence
            })
            
            # Display the plot
            fig = go.Figure()
            
            # Add prediction line
            fig.add_trace(go.Scatter(
                x=df["Date"],
                y=df["Price"],
                mode="lines",
                name="Predicted Price",
                line=dict(color="blue", width=2)
            ))
            
            # Add confidence interval
            upper_bound = [p * (1 + c/10) for p, c in zip(df["Price"], df["Confidence"])]
            lower_bound = [p * (1 - c/10) for p, c in zip(df["Price"], df["Confidence"])]
            
            fig.add_trace(go.Scatter(
                x=df["Date"],
                y=upper_bound,
                mode="lines",
                name="Upper Bound",
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=df["Date"],
                y=lower_bound,
                mode="lines",
                fill="tonexty",
                name="Confidence Interval",
                line=dict(width=0),
                fillcolor="rgba(0, 0, 255, 0.2)"
            ))
            
            fig.update_layout(
                title=f"Stock Price Prediction for {symbol}",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display the prediction data
            st.subheader("Prediction Details")
            st.dataframe(df)

def show_investment_advice_page():
    st.header("Personalized Investment Advice 💬")
    
    # User inputs
    col1, col2 = st.columns(2)
    
    with col1:
        risk_tolerance = st.select_slider(
            "Risk Tolerance:",
            options=["Very Low", "Low", "Medium", "High", "Very High"],
            value="Medium"
        )
        
        investment_goal = st.selectbox(
            "Investment Goal:",
            ["Retirement", "House Purchase", "Education", "Wealth Building", "Other"]
        )
    
    with col2:
        investment_horizon = st.slider("Investment Horizon (years):", 1, 40, 10)
        initial_amount = st.number_input("Initial Investment Amount ($):", min_value=1000, value=10000, step=1000)
    
    # Text input for emotion analysis
    st.subheader("How are you feeling about investing right now?")
    user_text = st.text_area("Share your thoughts:")
    
    emotion = None
    if user_text:
        # Analyze emotion (dummy for now)
        emotion = "anxious" if "worry" in user_text.lower() or "anxious" in user_text.lower() else "neutral"
        
        if emotion == "anxious":
            st.info("I notice you seem a bit concerned about investing. That's perfectly normal.")
        else:
            st.info("You seem to have a balanced perspective on investing. That's great!")
    
    if st.button("Get Advice"):
        with st.spinner("Generating personalized advice..."):
            # Call investment advice API (dummy for now)
            # response = requests.post(
            #     f"{API_BASE_URL}/get-investment-advice",
            #     json={
            #         "risk_tolerance": risk_tolerance,
            #         "investment_goal": investment_goal,
            #         "investment_horizon": investment_horizon,
            #         "initial_amount": initial_amount,
            #         "emotion": emotion
            #     }
            # )
            # advice = response.json()
            
            # Dummy advice for now
            advice_text = ""
            if emotion == "anxious":
                advice_text = "I understand you're feeling uncertain about investing right now. That's completely normal, especially given market volatility."
                advice_text += " Based on your longer-term goals, consider a more conservative approach initially, and gradually increase your risk exposure as you become more comfortable."
            else:
                advice_text = "Based on your risk tolerance and investment horizon, I recommend a diversified portfolio approach."
                
            if risk_tolerance in ["Very Low", "Low"]:
                stock_percentage = 30
                bond_percentage = 60
                cash_percentage = 10
            elif risk_tolerance == "Medium":
                stock_percentage = 60
                bond_percentage = 30
                cash_percentage = 10
            else:
                stock_percentage = 80
                bond_percentage = 15
                cash_percentage = 5
                
            allocation = {
                "stocks": stock_percentage,
                "bonds": bond_percentage,
                "cash": cash_percentage
            }
            
            st.success("Here's your personalized investment advice:")
            st.write(advice_text)
            
            st.subheader("Recommended Asset Allocation")
            
            # Create pie chart for asset allocation
            fig = go.Figure(data=[go.Pie(
                labels=list(allocation.keys()),
                values=list(allocation.values()),
                hole=.3,
                marker_colors=['#636EFA', '#EF553B', '#00CC96']
            )])
            
            fig.update_layout(
                title="Asset Allocation",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Projected growth
            growth_rate = {
                "stocks": 0.08,
                "bonds": 0.04,
                "cash": 0.01
            }
            
            # Calculate weighted average return
            avg_return = sum(allocation[asset] * growth_rate[asset] for asset in allocation) / 100
            
            # Calculate future value
            future_value = initial_amount * (1 + avg_return) ** investment_horizon
            
            st.metric(
                label=f"Projected Value After {investment_horizon} Years",
                value=f"${future_value:,.2f}",
                delta=f"${future_value - initial_amount:,.2f}"
            )

def show_comic_generator_page():
    st.header("Investment Comic Strip Generator 🎭")
    
    # User inputs
    investment_type = st.selectbox(
        "Investment Type:",
        ["Stocks", "Bonds", "Real Estate", "Mutual Funds", "ETFs", "Cryptocurrency"]
    )
    
    amount = st.number_input("Investment Amount ($):", min_value=1000, value=10000, step=1000)
    years = st.slider("Investment Period (years):", 1, 40, 10)
    expected_return = st.slider("Expected Annual Return (%):", 1.0, 20.0, 8.0) / 100
    
    if st.button("Generate Comic"):
        with st.spinner("Creating your personalized investment comic..."):
            # Call comic generator API (dummy for now)
            # response = requests.post(
            #     f"{API_BASE_URL}/generate-comic",
            #     json={
            #         "investment_type": investment_type,
            #         "amount": amount,
            #         "years": years,
            #         "expected_return": expected_return
            #     }
            # )
            # comic = response.json()
            
            # Dummy comic for now
            future_value = amount * (1 + expected_return) ** years
            gain = future_value - amount
            
            st.success("Your investment story has been visualized!")
            
            # Display a placeholder for the comic strip
            st.info("Comic strip visualization")
            
            st.image("https://via.placeholder.com/800x300?text=Your+Investment+Journey+Comic+Strip", 
                     caption="Investment Comic Strip (Placeholder)")
            
            # Display the investment outcome
            st.subheader("Investment Outcome")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Initial Investment", f"${amount:,.2f}")
            
            with col2:
                st.metric("Future Value", f"${future_value:,.2f}")
            
            with col3:
                st.metric("Total Gain", f"${gain:,.2f}", f"{gain/amount*100:.1f}%")
            
            # Show investment growth over time
            years_list = list(range(years + 1))
            values = [amount * (1 + expected_return) ** year for year in years_list]
            
            df = pd.DataFrame({
                "Year": years_list,
                "Value": values
            })
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df["Year"],
                y=df["Value"],
                mode="lines+markers",
                name="Investment Growth",
                line=dict(color="green", width=2)
            ))
            
            fig.update_layout(
                title=f"{investment_type} Investment Growth Over {years} Years",
                xaxis_title="Year",
                yaxis_title="Value ($)",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)

def show_voice_assistant_page():
    st.header("Voice-Activated Financial Assistant 🗣️")
    
    # Language selection
    language = st.selectbox(
        "Select your preferred language:",
        ["English", "Hindi", "Spanish", "French", "Mandarin", "Japanese"]
    )
    
    st.write("Click the button below and speak your financial question:")
    
    if st.button("Start Listening"):
        with st.spinner("Listening..."):
            # In a real implementation, this would use the microphone
            # For now, we'll simulate with a text input
            st.info("🎤 Listening for your question...")
            
            # Simulate voice recognition with a text input for now
            st.text_input("Type your question (voice simulation):")
            
            # Display a mock response
            st.success("I heard your question! Here's my response:")
            
            st.write("""
            Based on your current financial situation, I'd recommend starting with an emergency fund
            that covers 3-6 months of expenses. After that, consider investing in a low-cost index
            fund to build long-term wealth.
            """)
            
            # Add a playback option (simulated)
            st.audio("https://www2.cs.uic.edu/~i101/SoundFiles/StarWars3.wav", format="audio/wav")

if __name__ == "__main__":
    main() 