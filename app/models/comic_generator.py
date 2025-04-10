import os
import json
import random
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
import tempfile
import numpy as np

class ComicGenerator:
    """
    Generator for investment comics that visually explain investment choices and outcomes
    
    This is a placeholder implementation that creates simple comic frames with text.
    In a real implementation, this would likely use a more sophisticated image generation model.
    """
    
    def __init__(self, comic_templates_dir=None, output_dir=None):
        """
        Initialize the comic generator
        
        Args:
            comic_templates_dir: Directory containing comic templates (optional)
            output_dir: Directory to save generated comics (optional)
        """
        self.comic_templates_dir = comic_templates_dir
        self.output_dir = output_dir or os.path.join(tempfile.gettempdir(), "investment_comics")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Template scenarios for different investment types
        self.templates = {
            "Stocks": [
                "Buying individual stocks and watching their growth over time",
                "Diversifying across multiple stocks to reduce risk",
                "Reinvesting dividends to compound growth",
                "Navigating market volatility with a long-term perspective"
            ],
            "Bonds": [
                "Investing in bonds for steady income",
                "Understanding how interest rates affect bond prices",
                "Using bonds to balance a portfolio",
                "Comparing government and corporate bonds"
            ],
            "Real Estate": [
                "Investing in rental properties for income",
                "Exploring REITs as a way to invest in real estate without direct ownership",
                "Understanding property appreciation over time",
                "Balancing property maintenance costs with rental income"
            ],
            "Mutual Funds": [
                "Investing in mutual funds for instant diversification",
                "Understanding expense ratios and their impact",
                "Comparing actively managed funds to index funds",
                "Dollar-cost averaging with regular mutual fund investments"
            ],
            "ETFs": [
                "Investing in ETFs for diversification and liquidity",
                "Comparing sector-specific and broad market ETFs",
                "Understanding ETF expenses and trading costs",
                "Building a portfolio with different ETF types"
            ],
            "Cryptocurrency": [
                "Understanding the volatility of cryptocurrency investments",
                "Diversifying across different cryptocurrencies",
                "Balancing crypto with traditional investments",
                "Long-term hodling versus active trading strategies"
            ]
        }
        
        # Character templates
        self.characters = [
            "Prudent Penny", "Risk-taking Rick", "Balanced Barbara", "Nervous Nick", 
            "Excited Emma", "Thoughtful Thomas", "Diversified Diana", "Long-term Larry"
        ]
        
        # Panel templates
        self.panel_templates = [
            "Character introduces investment concept",
            "Character explains investment mechanics",
            "Character shows investment growth over time",
            "Character reveals final outcome and lessons learned"
        ]
    
    def _generate_simple_comic_frame(self, text, width=400, height=300, bg_color=(255, 255, 255), 
                                     text_color=(0, 0, 0), character=None):
        """Generate a simple comic frame with text and optional character"""
        # Create blank image
        img = Image.new('RGB', (width, height), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw border
        draw.rectangle([(5, 5), (width-5, height-5)], outline=(200, 200, 200), width=2)
        
        # Draw character placeholder at bottom left (if provided)
        if character:
            draw.ellipse([(20, height-80), (80, height-20)], fill=(150, 150, 220), outline=(100, 100, 200))
            char_font = font.font_variant(size=10)
            draw.text((30, height-60), character[:2], fill=(50, 50, 150), font=char_font)
        
        # Draw text with wrapping (simple implementation)
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            w, h = draw.textsize(test_line, font=font)
            
            if w <= width - 40:  # Leave margin
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                
        if current_line:
            lines.append(' '.join(current_line))
        
        y_position = 30
        for line in lines:
            draw.text((20, y_position), line, fill=text_color, font=font)
            y_position += 20
            
        return img
    
    def generate_comic_strip(self, investment_type, amount, years, expected_return):
        """
        Generate a comic strip based on investment choices
        
        Args:
            investment_type: Type of investment (Stocks, Bonds, etc.)
            amount: Investment amount
            years: Investment period in years
            expected_return: Expected annual return (decimal)
            
        Returns:
            dict: Path to the generated comic and description
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comic_filename = f"investment_comic_{timestamp}.jpg"
        comic_path = os.path.join(self.output_dir, comic_filename)
        
        # Calculate future value
        future_value = amount * (1 + expected_return) ** years
        gain = future_value - amount
        
        # Select a character
        character = random.choice(self.characters)
        
        # Select a template scenario for this investment type
        if investment_type in self.templates:
            scenario = random.choice(self.templates[investment_type])
        else:
            scenario = "Exploring different investment options and their outcomes"
        
        # Generate a 4-panel comic strip
        frames = []
        
        # Panel 1: Introduction
        intro_text = f"{character} is considering investing ${amount:,.2f} in {investment_type} for {years} years."
        frames.append(self._generate_simple_comic_frame(intro_text, character=character))
        
        # Panel 2: Investment mechanics
        mechanics_text = f"With {investment_type}, the expected annual return is about {expected_return*100:.1f}%. {scenario}."
        frames.append(self._generate_simple_comic_frame(mechanics_text, character=character))
        
        # Panel 3: Growth over time
        # Show some intermediate points
        if years > 5:
            midpoint = years // 2
            mid_value = amount * (1 + expected_return) ** midpoint
            growth_text = f"After {midpoint} years, the investment grows to ${mid_value:,.2f}. Patience is key!"
        else:
            growth_text = f"The investment grows steadily over time, with some ups and downs along the way."
        
        frames.append(self._generate_simple_comic_frame(growth_text, character=character))
        
        # Panel 4: Final outcome
        outcome_text = f"After {years} years, the ${amount:,.2f} investment in {investment_type} has grown to ${future_value:,.2f}, a gain of ${gain:,.2f} ({gain/amount*100:.1f}%)!"
        frames.append(self._generate_simple_comic_frame(outcome_text, character=character))
        
        # Combine the frames horizontally
        total_width = sum(frame.width for frame in frames)
        max_height = max(frame.height for frame in frames)
        
        comic_strip = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        
        for frame in frames:
            comic_strip.paste(frame, (x_offset, 0))
            x_offset += frame.width
        
        # Save the comic strip
        comic_strip.save(comic_path)
        
        # Generate a description of the comic
        description = (
            f"A {len(frames)}-panel comic strip showing {character}'s journey investing ${amount:,.2f} in {investment_type} "
            f"over {years} years. The investment grows to ${future_value:,.2f}, representing a {gain/amount*100:.1f}% return."
        )
        
        return {
            "comic_path": comic_path,
            "description": description,
            "character": character,
            "investment_type": investment_type,
            "initial_amount": amount,
            "years": years,
            "final_value": future_value,
            "gain_percentage": gain/amount*100
        }
    
    def generate_alternative_scenarios(self, base_investment_type, amount, years, expected_return):
        """
        Generate alternative investment scenarios for comparison
        
        Args:
            base_investment_type: The original investment type
            amount: Investment amount
            years: Investment period in years
            expected_return: Expected annual return (decimal)
            
        Returns:
            list: Alternative scenarios with their outcomes
        """
        alternatives = []
        
        # Define alternative investment types with different returns
        alternative_investments = {
            "Conservative Bond Portfolio": expected_return * 0.5,
            "Balanced Fund": expected_return * 0.8,
            "Aggressive Growth Fund": expected_return * 1.2,
            "High-Risk Sector ETF": expected_return * 1.5
        }
        
        for alt_type, alt_return in alternative_investments.items():
            # Skip if it's the same as the original
            if alt_type.lower() == base_investment_type.lower():
                continue
                
            future_value = amount * (1 + alt_return) ** years
            gain = future_value - amount
            
            alternatives.append({
                "investment_type": alt_type,
                "annual_return": alt_return * 100,  # Convert to percentage
                "initial_amount": amount,
                "years": years,
                "final_value": future_value,
                "gain": gain,
                "gain_percentage": gain/amount*100
            })
        
        return alternatives
    
    def create_decision_tree_comic(self, amount, years, decisions):
        """
        Create a comic showing different investment decisions and their outcomes
        
        Args:
            amount: Initial investment amount
            years: Investment period
            decisions: List of decision points and options
            
        Returns:
            dict: Path to the generated comic and description
        """
        # This is a placeholder for a more complex implementation
        # In a real app, this would generate a branching comic with multiple decision points
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comic_filename = f"decision_tree_comic_{timestamp}.jpg"
        comic_path = os.path.join(self.output_dir, comic_filename)
        
        # Example decisions format:
        # decisions = [
        #     {"question": "Risk level?", "options": ["Low", "Medium", "High"]},
        #     {"question": "Investment horizon?", "options": ["Short", "Medium", "Long"]}
        # ]
        
        # For now, just create a simple comic showing the first decision
        if decisions and len(decisions) > 0:
            first_decision = decisions[0]
            
            frames = []
            
            # Title frame
            title_text = f"Your Investment Journey: Starting with ${amount:,.2f} for {years} years"
            frames.append(self._generate_simple_comic_frame(title_text))
            
            # Decision frame
            decision_text = f"Decision: {first_decision['question']}"
            frames.append(self._generate_simple_comic_frame(decision_text))
            
            # Outcome frames for each option
            for option in first_decision.get('options', [])[:3]:  # Limit to 3 options for simplicity
                option_text = f"If you choose: {option}"
                frames.append(self._generate_simple_comic_frame(option_text))
            
            # Combine the frames into a grid (2x2 if 4 frames, otherwise as needed)
            if len(frames) <= 2:
                # Horizontal layout for 1-2 frames
                total_width = sum(frame.width for frame in frames)
                max_height = max(frame.height for frame in frames)
                
                comic = Image.new('RGB', (total_width, max_height))
                x_offset = 0
                
                for frame in frames:
                    comic.paste(frame, (x_offset, 0))
                    x_offset += frame.width
            else:
                # Grid layout for 3+ frames
                rows = (len(frames) + 1) // 2  # Ceiling division
                cols = min(2, len(frames))
                
                frame_width = frames[0].width
                frame_height = frames[0].height
                
                comic = Image.new('RGB', (cols * frame_width, rows * frame_height))
                
                for i, frame in enumerate(frames):
                    row = i // cols
                    col = i % cols
                    comic.paste(frame, (col * frame_width, row * frame_height))
            
            # Save the comic
            comic.save(comic_path)
            
            description = f"A comic showing investment decisions starting with ${amount:,.2f} for {years} years."
            
            return {
                "comic_path": comic_path,
                "description": description
            }
        else:
            # Fallback if no decisions provided
            return self.generate_comic_strip("Diversified Portfolio", amount, years, 0.07)  # 7% default return 