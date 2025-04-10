import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import re
import emoji
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (uncomment for first run)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

class EmotionAnalyzer:
    """
    A class to detect emotions in text using a pre-trained transformer model
    or a simple rule-based approach as fallback
    """
    
    # Emotions we're interested in detecting
    EMOTIONS = {
        'anxious': ['anxious', 'worried', 'nervous', 'uneasy', 'fearful', 'stressed', 'panicked', 'tense', 'scared', 'afraid', 'concerned', 'uncertain', 'doubtful', 'hesitant'],
        'excited': ['excited', 'thrilled', 'enthusiastic', 'eager', 'happy', 'joyful', 'elated', 'ecstatic', 'delighted', 'pleased'],
        'confident': ['confident', 'assured', 'certain', 'positive', 'optimistic', 'sure', 'determined', 'resolute', 'steadfast'],
        'frustrated': ['frustrated', 'annoyed', 'irritated', 'exasperated', 'upset', 'agitated', 'troubled', 'bothered', 'dissatisfied', 'displeased'],
        'confused': ['confused', 'puzzled', 'perplexed', 'bewildered', 'unsure', 'lost', 'disoriented', 'uncertain', 'unclear', 'baffled'],
        'neutral': ['neutral', 'balanced', 'impartial', 'objective', 'calm', 'indifferent', 'moderate', 'fair', 'unbiased', 'dispassionate']
    }
    
    # Emoji to emotion mapping
    EMOJI_MAP = {
        '😨': 'anxious', '😰': 'anxious', '😱': 'anxious', '😢': 'anxious', '😟': 'anxious', '😧': 'anxious', '😬': 'anxious',
        '😀': 'excited', '😁': 'excited', '😃': 'excited', '😄': 'excited', '🤩': 'excited', '🎉': 'excited', '🙌': 'excited',
        '👍': 'confident', '💪': 'confident', '🚀': 'confident', '😎': 'confident', '👏': 'confident',
        '😠': 'frustrated', '😡': 'frustrated', '😤': 'frustrated', '😒': 'frustrated', '👎': 'frustrated',
        '🤔': 'confused', '😕': 'confused', '❓': 'confused', '❔': 'confused', '😵‍💫': 'confused',
        '😐': 'neutral', '😶': 'neutral', '🤷': 'neutral', '⚖️': 'neutral'
    }
    
    def __init__(self, use_transformer=True, model_name="distilbert-base-uncased", device=None):
        """
        Initialize the emotion analyzer
        
        Args:
            use_transformer: Whether to use transformer model (if available) or rule-based approach
            model_name: Name of the transformer model to use
            device: Device to run the model on (None for auto-detection)
        """
        self.use_transformer = use_transformer
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize lemmatizer and stopwords for text preprocessing
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize transformer model if required
        if self.use_transformer:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.model.to(self.device)
                
                # Map model outputs to our emotion categories
                # Note: This would need to be adjusted based on the actual model used
                self.label_map = {
                    0: 'neutral',
                    1: 'anxious',    # Assuming this maps to "negative" or similar in model output
                    2: 'excited',    # Assuming this maps to "positive" or similar in model output
                    3: 'confused',
                    4: 'frustrated',
                    5: 'confident'
                }
                
            except Exception as e:
                print(f"Error loading transformer model: {e}")
                print("Falling back to rule-based approach")
                self.use_transformer = False
    
    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Extract emojis and store them
        self.emojis_in_text = []
        for char in text:
            if char in emoji.UNICODE_EMOJI['en']:
                self.emojis_in_text.append(char)
        
        # Replace emojis with space
        text = emoji.replace_emoji(text, replace='')
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token) for token in tokens 
            if token not in self.stop_words
        ]
        
        return ' '.join(processed_tokens), self.emojis_in_text
    
    def rule_based_analysis(self, text, emojis):
        """
        Analyze emotions in text using rule-based approach
        
        This is a simple implementation that:
        1. Counts occurrences of emotion words in the text
        2. Analyzes emojis
        3. Weighs these factors to determine the overall emotion
        """
        # Initialize emotion scores
        emotion_scores = {emotion: 0.0 for emotion in self.EMOTIONS}
        
        # Count emotion words
        words = text.split()
        for word in words:
            for emotion, keywords in self.EMOTIONS.items():
                if word in keywords:
                    emotion_scores[emotion] += 1.0
        
        # Analyze emojis
        for emoji_char in emojis:
            if emoji_char in self.EMOJI_MAP:
                emotion = self.EMOJI_MAP[emoji_char]
                emotion_scores[emotion] += 2.0  # Emojis have higher weight
        
        # If no signals found, default to neutral
        if sum(emotion_scores.values()) == 0:
            emotion_scores['neutral'] = 1.0
        
        # Determine the dominant emotion
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        
        # Calculate confidence based on relative strength of the signal
        total_score = sum(emotion_scores.values())
        confidence = dominant_emotion[1] / total_score if total_score > 0 else 0.5
        
        return dominant_emotion[0], confidence
    
    def transformer_analysis(self, text):
        """Analyze emotions using the transformer model"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Get probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1)
        
        predicted_label = predictions.item()
        confidence = probs[0][predicted_label].item()
        
        # Map the label to our emotion categories
        emotion = self.label_map.get(predicted_label, 'neutral')
        
        return emotion, confidence
    
    def analyze(self, text):
        """
        Analyze the emotions in the given text
        
        Args:
            text: The text to analyze
            
        Returns:
            tuple: (emotion, confidence)
        """
        if not text:
            return 'neutral', 0.5
        
        # Preprocess the text
        processed_text, emojis = self.preprocess_text(text)
        
        # Use transformer if available, otherwise use rule-based approach
        if self.use_transformer:
            try:
                return self.transformer_analysis(processed_text)
            except Exception as e:
                print(f"Error in transformer analysis: {e}")
                print("Falling back to rule-based approach")
                return self.rule_based_analysis(processed_text, emojis)
        else:
            return self.rule_based_analysis(processed_text, emojis)
    
    def get_investment_advice(self, emotion, risk_tolerance="Medium", investment_goal="Wealth Building"):
        """
        Get personalized investment advice based on the detected emotion
        
        Args:
            emotion: The detected emotion
            risk_tolerance: The user's risk tolerance
            investment_goal: The user's investment goal
            
        Returns:
            str: Personalized investment advice
        """
        advice = ""
        
        # Emotion-specific advice
        if emotion == 'anxious':
            advice += "I notice you're feeling anxious about investing, which is completely understandable. "
            advice += "Markets can be volatile, but remember that investing is a long-term journey. "
            advice += "Consider starting with more conservative investments and gradually increasing your exposure as you become more comfortable. "
            advice += "Dollar-cost averaging (investing fixed amounts regularly) can also help reduce anxiety about market timing."
            
        elif emotion == 'excited':
            advice += "Your enthusiasm about investing is great to see! "
            advice += "While excitement can be motivating, make sure to balance it with thoughtful research. "
            advice += "Consider channeling that excitement into building a diversified portfolio rather than concentrating on a single 'exciting' investment. "
            advice += "Remember that consistent, disciplined investing often outperforms chasing the latest hot stock."
            
        elif emotion == 'confident':
            advice += "Your confidence is a great asset for investing. "
            advice += "Just make sure it's paired with thorough research and a clear strategy. "
            advice += "Even the most confident investors benefit from diversification and regular portfolio reviews. "
            advice += "Consider challenging your investment theses to ensure they're based on solid analysis."
            
        elif emotion == 'frustrated':
            advice += "Investing can certainly be frustrating at times. "
            advice += "If you're feeling frustrated with your investments, it might be a good time to step back and review your overall strategy. "
            advice += "Make sure your portfolio aligns with your goals and time horizon. "
            advice += "Sometimes, simplifying your investment approach can reduce frustration and improve results."
            
        elif emotion == 'confused':
            advice += "Investment concepts can be complex, and it's normal to feel confused sometimes. "
            advice += "Consider focusing on understanding the basics before moving to more complex strategies. "
            advice += "Index funds or target-date funds can be good options while you're building your knowledge. "
            advice += "Don't hesitate to ask questions or seek education from reputable sources."
            
        else:  # neutral or any other emotion
            advice += "Based on your balanced perspective, you're in a good position to make thoughtful investment decisions. "
            advice += "Focus on aligning your investments with your long-term goals and risk tolerance. "
            advice += "Regular reviews of your portfolio and adjustment as needed will help you stay on track."
        
        # Add risk tolerance and goal-specific advice
        if risk_tolerance in ["Very Low", "Low"]:
            advice += "\n\nWith your conservative risk profile, consider emphasizing bonds, dividend stocks, and other income-generating investments."
        elif risk_tolerance == "Medium":
            advice += "\n\nWith your balanced risk profile, a mix of growth and income investments may be appropriate."
        else:  # High or Very High
            advice += "\n\nWith your aggressive risk profile, you might consider allocating more to growth stocks and emerging markets, while maintaining some diversification."
            
        if investment_goal == "Retirement":
            advice += " For retirement planning, consistency and tax-advantaged accounts like IRAs or 401(k)s are key."
        elif investment_goal == "House Purchase":
            advice += " When saving for a house, consider the timeframe — shorter horizons may call for more conservative allocations."
        elif investment_goal == "Education":
            advice += " For education funding, 529 plans or education-specific savings vehicles offer tax advantages worth exploring."
            
        return advice
        
    def save_model(self, path):
        """Save the model to the specified path"""
        if not self.use_transformer:
            print("No model to save in rule-based approach")
            return
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the model and tokenizer
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
    def load_model(self, path):
        """Load the model from the specified path"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model directory not found at {path}")
            
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        self.use_transformer = True 