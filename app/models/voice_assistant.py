import os
import speech_recognition as sr
from gtts import gTTS
import tempfile
import time
import uuid
from app.models.emotion_analyzer import EmotionAnalyzer
import re
import threading

class VoiceAssistant:
    """
    A voice assistant that can understand speech input and respond in multiple languages
    """
    
    # Supported languages with their codes for TTS
    SUPPORTED_LANGUAGES = {
        "English": "en",
        "Hindi": "hi",
        "Spanish": "es",
        "French": "fr",
        "Mandarin": "zh-CN",
        "Japanese": "ja",
        "German": "de",
        "Italian": "it",
        "Portuguese": "pt",
        "Russian": "ru",
        "Arabic": "ar"
    }
    
    # Keywords in different languages to detect financial queries
    FINANCIAL_KEYWORDS = {
        "en": ["invest", "stock", "money", "financial", "fund", "market", "portfolio", "retirement", "savings", "dividend", "risk"],
        "hi": ["निवेश", "शेयर", "पैसा", "वित्तीय", "फंड", "बाजार", "पोर्टफोलियो", "सेवानिवृत्ति", "बचत"],
        "es": ["invertir", "acción", "dinero", "financiero", "fondo", "mercado", "cartera", "jubilación", "ahorro"],
        "fr": ["investir", "action", "argent", "financier", "fonds", "marché", "portefeuille", "retraite", "épargne"],
        "zh-CN": ["投资", "股票", "钱", "金融", "基金", "市场", "投资组合", "退休", "储蓄"],
        "ja": ["投資", "株式", "お金", "金融", "ファンド", "市場", "ポートフォリオ", "退職", "貯蓄"]
    }
    
    # Common financial questions and responses in different languages
    COMMON_RESPONSES = {
        "en": {
            "greeting": "Hello! I'm your financial assistant. How can I help you with your investments today?",
            "not_understood": "I'm sorry, I couldn't understand that. Could you please repeat?",
            "thinking": "Let me think about that...",
            "market_status": "The market is showing mixed signals today, with technology stocks performing well but energy stocks facing some pressure.",
            "investment_advice": "For most investors, a diversified portfolio that includes a mix of stocks, bonds, and other assets is recommended.",
            "default": "That's an interesting question about finance. Would you like me to connect you with a financial advisor for more personalized advice?"
        },
        "hi": {
            "greeting": "नमस्ते! मैं आपका वित्तीय सहायक हूँ। आज मैं आपके निवेश में कैसे मदद कर सकता हूँ?",
            "not_understood": "मुझे माफ करें, मुझे वह समझ नहीं आया। क्या आप दोहरा सकते हैं?",
            "thinking": "मुझे इस पर विचार करने दें...",
            "market_status": "आज बाजार में मिश्रित संकेत दिख रहे हैं, प्रौद्योगिकी शेयरों का प्रदर्शन अच्छा है लेकिन ऊर्जा शेयरों को कुछ दबाव का सामना करना पड़ रहा है।",
            "investment_advice": "अधिकांश निवेशकों के लिए, एक विविध पोर्टफोलियो जिसमें शेयरों, बॉन्ड और अन्य संपत्तियों का मिश्रण शामिल है, की सिफारिश की जाती है।",
            "default": "यह वित्त के बारे में एक दिलचस्प सवाल है। क्या आप अधिक वैयक्तिकृत सलाह के लिए मुझे एक वित्तीय सलाहकार से जोड़ना चाहेंगे?"
        },
        # Add other languages as needed
    }
    
    def __init__(self, output_dir=None, preferred_language="English", use_emotion_analysis=True):
        """
        Initialize the voice assistant
        
        Args:
            output_dir: Directory to save audio files (optional)
            preferred_language: Default language to use for responses
            use_emotion_analysis: Whether to analyze emotion in speech
        """
        self.output_dir = output_dir or os.path.join(tempfile.gettempdir(), "voice_assistant")
        self.preferred_language = preferred_language
        self.use_emotion_analysis = use_emotion_analysis
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        
        # Initialize emotion analyzer if needed
        if use_emotion_analysis:
            self.emotion_analyzer = EmotionAnalyzer(use_transformer=False)  # Use rule-based for simplicity
    
    def recognize_speech(self, audio_file=None, language="en-US"):
        """
        Recognize speech from audio file or microphone
        
        Args:
            audio_file: Path to audio file (if None, use microphone)
            language: Language code for speech recognition
            
        Returns:
            str: Recognized text
        """
        try:
            if audio_file:
                # Recognize from file
                with sr.AudioFile(audio_file) as source:
                    audio_data = self.recognizer.record(source)
            else:
                # Recognize from microphone
                with sr.Microphone() as source:
                    print("Listening...")
                    self.recognizer.adjust_for_ambient_noise(source)
                    audio_data = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            # Convert speech to text
            text = self.recognizer.recognize_google(audio_data, language=language)
            return text
        except sr.UnknownValueError:
            return "Speech not understood"
        except sr.RequestError:
            return "Could not request results"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def text_to_speech(self, text, language_code="en", filename=None):
        """
        Convert text to speech
        
        Args:
            text: Text to convert to speech
            language_code: Language code for TTS
            filename: Output filename (if None, generate random name)
            
        Returns:
            str: Path to the generated audio file
        """
        if not filename:
            filename = f"response_{str(uuid.uuid4())[:8]}.mp3"
        
        file_path = os.path.join(self.output_dir, filename)
        
        # Generate speech
        tts = gTTS(text=text, lang=language_code, slow=False)
        tts.save(file_path)
        
        return file_path
    
    def detect_language(self, text):
        """
        Simple language detection based on keywords
        
        In a real implementation, this would use a proper language detection library,
        but for demonstration we'll use a simplified approach.
        
        Args:
            text: Text to detect language for
            
        Returns:
            str: Detected language code
        """
        # Simple language detection heuristics
        # In a real app, use a proper language detection library like langdetect
        
        # Convert to lowercase for matching
        text_lower = text.lower()
        
        # Hindi characters unicode range
        if re.search(r'[\u0900-\u097F]', text):
            return "hi"
        
        # Chinese characters unicode range
        if re.search(r'[\u4e00-\u9fff]', text):
            return "zh-CN"
        
        # Japanese characters unicode range
        if re.search(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]', text):
            return "ja"
        
        # Spanish keywords
        spanish_words = ["hola", "como", "está", "gracias", "por", "favor", "dinero", "invertir"]
        if any(word in text_lower for word in spanish_words):
            return "es"
        
        # French keywords
        french_words = ["bonjour", "comment", "merci", "s'il", "vous", "plaît", "argent", "investir"]
        if any(word in text_lower for word in french_words):
            return "fr"
        
        # Default to English
        return "en"
    
    def is_financial_query(self, text, lang_code="en"):
        """
        Check if the query is related to finance
        
        Args:
            text: Query text
            lang_code: Language code of the text
            
        Returns:
            bool: True if it's a financial query
        """
        text_lower = text.lower()
        
        # Get financial keywords for the language, fall back to English if not available
        keywords = self.FINANCIAL_KEYWORDS.get(lang_code, self.FINANCIAL_KEYWORDS["en"])
        
        # Check if any keyword is in the text
        return any(keyword.lower() in text_lower for keyword in keywords)
    
    def get_response(self, query, lang_code="en"):
        """
        Generate a response to the query
        
        Args:
            query: User query
            lang_code: Language code of the query
            
        Returns:
            str: Response text
        """
        # In a real implementation, this would connect to a more sophisticated
        # language model or AI system. For demo purposes, we'll use templates.
        
        query_lower = query.lower()
        
        # Get response templates for the language, fall back to English if not available
        responses = self.COMMON_RESPONSES.get(lang_code, self.COMMON_RESPONSES["en"])
        
        # Check for specific query patterns
        if any(greeting in query_lower for greeting in ["hello", "hi", "hey", "greetings", "namaste", "hola"]):
            return responses.get("greeting")
        
        if any(keyword in query_lower for keyword in ["market", "today", "stock price", "prices"]):
            return responses.get("market_status")
        
        if any(keyword in query_lower for keyword in ["invest", "portfolio", "strategy", "advice", "recommend"]):
            return responses.get("investment_advice")
        
        # If no specific pattern matched, return default response
        return responses.get("default")
    
    def analyze_emotion_in_text(self, text):
        """
        Analyze the emotion in the text
        
        Args:
            text: Text to analyze
            
        Returns:
            tuple: (emotion, confidence)
        """
        if not self.use_emotion_analysis:
            return None, None
        
        try:
            return self.emotion_analyzer.analyze(text)
        except Exception as e:
            print(f"Error analyzing emotion: {e}")
            return None, None
    
    def adjust_response_for_emotion(self, base_response, emotion):
        """
        Adjust the response based on detected emotion
        
        Args:
            base_response: Original response
            emotion: Detected emotion
            
        Returns:
            str: Adjusted response
        """
        if not emotion or emotion == "neutral":
            return base_response
        
        if emotion == "anxious":
            prefix = "I understand you might be feeling concerned. "
            suffix = " Remember that investing is a long-term journey with ups and downs, and it's normal to have questions."
            return f"{prefix}{base_response}{suffix}"
        
        elif emotion == "excited":
            prefix = "I can sense your enthusiasm! "
            suffix = " It's great to see you engaged with your financial future."
            return f"{prefix}{base_response}{suffix}"
        
        elif emotion == "confused":
            prefix = "I notice you might be feeling a bit uncertain. "
            suffix = " Please feel free to ask for clarification on any financial terms or concepts."
            return f"{prefix}{base_response}{suffix}"
        
        elif emotion == "frustrated":
            prefix = "I understand this might be frustrating. "
            suffix = " Let's break this down step by step to make it clearer."
            return f"{prefix}{base_response}{suffix}"
        
        return base_response
    
    def process_query(self, query=None, audio_file=None):
        """
        Process a user query from text or audio
        
        Args:
            query: Text query (if None, use audio_file or microphone)
            audio_file: Path to audio file (if None and query is None, use microphone)
            
        Returns:
            dict: Response data including text and audio path
        """
        # If no query provided, recognize from audio
        if not query:
            query = self.recognize_speech(audio_file)
            
        if query == "Speech not understood" or not query:
            lang_code = self.SUPPORTED_LANGUAGES.get(self.preferred_language, "en")
            response_text = self.COMMON_RESPONSES.get(lang_code, self.COMMON_RESPONSES["en"]).get("not_understood")
            audio_path = self.text_to_speech(response_text, lang_code)
            return {
                "query": None,
                "response_text": response_text,
                "audio_path": audio_path,
                "language": lang_code,
                "emotion": None
            }
        
        # Detect language
        lang_code = self.detect_language(query)
        
        # Check if it's a financial query
        is_financial = self.is_financial_query(query, lang_code)
        
        # Analyze emotion
        emotion, confidence = self.analyze_emotion_in_text(query) if self.use_emotion_analysis else (None, None)
        
        # Generate response
        base_response = self.get_response(query, lang_code)
        
        # Adjust response based on emotion
        if emotion and self.use_emotion_analysis:
            response_text = self.adjust_response_for_emotion(base_response, emotion)
        else:
            response_text = base_response
        
        # Convert response to speech
        audio_path = self.text_to_speech(response_text, lang_code)
        
        return {
            "query": query,
            "response_text": response_text,
            "audio_path": audio_path,
            "language": lang_code,
            "emotion": emotion,
            "is_financial": is_financial
        }
    
    def start_listening(self, callback=None):
        """
        Start listening for voice input in a separate thread
        
        Args:
            callback: Function to call with response data
            
        Returns:
            thread: The listening thread
        """
        def listen_loop():
            while True:
                try:
                    response_data = self.process_query()
                    if callback:
                        callback(response_data)
                    else:
                        print(f"Response: {response_data['response_text']}")
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error in listening loop: {e}")
                    time.sleep(1)
        
        thread = threading.Thread(target=listen_loop)
        thread.daemon = True
        thread.start()
        
        return thread 