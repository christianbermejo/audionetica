import os
import openai

class Translator:
    def __init__(self, source_lang="en", target_lang="ko"):
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        self.client = openai.OpenAI()
        self.source_lang = source_lang
        self.target_lang = target_lang
        
    def translate(self, text):
        if not text.strip():
            return ""
            
        # Use the most cost-effective model
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",  # Most cost-effective model for this task
            messages=[
                {"role": "system", "content": f"Translate the following text from {self.source_lang} to {self.target_lang}. Respond with only the translation."},
                {"role": "user", "content": text}
            ],
            temperature=0.3,  # Lower temperature for more consistent translations
            max_tokens=150    # Limit token usage to control costs
        )
        
        return response.choices[0].message.content.strip()