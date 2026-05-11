import urllib.request
import json
import os
from dotenv import load_dotenv

# Load your API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ ERROR: GOOGLE_API_KEY not found in .env file!")
    exit()

print("🔍 Pinging Google's servers for YOUR active models...\n")
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

try:
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())
        
        print("--- 🧠 AVAILABLE EMBEDDING MODELS ---")
        for m in data.get('models', []):
            if 'embedContent' in m.get('supportedGenerationMethods', []):
                print(m['name'])
                
        print("\n--- 💬 AVAILABLE CHAT MODELS ---")
        for m in data.get('models', []):
            if 'generateContent' in m.get('supportedGenerationMethods', []):
                print(m['name'])
                
except Exception as e:
    print(f"API Error: {e}")