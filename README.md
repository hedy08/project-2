# project-2
My second repository for project 2. 

import os
import google.generativeai as genai
from dotenv import load_dotenv

def main():
    """
    Main function to run the AI Brain interaction.
    """
    # 1. Load environment variables from .env file
    load_dotenv()

    # 2. Configure the Gemini API
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"Error configuring the API: {e} ❌")
        return

    # 3. Initialize the generative model
    # We use 'gemini-pro' which is a powerful and versatile model for text tasks.
    model = genai.GenerativeModel('gemini-pro')

    print("🤖 Your AI Brain is ready. Ask me anything!")
    print("Type 'exit' or 'quit' to end the session.")

    # 4. Start an interactive chat session
    while True:
        user_question = input("\nYour Question: ")
        
        if user_question.lower() in ['exit', 'quit']:
            print("Session ended. Goodbye! 👋")
            break
            
        if not user_question:
            print("Please enter a question.")
            continue

        try:
            # 5. Send the question to the model and get a response
            print("\n🧠 Thinking...")
            response = model.generate_content(user_question)
            
            # 6. Print the AI's response
            print("\n💡 AI Response:")
            print(response.text)
            
        except Exception as e:
            print(f"An error occurred while generating a response: {e} ❌")

if __name__ == "__main__":
    main()


(venv) C:\Open impact lab\Project 2>python main.py
🤖 Your AI Brain is ready. Ask me anything!
Type 'exit' or 'quit' to end the session.


import os
import requests
import json

# --- Configuration ---
# IMPORTANT: For security, use environment variables to store your API keys.
# How to set them:
# In your terminal (Linux/macOS): export GOOGLE_API_KEY="your_api_key_here"
# In your terminal (Windows): set GOOGLE_API_KEY="your_api_key_here"
# You'll also need to set GOOGLE_CSE_ID="your_search_engine_id_here"
API_KEY = os.getenv("GOOGLE_API_KEY")
SEARCH_ENGINE_ID = os.getenv("GOOGLE_CSE_ID")

# Fallback values if environment variables are not set (for quick testing only)
if not API_KEY or not SEARCH_ENGINE_ID:
    print("⚠️ Warning: API_KEY or SEARCH_ENGINE_ID not found in environment variables.")
    print("Using placeholder values. Please set them for the script to work.")
    # Replace these with your actual key and ID for a quick test, but avoid committing them.
    API_KEY = "YOUR_API_KEY_HERE" 
    SEARCH_ENGINE_ID = "YOUR_SEARCH_ENGINE_ID_HERE"

def google_search(query: str, num_results: int = 5):
    """
    Performs a Google search using the Custom Search JSON API.
    
    Args:
        query (str): The search term.
        num_results (int): The number of results to return.
        
    Returns:
        A list of search result items or None if the request fails.
    """
    if API_KEY == "YOUR_API_KEY_HERE" or SEARCH_ENGINE_ID == "YOUR_SEARCH_ENGINE_ID_HERE":
        print("❌ Error: Please replace placeholder API_KEY and SEARCH_ENGINE_ID.")
        return None

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'key': API_KEY,
        'cx': SEARCH_ENGINE_ID,
        'q': query,
        'num': num_results
    }
    
    try:
        print(f"⚡ Performing web search for: '{query}'...")
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        
        search_results = response.json()
        return search_results.get('items', [])

    except requests.exceptions.RequestException as e:
        print(f"❌ An error occurred during the web search: {e}")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return None

# --- Main execution ---
if __name__ == "__main__":
    search_query = "What is tool use in large language models?"
    results = google_search(search_query)
    
    if results:
        print(f"\n✅ Found {len(results)} results:\n")
        for i, item in enumerate(results, 1):
            print(f"--- Result {i} ---")
            print(f"Title: {item.get('title')}")
            print(f"URL: {item.get('link')}")
            print(f"Snippet: {item.get('snippet')}\n")
    else:
        print("No search results found or an error occurred.")

C:\Users\Hedy Kuo>python "C:\Open impact lab\Project 2 Task 2\web_search.py"
⚡ Performing web search for: 'What is tool use in large language models?'...

✅ Found 5 results:

--- Result 1 ---
Title: ToolLLM: Facilitating Large Language Models to Master 16000+ ...
URL: https://arxiv.org/abs/2307.16789
Snippet: Jul 31, 2023 ... Despite the advancements of open-source large language models (LLMs), e.g., LLaMA, they remain significantly limited in tool-use capabilities, ...

--- Result 2 ---
Title: Agentic Design Patterns Part 3, Tool Use How large language ...
URL: https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-3-tool-use/
Snippet: Apr 3, 2024 ... Early in the history of LLMs, before widespread availability of large multimodal models (LMMs) like LLaVa, GPT-4V, and Gemini, LLMs could not ...

--- Result 3 ---
Title: Tool Learning with Large Language Models: A Survey
URL: https://arxiv.org/abs/2405.17935
Snippet: May 28, 2024 ... In this survey, we focus on reviewing existing literature from the two primary aspects (1) why tool learning is beneficial and (2) how tool learning is ...

--- Result 4 ---
Title: Adaptive Tool Use in Large Language Models with Meta-Cognition ...
URL: https://aclanthology.org/2025.acl-long.655/
Snippet: While existing research expands LLMs access to diverse tools (e.g., program interpreters, search engines, calculators), the necessity of using these tools is ...

--- Result 5 ---
Title: ART: Automatic multi-step reasoning and tool-use for large ...
URL: https://arxiv.org/abs/2303.09014
Snippet: Mar 16, 2023 ... Abstract:Large language models (LLMs) can perform complex reasoning in few- and zero-shot settings by generating intermediate chain of ...
