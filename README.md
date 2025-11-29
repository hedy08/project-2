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



# Filename: research_bot.py

import os
import google.generativeai as genai
from googleapiclient.discovery import build

# --- Configuration ---
# IMPORTANT: Set these as environment variables for security.
# Example in terminal:
# export GOOGLE_API_KEY="your_gemini_api_key"
# export CSE_ID="your_custom_search_engine_id"
# export SEARCH_API_KEY="your_google_search_api_key"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CSE_ID = os.getenv("CSE_ID")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")

# --- Tool Definition: Web Search ---
# This function is our "tool" that the AI agent can decide to use.

def web_search(query: str) -> str:
    """
    Performs a web search using the Google Custom Search API and returns the top results.

    Args:
        query (str): The search query.

    Returns:
        str: A formatted string of search results including title, link, and snippet.
    """
    print(f"⚡ Performing web search for: '{query}'")
    try:
        service = build("customsearch", "v1", developerKey=SEARCH_API_KEY)
        # Execute the search
        res = service.cse().list(q=query, cx=CSE_ID, num=3).execute() # Get top 3 results

        if 'items' not in res:
            return "No relevant search results found."

        # Format the results for the LLM
        search_results = []
        for item in res['items']:
            search_results.append(
                f"Title: {item['title']}\n"
                f"Link: {item['link']}\n"
                f"Snippet: {item['snippet']}\n"
            )
        
        return "\n---\n".join(search_results)

    except Exception as e:
        return f"An error occurred during search: {e}"

# --- Agent Orchestration ---

def run_agent():
    """
    Initializes and runs the main loop for the AI research agent.
    """
    # 1. Configure the Gemini Model
    if not GOOGLE_API_KEY:
        print("⚠️ ERROR: GOOGLE_API_KEY environment variable not set.")
        return
    genai.configure(api_key=GOOGLE_API_KEY)

    # 2. Instantiate the model with the tool
    # We tell the model about the 'web_search' tool it can use.
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        tools=[web_search] # Pass the function directly
    )

    # 3. Start the conversation
    chat = model.start_chat()
    print("🚀 AI Research Bot is active. Ask me anything! (Type 'exit' to quit)")

    while True:
        # 4. Get user input
        user_prompt = input("You: ")
        if user_prompt.lower() == 'exit':
            print("Bot shutting down. Goodbye!")
            break

        # 5. Send prompt to the model
        response = chat.send_message(user_prompt)
        
        # 6. Check if the model wants to use a tool
        try:
            function_call = response.candidates[0].content.parts[0].function_call
            
            # If the model makes a function call, execute it
            if function_call.name == "web_search":
                # Get the query the model decided on
                query = function_call.args['query']
                
                # Execute the tool
                tool_output = web_search(query=query)
                
                # Send the tool's output back to the model
                response = chat.send_message(
                    part=genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name='web_search',
                            response={'result': tool_output}
                        )
                    )
                )

        except (AttributeError, IndexError):
            # No function call was made, the model responded directly.
            pass
            
        # 7. Print the final answer
        print(f"AI Bot: {response.text}\n")


if __name__ == "__main__":
    if not all([CSE_ID, SEARCH_API_KEY]):
        print("⚠️ ERROR: CSE_ID or SEARCH_API_KEY environment variables not set.")
    else:
        run_agent()


# main.py (Corrected and Reverted)

import os
import textwrap
import google.generativeai as genai
import asyncio

from google.adk.agents import Agent
#from adk import tool
# ⬇️ THIS IS THE CORRECT IMPORT STATEMENT
#from google.generativeai import Gemini
#from adk.tools import Tool, tool
#from adk.tools import Tool
#from google_custom_search import CustomSearch
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
# --- Configuration ---
try:
    gemini_api_key = os.environ["GEMINI_API_KEY"]
    google_api_key = os.environ["GOOGLE_API_KEY"]
    google_cse_id = os.environ["GOOGLE_CSE_ID"]
except KeyError:
    print("🔴 ERROR: Please set the required environment variables: GEMINI_API_KEY, GOOGLE_API_KEY, GOOGLE_CSE_ID")
    exit()

# --- Tool Definition ---
#@tool
def search(query: str) -> str:
    """
    Searches the web for the given query using Google Custom Search
    and returns the most relevant results.
    """
    print(f"⚡ Performing web search for: '{query}'")
    search_results = CustomSearch(api_key=google_api_key, cse_id=google_cse_id).search(query)
    
    snippets = [item.get("snippet", "") for item in search_results.get("items", [])]
    formatted_results = "\n".join(f"- {snippet}" for snippet in snippets if snippet)
    
    if not formatted_results:
        return "No relevant information found."
        
    return f"Here are the search results for '{query}':\n{formatted_results}"

# --- Main Application Logic ---
def main():
    """Main function to run the AI Web Explorer bot."""
    
    print("="*50)
    print("🤖 Welcome to the AI Web Explorer Bot! 🤖")
    print("="*50)
    print("Ask me any question, and I'll search the web for an answer.")
    print("Type 'exit' or 'quit' to end the session.\n")

    # ⬇️ THIS IS THE CORRECT AGENT INITIALIZATION
    #Agent = genai(api_key=gemini_api_key)
    #Agent = genai(api_key=gemini_api_key, tools=[search])
    while True:
        user_question = input("❓ Your question: ")

        if user_question.lower() in ['exit', 'quit']:
            print("\n👋 Goodbye!")
            break

        if not user_question:
            print("⚠️ Please enter a question.")
            continue
            
        print("\n🧠 Thinking... I'm searching the web for answers...\n")

        try:
            response = Agent.chat(f"Please provide a comprehensive and clear answer to the following question, based on the search results: '{user_question}'")
            #response = Agent.chat(f"Please provide a comprehensive and clear answer to the following question, based on the search results: '{user_question}'")
            print("="*50)
            print("✅ Here is your answer:")
            print("="*50)
            
            wrapped_text = textwrap.fill(response, width=80)
            print(wrapped_text)
            print("\n" + "="*50 + "\n")

        except Exception as e:
            print(f"🔴 An error occurred: {e}")
            print("Please check your API keys and network connection.\n")


if __name__ == "__main__":
    main()

# 🤖 AI Web Explorer: Your Smart Research Bot

This project is an intelligent AI agent that acts as a research companion. When you ask a question, the bot automatically searches the web, processes the results, and provides a clear, concise summary.

This project was built as part of the OpenImpactLab "[Phase 1 - Project 2] AI Web Explorer" curriculum.

---

## ✨ Features

-   **Natural Language Questions:** Ask questions in plain English.
-   **Automated Web Search:** Leverages the Google Custom Search API to find relevant information.
-   **AI-Powered Summarization:** Uses Google's Gemini model to understand the search results and generate a coherent answer.
-   **Interactive CLI:** A clean and easy-to-use command-line interface.

---

## 🔧 Prerequisites

-   Python 3.8 or higher
-   Access to Google Cloud Platform to obtain API keys.

---

## ⚙️ Setup & Installation

1.  **Clone the Repository:**
    ```bash
    git clone <https://github.com/hedy08/project-2.git>
    cd <hedy08>
    ```

2.  **Install Dependencies:**
    Install the required Python libraries using pip.
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file. See below.)*

3.  **Set Up API Keys:**
    This project requires three API keys/IDs. You must set them as environment variables for security.

    -   `GEMINI_API_KEY`: Your API key for the Google Gemini API.
    -   `GOOGLE_API_KEY`: Your API key for the Google Custom Search JSON API.
    -   `GOOGLE_CSE_ID`: Your Programmable Search Engine ID.

    **On macOS/Linux:**
    ```bash
    export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    export GOOGLE_CSE_ID="YOUR_CSE_ID"
    ```

    **On Windows (Command Prompt):**
    ```bash
    set GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    set GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    set GOOGLE_CSE_ID="YOUR_CSE_ID"
    ```
    *Replace `"YOUR_..._KEY"` with your actual credentials.*

---

## 🚀 Usage

Once the setup is complete, run the bot from your terminal:

```bash
python main.py

C:\Open impact lab\Project 2>python.exe main.py
==================================================
🤖 Welcome to the AI Web Explorer Bot! 🤖
==================================================
Ask me any question, and I'll search the web for an answer.
Type 'exit' or 'quit' to end the session.

❓ Your question: he

🧠 Thinking... I'm searching the web for answers...

🔴 An error occurred: chat
Please check your API keys and network connection.

❓ Your question: exit

C:\Users\Hedy Kuo>set CSE_ID=35023dfc921264f57

C:\Users\Hedy Kuo>python "C:\Open impact lab\Project 2 Task 3\research_bot.py"
🚀 AI Research Bot is active. Ask me anything! (Type 'exit' to quit)
You: What is the latest news about Taiwan?
Traceback (most recent call last):
