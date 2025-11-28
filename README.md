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
