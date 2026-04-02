import os
import json
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

app = Flask(__name__)

# Load your full resume data
def load_resume_data():
    with open('resume.json', 'r') as f:
        return json.dumps(json.load(f), indent=2)

resume_context = load_resume_data()

# Global list to store the last 10 messages (approx 5 turns)
chat_history = []

llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    openai_api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/projects')
def projects():
    return render_template('projects.html')

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history

    user_query = request.json.get("message")

    # 1. Format the history into a string for the prompt
    history_text = "\n".join(chat_history)

    prompt = f"""
You are Amos's AI alter-ego. Use the JSON data below to represent him.

RULES:
- Be witty, professional, and use "haha" or "lol" occasionally.
- Reference previous parts of the conversation if the user asks follow-up questions.
- If the user mutes you, don't take it personally! lol

RESUME DATA:
{resume_context}

RECENT CONVERSATION HISTORY:
{history_text}

NEW USER QUESTION:
{user_query}
"""

    response = llm.invoke(prompt)
    bot_reply = response.content

    # 2. Update history (Keep only last 10 items to stay within 5 rounds)
    chat_history.append(f"User: {user_query}")
    chat_history.append(f"AI: {bot_reply}")
    chat_history = chat_history[-10:]

    return jsonify({"response": bot_reply})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)