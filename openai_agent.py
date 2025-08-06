import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_intent_from_prompt(prompt: str) -> str:
    system = "You are an assistant that maps user queries to backend functions."
    examples = {
        "What is the 7-day average gold price?": "get_7_day_avg",
        "Give forecast for gold price next month.": "forecast_next_month"
    }

    messages = [{"role": "system", "content": system}]
    for q, a in examples.items():
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0
    )
    return response.choices[0].message.content.strip()
