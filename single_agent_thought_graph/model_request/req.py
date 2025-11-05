import re
import json
from openai import OpenAI
import time
import requests


def extract_json(text: str) -> dict | None:
    json_pattern = re.compile(r'{.*?}', re.DOTALL)

    match = json_pattern.search(text)

    if match:
        json_str = match.group()
        if json_str.strip():  # Check if the JSON string is not empty
            try:
                json_data = json.loads(json_str)
                return json_data
            except json.JSONDecodeError as e:
                return None

    return None


def validate_prediction(prediction: dict, class_labels: set) -> bool:
    valid_values = {0, 1}  # Set of valid values

    for label in class_labels:
        if label not in prediction:
            return False

        try:
            value = int(prediction[label])
        except (ValueError, TypeError):
            return False

        if value not in valid_values:
            return False

    return True


def openai_request(system_content: str, prompt: str, api_key: str, model: str = "gpt-4o-mini", temperature: float = 0.2, max_tokens: int = 500) -> str:
    client = OpenAI(api_key=api_key)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    return completion.choices[0].message.content


def togetherai_request(system_content: str, prompt: str, api_key: str, model: str, temperature: float = 0.2, max_tokens: int = 500) -> str | None:
    url = "https://api.together.xyz/v1/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Bearer {api_key}"
    }

    for _ in range(5):
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 503 or response.status_code == 500:
            time.sleep(1)
        else:
            data = response.json()

            if 'error' in data:
                print(data)
                return None

            return data['choices'][0]['message']['content']

    return None
