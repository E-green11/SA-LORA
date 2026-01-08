from openai import OpenAI
import os

# Configuration: read sensitive values from environment variables to avoid embedding secrets.
MY_API_KEY = os.getenv("MY_API_KEY", "")
MY_BASE_URL = os.getenv("MY_BASE_URL", "https://api.bltcy.ai/v1")

client = OpenAI(api_key=MY_API_KEY, base_url=MY_BASE_URL)

print("Attempting to connect to the model...")

if not MY_API_KEY:
    print("API key not set. Please set the MY_API_KEY environment variable and retry.")
else:
    try:
        # Send a simple test request
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Please reply with 'connection successful'."}],
        )

        # Print the result
        print("Model reply:", response.choices[0].message.content)
        print("✅ Test passed.")

    except Exception as e:
        print("❌ Connection failed. Error:")
        print(e)