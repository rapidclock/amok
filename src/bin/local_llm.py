"""A local LLM server for testing."""

from openai import OpenAI


def main():
    """Main function to run the OpenAI client."""
    ai_client = OpenAI(base_url="http://localhost:1234/v1", api_key="sk-lm-studio")
    completion = ai_client.chat.completions.create(
        model="microsoft/phi-4-mini-reasoning",
        messages=[
            {"role": "system", "content": "Always answer in rhymes."},
            {"role": "user", "content": "Introduce yourself."},
        ],
        temperature=0.7,
        max_tokens=-1,
        stream=False,
    )
    print(completion)
    print(completion.choices[0].message)


if __name__ == "__main__":
    main()
