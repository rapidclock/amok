"""A local LLM server for testing."""

# from openai import OpenAI
from amok import ActionAgent, ActionAgentSettings


def trial_action_agent():
    settings = ActionAgentSettings(
        base_url="http://localhost:1234/v1",
        model="google/gemma-3-1b",
        api_key="sk-lm-studio",
        temperature=0.7,
        max_tokens=10000,
        ssl_verify=True,
        thinking_mode=False,
        description="A poets take on things.",
        commands=[
            "You will always answer in rhymes.",
            "You will speak like a poet.",
            "You will use metaphors and similes.",
            "You will use alliteration and assonance.",
        ],
    )
    agent = ActionAgent(settings)
    body = "Tell me about the beauty of nature."
    sys, user = agent.generate_prompts(body)
    print(f"System Prompt:\n{sys}\n")
    print(f"User Prompt:\n{user}\n")
    response = agent.run(body)
    print(f"Thought: {response.thought}")
    print(f"Response: {response.response}")


# def main():
#     """Main function to run the OpenAI client."""
#     ai_client = OpenAI(base_url="http://localhost:1234/v1", api_key="sk-lm-studio")
#     # model = "microsoft/phi-4-mini-reasoning"
#     # model = "mistral-small-3.2-24b-instruct-2506"
#     model = "microsoft/phi-4-reasoning-plus"
#     completion = ai_client.chat.completions.create(
#         model="microsoft/phi-4-mini-reasoning",
#         messages=[
#             {"role": "system", "content": "Always answer in rhymes."},
#             {"role": "user", "content": "Introduce yourself."},
#         ],
#         temperature=0.7,
#         max_tokens=-1,
#         stream=False,
#     )
#     print(completion)
#     print(completion.choices[0].message)


if __name__ == "__main__":
    # main()
    trial_action_agent()
