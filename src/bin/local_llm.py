"""A local LLM server for testing."""

# from openai import OpenAI
from amok import ActionAgent, ActionAgentSettings, OptionAgentSettings, OptionAgent
from amok.utils import chain_agents


def trial_action_agent():
    settings = ActionAgentSettings(
        base_url="http://localhost:1234/v1",
        model="google/gemma-3-4b",
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


def trail_option_agent():
    """A trial function for the OptionAgent."""
    settings = OptionAgentSettings(
        base_url="http://localhost:1234/v1",
        model="google/gemma-3-4b",
        # model="microsoft/phi-4-mini-reasoning",
        api_key="sk-lm-studio",
        temperature=0.7,
        max_tokens=10000,
        ssl_verify=True,
        thinking_mode=True,
        description="You need to choose if the given belongs to any of the categories",
        commands=[
            "You must consider the general concept of the given entity.",
            "Make your best guess.",
            "You need to be as accurate as possible.",
        ],
        options=[
            "Belongs to this earth.",
            "Belongs to the solar system.",
            "Belongs beyond.",
            "Does not exist!",
        ],
    )
    agent = OptionAgent(settings)
    body = "A car"
    sys, user = agent.generate_prompts(body)
    print(f"System Prompt:\n{sys}\n")
    print(f"User Prompt:\n{user}\n")
    result = agent.run(body)
    print(f"Thoughts: {result.thought}")
    print(f"Response: {result.response}")
    print(f"Option: {result.option_index}")


def get_option_agent():
    settings = OptionAgentSettings(
        base_url="http://localhost:1234/v1",
        model="google/gemma-3-12b",
        # model="microsoft/phi-4-mini-reasoning",
        api_key="sk-lm-studio",
        temperature=0.7,
        max_tokens=10000,
        ssl_verify=True,
        thinking_mode=True,
        description="You need to choose if the given belongs to any of the categories",
        commands=[
            "You must consider the general concept of the given entity.",
            "Make your best guess.",
            "You need to be as accurate as possible.",
        ],
        options=[
            "Belongs to this earth.",
            "Belongs to the solar system.",
            "Belongs beyond.",
            "Does not exist!",
        ],
    )
    agent = OptionAgent(settings)
    return agent


def get_discerning_agent():
    settings = ActionAgentSettings(
        base_url="http://localhost:1234/v1",
        model="google/gemma-3-12b",
        # model="microsoft/phi-4-mini-reasoning",
        api_key="sk-lm-studio",
        temperature=0.7,
        max_tokens=10000,
        ssl_verify=True,
        thinking_mode=True,
        description="You Need to Help determine if The commands have been followed.",
        commands=[
            "You are to reply with either true or false.",
            "Examine the body and tell me if the output is purely a whole number greater than or equal to 0.",
            "If there is any additional text with or without formatting or explanations, you will reply with false.",
        ],
    )
    agent = ActionAgent(settings)
    return agent


def combo_pack():
    option_agent = get_option_agent()
    discerning_agent = get_discerning_agent()
    body = "Neptune"
    # response = option_agent.run(body)
    # print(f"Option Agent Response: {response}")
    # discerning_response = discerning_agent.run(response.response)
    # print(f"Discerning Agent Response: {discerning_response.response}")
    final_resp, list_of_results = chain_agents(body, option_agent, discerning_agent)
    print(f"Final Response: {final_resp}")
    print(f"List of Results: {list_of_results}")


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
    # trial_action_agent()
    # trail_option_agent()
    combo_pack()
