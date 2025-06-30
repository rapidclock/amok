from openai import OpenAI
import requests
import base64


# Modify OpenAI's API key and API base to use vLLM's API server.
def run():
    openai_api_key = "sk-xxxxxxx"
    openai_api_base = "http://localhost:1234/v1"

    TEMP = 0.15
    MAX_TOK = 8000

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    # models = client.models.list()
    # model = models.data[0].id

    def read_file(file_path: str) -> str:
        """Read and return the contents of a file."""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    model_id = "microsoft/phi-4-reasoning-plus"
    SYSTEM_PROMPT = read_file("./prompts/SYSTEM_PROMPT_01.txt")

    image_url = "https://huggingface.co/datasets/patrickvonplaten/random_img/resolve/main/europe.png"

    # Fetch the image and encode it in base64
    response = requests.get(image_url)
    response.raise_for_status()
    base64_image = base64.b64encode(response.content).decode("utf-8")
    image_data_url = f"data:image/png;base64,{base64_image}"

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_population",
                "description": "Get the up-to-date population of a given country.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "country": {
                            "type": "string",
                            "description": "The country to find the population of.",
                        },
                        "unit": {
                            "type": "string",
                            "description": "The unit for the population.",
                            "enum": ["millions", "thousands"],
                        },
                    },
                    "required": ["country", "unit"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rewrite",
                "description": "Rewrite a given text for improved clarity",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The input text to rewrite",
                        }
                    },
                },
            },
        },
    ]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Could you please make the below article more concise?\n\nOpenAI is an artificial intelligence research laboratory consisting of the non-profit OpenAI Incorporated and its for-profit subsidiary corporation OpenAI Limited Partnership.",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "bbc5b7ede",
                    "type": "function",
                    "function": {
                        "name": "rewrite",
                        "arguments": '{"text": "OpenAI is an artificial intelligence research laboratory consisting of the non-profit OpenAI Incorporated and its for-profit subsidiary corporation OpenAI Limited Partnership."}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"action":"rewrite","outcome":"OpenAI is a FOR-profit company."}',
            "tool_call_id": "bbc5b7ede",
            "name": "rewrite",
        },
        {
            "role": "assistant",
            "content": "---\n\nOpenAI is a FOR-profit company.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Can you tell me what is the biggest country depicted on the map?",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url,
                    },
                },
            ],
        },
    ]

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=TEMP,
        max_tokens=MAX_TOK,
        tools=tools,
        tool_choice="auto",
    )

    assistant_message = response.choices[0].message.content
    print(assistant_message)
    # The biggest country depicted on the map is Russia.

    messages.extend(
        [
            {"role": "assistant", "content": assistant_message},
            {
                "role": "user",
                "content": "What is the population of that country in millions?",
            },
        ]
    )

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=TEMP,
        max_tokens=MAX_TOK,
        tools=tools,
        tool_choice="auto",
    )

    print(response.choices[0].message.tool_calls)
    # [ChatCompletionMessageToolCall(id='3e92V6Vfo', function=Function(arguments='{"country": "Russia", "unit": "millions"}', name='get_current_population'), type='function')]


if __name__ == "__main__":
    run()
