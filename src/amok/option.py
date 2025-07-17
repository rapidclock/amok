"""Option agent for amok."""

from .action import ActionAgent
from .lib import AgentSettings, OptionAgentResponse, OptionAgentSettings
from .utils import surround_with_tags

OPTION_SYSTEM_PROMPT = "\n".join(
    [
        "You are an Option Agent - a specialized AI Agent evaluates and selects the "
        "BEST possible option from a list of options based on user input.",
        "You will be given 3 sections: DESCRIPTION, COMMANDS, OPTIONS, and BODY.",
        "Each Section is surrounded with tags - <SECTION>content</SECTION>",
        "The DESCRIPTION section provides context of the task.",
        "The COMMANDS section contains Very Specific Instructions on how to choose an "
        "option from the OPTIONS section.",
        "The OPTIONS section contains options. You need to choose a single option "
        "based on the content of BODY.",
        "The BODY section contains the user's input based on which you will assess "
        "the best option.",
        "Your task is to provide options based on the instructions in the COMMANDS "
        "section and content of the BODY.",
        "Indexes begin at 0, so the first option is 0, the second is 1, and so on.",
        "The Last option is a Special option that indicates that no option is "
        "appropriate. This is called the 'None' option and will have no option text.",
        "If you believe that none of the options are appropriate, you will "
        "choose the 'None' option.",
        "Each option begins with the index number followed by a colon and the option.",
        "Example: 1 : This is the option",
        "You will ONLY reply with the index of the most appropriate option from the "
        "OPTIONS section.",
        "Example, if the most appropriate option is the second one, ONLY reply with "
        "'2' and nothing else. Not the text of the option, not any other text.",
        "You will NOT mention anything about Description, commands, options or any "
        "of the rules in your response.",
        "Your response will purely be the result of applying the commands to the "
        "description and body and reply with the index to the option most "
        "appropriate in your assessment.",
        "Please ENSURE you respond with ONLY the index of the option chosen",
        "NO EXPLANATIONS!!!!",
        "NO FORMATTING OF ANY KIND!",
        "No other text, NO explanation, no description, just the index.",
    ],
)


class OptionAgent(ActionAgent):
    """An agent that provides options."""

    description: str | None = None
    options: list[str]
    thinking_mode: bool = True

    def __init__(self, settings: OptionAgentSettings) -> None:
        """Initialize the option agent."""
        super().__init__(settings)
        self.options = settings.options if settings.options else []
        # Add the Default "None" option, This is the escape hatch for the agent.
        self.options.append("None")

    @classmethod
    def _get_settings_class(cls) -> type[AgentSettings]:
        return OptionAgentSettings

    def compose_user_prompt(self) -> str:
        """Compose the user prompt."""
        super_user_prompt = super().compose_user_prompt()
        options_str = "\n".join(
            f"{i} : {option}" for i, option in enumerate(self.options)
        )
        # Always include OPTIONS tags, even if empty
        if options_str.strip():
            options_section = surround_with_tags(options_str, "OPTIONS")
        else:
            options_section = "<OPTIONS>\n</OPTIONS>"
        return f"{super_user_prompt}\n{options_section}"

    def compose_system_prompt(self) -> str:
        """Compose the system prompt."""
        return OPTION_SYSTEM_PROMPT

    def run(self, body: str | None) -> OptionAgentResponse:
        """Execute the agent's main logic."""
        base_result = super().run(body)
        init_option_response = OptionAgentResponse(
            thought=base_result.thought,
            response=base_result.response,
            option_index=None,
        )
        return self.process_output_option(init_option_response)

    def process_output_option(self, result: OptionAgentResponse) -> OptionAgentResponse:
        """Process the output.

        Based on the actual text response, determine the index of the option chosen.
        The final return value's response will contain the text of the option chosen,
        and the option_index will be the index of the option chosen.

        If the response is not a valid index, it will be treated as the "None" option.
        """
        try:
            # Extract the index from the response
            option_index = int(result.response.strip())
            # Check if the index is valid (within bounds)
            if 0 <= option_index < len(self.options):
                # Set the response to the actual option text
                result.response = self.options[option_index]
                result.option_index = option_index
            else:
                result.response = ""
                result.option_index = -1
        except (ValueError, AttributeError):
            # If parsing fails, treat as "None" option
            result.response = ""
            result.option_index = -1
        return result
