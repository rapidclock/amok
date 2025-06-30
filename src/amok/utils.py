"""Utility functions for the Amok project."""


def surround_with_tags(text: str, tag: str) -> str:
    """Surround the given text with specified tags.

    Args:
        text: The text to be surrounded.
        tag: The tag to use for surrounding the text.

    Returns:
        The text surrounded by the specified tags.

    """
    return f"<{tag}>{text}</{tag}>"
