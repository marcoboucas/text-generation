"""Text operations."""
from typing import List

def process_text(text: str) -> str:
    """Process the text."""
    return text.lower()

def split_to_tokens(text: str) -> List[str]:
    """Split a text into tokens."""
    return text.split(' ')