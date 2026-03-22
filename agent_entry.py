"""
Legacy entrypoint kept for compatibility with existing skill.json.

It now forwards to the modular NeoSpark architecture in `neospark/`.
Running this file directly starts autonomous mode.
"""

from neospark.autonomous import run_autonomous
from neospark.responder import handle_interactive_request


def handle_request(data):
    """Compatibility adapter for request/response mode."""
    return handle_interactive_request(data)


if __name__ == "__main__":
    run_autonomous()
