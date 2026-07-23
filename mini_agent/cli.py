import os

import typer
from dotenv import load_dotenv

from .agent import DefaultAgent
from .environment import LocalEnvironment
from .model import OpenAIModel

load_dotenv()

app = typer.Typer()


@app.command()
def main(
    model: str = typer.Option(os.getenv("MODEL_NAME", "deepseek-v4-flash"), "-m", "--model", help="Model name"),
    api_base: str = typer.Option(os.getenv("API_BASE", "https://opencode.ai/zen/go/v1"), "--api-base", help="API base URL"),
    api_key: str = typer.Option(os.getenv("API_KEY", ""), "--api-key", help="API key", show_default=False, show_envvar=False),
    max_context: int = typer.Option(96000, "--max-ctx", help="Max context tokens before compression"),
    keep_turns: int = typer.Option(2, "--keep-turns", help="Full-fidelity turns to keep in context"),
):
    if api_key:
        os.environ.setdefault("OPENAI_API_KEY", api_key)

    agent = DefaultAgent(
        OpenAIModel(model_name=model, api_key=api_key, api_base=api_base),
        LocalEnvironment(),
        input_cost_per_token=float(os.getenv("INPUT_PRICE", "0")),
        output_cost_per_token=float(os.getenv("OUTPUT_PRICE", "0")),
        cached_input_cost_per_token=float(os.getenv("CACHED_PRICE", "0")),
        max_context_tokens=max_context,
        keep_turns=keep_turns,
    )
    agent.start()


if __name__ == "__main__":
    app()
