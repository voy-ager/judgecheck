"""Typer command line interface."""

import typer

app = typer.Typer(help="Check whether your LLM-judge eval scores can be trusted.")


@app.command()
def audit() -> None:
    """Audit judge scores against human labels."""
    typer.echo("not implemented yet")


@app.command()
def probe() -> None:
    """Run bias probes on a sample."""
    typer.echo("not implemented yet")


@app.command()
def report() -> None:
    """Render a Markdown or HTML report."""
    typer.echo("not implemented yet")
