"""Click CLI entry point."""

import click


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """Kangaroo Shift — seamless context transfer between LLMs."""


@cli.command()
def health() -> None:
    """Check API server health."""
    click.echo("Kangaroo Shift CLI v0.1.0")


if __name__ == "__main__":
    cli()
