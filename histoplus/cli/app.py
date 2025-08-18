"""Histoplus CLI entrypoint.

This module provides a CLI for the Histoplus.

The CLI provides a simplified way to extract cell masks from slides.
"""

from pathlib import Path
from typing import List, Optional

import typer

from histoplus.cli.extract import extract_command


app = typer.Typer(help="HistoPLUS CLI")


@app.command(name="extract")
def extract_wrapper(
    ctx: typer.Context,
    # Required parameters
    slides: List[str] = typer.Option(
        None,
        "--slides",
        help="Path(s) to slides. Can be absolute (like /data/slides/*.tif) "
        "or relative (like ../../slides/*/*.svs), "
        "and can contain shell-style wildcards. Compatible with s3 path.",
    ),
    export_dir: Path = typer.Option(
        None,
        "--export_dir",
        "-e",
        help="Directory where results will be saved. Compatible with s3 path.",
    ),
    # Tile parameters
    tile_size: int = typer.Option(
        224,
        "--tile_size",
        help="Size of tiles to extract",
    ),
    n_tiles: Optional[int] = typer.Option(
        None,
        "--n_tiles",
        help="Number of tiles to extract. If None, all the tiles are extracted.",
    ),
    # Processing parameters
    n_workers: int = typer.Option(
        4,
        "--n_workers",
        help="Number of parallel workers",
    ),
    batch_size: int = typer.Option(
        8,
        "--batch_size",
        help="Batch size for processing",
    ),
    # Output parameters
    verbose: int = typer.Option(
        0,
        "--verbose",
        help="Verbosity level",
    ),
) -> None:
    """Extract cell masks from slides using HIPE segmentation."""
    return extract_command(
        ctx=ctx,
        slides=slides,
        export_dir=export_dir,
        tile_size=tile_size,
        n_tiles=n_tiles,
        n_workers=n_workers,
        batch_size=batch_size,
        verbose=verbose,
    )


def main():
    """Entry point for the application."""
    app()


if __name__ == "__main__":
    main()
