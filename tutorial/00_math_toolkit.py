# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "numpy", "matplotlib", "scipy"]
# ///
"""Companion notebook for Part I, the mathematical toolkit: slide decks 01-04.

Backs slides/01_vectors_and_matrices.html, 02_eigenvalues_and_least_squares.html,
03_derivatives_and_jacobians.html and 04_probability_and_sampling.html. Every number on
those slides that a reader cannot compute by hand is computed by a function in this notebook,
in the section after that deck's "Deck NN" header cell; slides link to those functions with
data-code="00_math_toolkit.py:function_name".

Run locally:        uvx marimo edit --sandbox 00_math_toolkit.py
Run as a script:    uv run --script 00_math_toolkit.py
Slides:             open slides/index.html (Part I)
"""

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.linalg

    return mo, np, plt, scipy


@app.cell
def _(mo):
    mo.md(r"""
    # Mathematical toolkit: numerical companion to decks 01–04

    Part I builds the tools the rest of the series uses: vectors and matrices as linear maps,
    rank and null spaces, projections, least squares and the SVD, derivatives and Jacobians,
    and Gaussians and sampling. The cells after each deck's header compute the numbers on that
    deck's slides. Numbers quoted from the tex tutorial are treated as unverified until a cell
    here recomputes them.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Deck 01 · Vectors, matrices, and linear maps

    Slides: `slides/01_vectors_and_matrices.html`
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Deck 02 · Eigenvalues, singular values, and least squares

    Slides: `slides/02_eigenvalues_and_least_squares.html`
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Deck 03 · Derivatives, gradients, and Jacobians

    Slides: `slides/03_derivatives_and_jacobians.html`
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Deck 04 · Probability, Gaussians, and sampling

    Slides: `slides/04_probability_and_sampling.html`
    """)
    return


if __name__ == "__main__":
    app.run()
