# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "numpy", "matplotlib", "scipy", "mujoco==3.14.0"]
# ///
"""Companion notebook for Part VI, slide deck 21 (hybrid force-velocity control), and the lab.

Backs slides/21_hybrid_force_velocity_control.html and the lab in labs/hybrid_servoing/,
which reimplements Hou & Mason, ICRA 2019 (arXiv 1903.02715; Algorithms 1 and 2) and ICRA 2021
(arXiv 2011.04872; the closed-form optimal hybrid servoing, OCHS) and runs them in MuJoCo on
the block-tilting task. Every number on deck 21 that a reader cannot compute by hand is computed
by a function in the section after the "Deck 21" header cell; slides link to those functions
with data-code="07_hybrid_servoing.py:function_name".

mujoco (pinned to 3.14.0, the version of the WebAssembly build in slides/lib/mujoco/) is
imported inside try: so the notebook still opens in the browser (molab / Pyodide), where it is
unavailable; the MuJoCo cells then print a message instead of running.

Run locally:        uvx marimo edit --sandbox 07_hybrid_servoing.py
Run as a script:    uv run --script 07_hybrid_servoing.py
Slides:             open slides/index.html (Part VI)
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

    try:
        import mujoco
    except ImportError:  # the browser build (Pyodide) has no mujoco
        mujoco = None
    return mo, mujoco, np, plt, scipy


@app.cell
def _(mo):
    mo.md(r"""
    # Hybrid force-velocity control: numerical companion to deck 21 and the lab

    Deck 21 assembles hybrid force-velocity control from the earlier decks: natural and
    artificial constraints (Mason 1981), selection matrices (Raibert and Craig 1981), and the
    problem formulation, Algorithms 1 and 2 of Hou and Mason 2019 and the closed-form optimal
    hybrid servoing of Hou and Mason 2021. The cells after the deck header compute the numbers
    on its slides; the cells after the lab header run the lab's implementation in MuJoCo on the
    block-tilting task.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Deck 21 · Hybrid force-velocity control

    Slides: `slides/21_hybrid_force_velocity_control.html`
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Lab · Hou and Mason hybrid servoing on the block-tilting task

    Code: `labs/hybrid_servoing/`
    """)
    return


if __name__ == "__main__":
    app.run()
