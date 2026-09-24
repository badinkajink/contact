# /// script
# requires-python = ">=3.10"
# dependencies = ["marimo", "numpy", "matplotlib", "scipy", "mujoco==3.14.0"]
# ///
"""Companion notebook for Part VI, slide deck 22 (trajectory optimization through contact).

Backs slides/22_contact_trajopt.html (tex chapter lines 3845-3977): contact-implicit
trajectory optimization in the style of Posa et al. 2014, and predictive-sampling MPC pushing a
block to a goal in MuJoCo. Every number on deck 22 that a reader cannot compute by hand is
computed by a function in the section after the "Deck 22" header cell; slides link to those
functions with data-code="08_contact_trajopt.py:function_name".

mujoco (pinned to 3.14.0, the version of the WebAssembly build in slides/lib/mujoco/) is
imported inside try: so the notebook still opens in the browser (molab / Pyodide), where it is
unavailable; the MuJoCo cells then print a message instead of running.

Run locally:        uvx marimo edit --sandbox 08_contact_trajopt.py
Run as a script:    uv run --script 08_contact_trajopt.py
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
    import scipy.optimize

    try:
        import mujoco
    except ImportError:  # the browser build (Pyodide) has no mujoco
        mujoco = None
    return mo, mujoco, np, plt, scipy


@app.cell
def _(mo):
    mo.md(r"""
    # Trajectory optimization through contact: numerical companion to deck 22

    Deck 22 treats contact forces as decision variables of the trajectory optimization, with
    complementarity constraints between gap and normal force, and compares the result with
    sampling-based MPC that pushes a block to a goal through MuJoCo rollouts. The cells after
    the deck header compute the numbers on its slides.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Deck 22 · Trajectory optimization through contact

    Slides: `slides/22_contact_trajopt.html`
    """)
    return


if __name__ == "__main__":
    app.run()
