"""mjview.py: an mjviser viewer (Kevin Zakka's viser-based MuJoCo viewer) inside a marimo notebook.

    import mujoco, mjview
    model = mujoco.MjModel.from_xml_path("models/box_on_plane.xml")
    data = mujoco.MjData(model)
    view = mjview.show(model, data)      # starts a viser server on a free port, returns a View
    view                                 # as a cell's last expression: the viewer in an iframe
    view.run(seconds=3)                  # steps data with mj_step in a background thread
    view.stop()                          # stops the thread and the server

show(model, data, port=None, height=480, host="127.0.0.1", contacts=True, key="default")
    Starts viser.ViserServer(host, port) (port=None picks a free one) and builds
    mjviser.scene.ViserMujocoScene(server, copy of model, num_envs=1), which does not block.
    The model is copied because the scene rescales model.vis for drawing; physics is unchanged.
    contacts=True shows MuJoCo's contact points and contact-force arrows. A second show() with
    the same key stops the first viewer, so re-running a notebook cell does not leak servers.

View
    .url, .port      http://localhost:PORT, where the viewer is served
    .iframe          HTML: <iframe src="http://localhost:PORT" ...>, for mo.Html(view.iframe);
                     marimo also shows a View directly (_repr_html_)
    .update(data=None)   pushes the poses and contacts of data (default: the show() data)
    .run(step_fn=None, seconds=5.0, speed=1.0, fps=30) -> threading.Thread
                     calls step_fn(model, data) (default mujoco.mj_step; it must advance
                     data.time) until `seconds` of simulation time have passed, paced to wall
                     time x speed, and pushes a frame fps times per second; a second run()
                     stops the first one
    .running         True while a run() thread is stepping
    .join(timeout=None)  waits for the run() thread
    .lock            held by the run() thread around every step and push; hold it
                     (with view.lock: ...) to edit data while a run is going
    .stop()          ends the run() thread and stops the server; the View is dead after

mjview.stop(key="default") stops the View a show() with that key started; mjview.stop_all()
stops every View, and runs at interpreter exit.

When mujoco or mjviser cannot be imported (the browser build of a notebook, or an environment
without them), show() returns a stand-in whose .iframe is a message saying what is missing and
whose methods do nothing, so the calling cell still runs; .error holds the reason. A server or
scene that fails to start (a port in use) gives the same stand-in. mjviser is not a dependency of the
notebooks; add it with `uv run --with mjviser --with mujoco==3.14.0 marimo edit NOTEBOOK.py` or
`pip install mjviser`. mjviser 0.0.14 with viser 1.1.1 and mujoco 3.14.0 was the tested set.

The iframe points at localhost, so it works when the browser runs on the machine that runs the
notebook kernel. Over SSH, forward the port as well (ssh -L PORT:localhost:PORT host).
"""

from __future__ import annotations

import atexit
import copy
import html
import socket
import threading
import time

__all__ = ["show", "available", "stop", "stop_all", "View", "Unavailable"]

_LIVE: dict[str, "View"] = {}


def available() -> str | None:
    """None when mujoco and mjviser import; otherwise the reason they do not."""
    try:
        import mujoco  # noqa: F401
    except ImportError as e:
        return f"mujoco is not installed ({e})"
    try:
        import mjviser.scene  # noqa: F401
        import viser  # noqa: F401
    except ImportError as e:
        return f"mjviser is not installed ({e}); pip install mjviser"
    return None


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _message_html(text: str, height: int) -> str:
    return (
        f'<div style="height:{int(height)}px;display:flex;align-items:center;justify-content:center;'
        'border:1px solid #999;font-family:Times New Roman,serif;color:#555;padding:0 1em">'
        f"{html.escape(text)}</div>"
    )


class Unavailable:
    """Stand-in returned by show() when mujoco or mjviser is missing."""

    def __init__(self, reason: str, height: int = 480):
        self.error = reason
        self.url = None
        self.port = None
        self.iframe = _message_html("mjviser viewer unavailable: " + reason, height)
        self.lock = threading.Lock()
        self.running = False

    def update(self, data=None):
        return self

    def run(self, step_fn=None, seconds=5.0, speed=1.0, fps=30):
        return None

    def join(self, timeout=None):
        return self

    def stop(self):
        return None

    def _repr_html_(self):
        return self.iframe


class View:
    """A running viser server that shows one MuJoCo model; made by show()."""

    def __init__(self, model, data, port, height, host, contacts):
        import mujoco
        import viser
        from mjviser.scene import ViserMujocoScene

        self._mj = mujoco
        self.model = model
        self.data = data
        self.height = int(height)
        self.lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._halt = threading.Event()
        port = int(port) if port else _free_port(host)
        try:
            self.server = viser.ViserServer(host=host, port=port, verbose=False)
        except TypeError:  # viser releases without the verbose flag
            self.server = viser.ViserServer(host=host, port=port)
        self.port = self.server.get_port()
        self.url = f"http://localhost:{self.port}"
        self.dead = False
        try:
            # The scene rescales model.vis (contact and frame widths) in place; give it a copy.
            self.scene = ViserMujocoScene(self.server, copy.copy(model), num_envs=1)
            self.scene.camera_tracking_enabled = False
            if contacts:
                self.scene.show_contact_points = True
                self.scene.show_contact_forces = True
            self.update()
        except BaseException:
            self.dead = True
            self.server.stop()
            raise

    @property
    def iframe(self) -> str:
        return (
            f'<iframe src="{self.url}" width="100%" height="{self.height}" '
            'style="border:1px solid #999" title="mjviser"></iframe>'
        )

    def _repr_html_(self):
        return self.iframe

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def update(self, data=None):
        """Push the poses (and contacts) of data, default the data given to show()."""
        if self.dead:
            return self
        with self.lock:
            self.scene.update_from_mjdata(self.data if data is None else data)
        return self

    def run(self, step_fn=None, seconds=5.0, speed=1.0, fps=30):
        """Step in a background thread for `seconds` of simulation time; returns the thread."""
        if self.dead:
            raise RuntimeError("mjview: this View was stopped")
        self._end_run()
        step = step_fn or self._mj.mj_step
        m, d, h = self.model, self.data, self.model.opt.timestep
        t_end = d.time + float(seconds)
        frame = 1.0 / max(1.0, float(fps))
        halt = self._halt = threading.Event()

        def loop():
            wall0, sim0 = time.perf_counter(), d.time
            while not halt.is_set() and d.time < t_end - 0.5 * h:
                target = min(t_end, sim0 + (time.perf_counter() - wall0) * speed)
                with self.lock:
                    n = 0
                    while d.time < target - 0.5 * h and n < 100000:
                        step(m, d)
                        n += 1
                    self.scene.update_from_mjdata(d)
                halt.wait(frame)

        self._thread = threading.Thread(target=loop, name=f"mjview:{self.port}", daemon=True)
        self._thread.start()
        return self._thread

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)
        return self

    def _end_run(self):
        self._halt.set()
        if self._thread is not None:
            self._thread.join(5.0)
        self._thread = None

    def stop(self):
        """End the run() thread and stop the viser server."""
        if self.dead:
            return
        self._end_run()
        self.dead = True
        self.server.stop()
        for k, v in list(_LIVE.items()):
            if v is self:
                del _LIVE[k]


def show(model, data, port=None, height=480, host="127.0.0.1", contacts=True, key="default"):
    """A View of model/data served by viser, or an Unavailable stand-in (see the module doc)."""
    why = available()
    if why is not None:
        return Unavailable(why, height)
    old = _LIVE.pop(key, None)
    if old is not None:
        old.stop()
    try:
        view = View(model, data, port, height, host, contacts)
    except Exception as e:  # noqa: BLE001 - a port in use, a model mjviser cannot draw
        return Unavailable(f"{type(e).__name__}: {e}", height)
    _LIVE[key] = view
    return view


def stop(key="default"):
    """Stop the View that show(..., key=key) started, if it is still running."""
    view = _LIVE.pop(key, None)
    if view is not None:
        view.stop()


def stop_all():
    """Stop every View this module started."""
    for v in list(_LIVE.values()):
        v.stop()


atexit.register(stop_all)
