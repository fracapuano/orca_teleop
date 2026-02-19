import re
import time
import webbrowser
from pathlib import Path
from functools import partial

import numpy as np
import viser
from viser.extras import ViserUrdf
import yourdfpy


FINGER_COLORS = np.array([
    [200, 200, 200],  # 0: wrist
    [255, 80, 80],    # 1-4: thumb
    [255, 80, 80],
    [255, 80, 80],
    [255, 80, 80],
    [80, 255, 80],    # 5-8: index
    [80, 255, 80],
    [80, 255, 80],
    [80, 255, 80],
    [80, 80, 255],    # 9-12: middle
    [80, 80, 255],
    [80, 80, 255],
    [80, 80, 255],
    [255, 255, 80],   # 13-16: ring
    [255, 255, 80],
    [255, 255, 80],
    [255, 255, 80],
    [255, 80, 255],   # 17-20: pinky
    [255, 80, 255],
    [255, 80, 255],
    [255, 80, 255],
], dtype=np.uint8)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def _package_uri_handler(fname, dir):
    stripped = re.sub(r"^package://[^/]*/", "", fname)
    resolved = Path(dir) / stripped
    if resolved.exists():
        return str(resolved)
    return yourdfpy.filename_handler_magic(fname, dir=dir)


class URDFViewer:
    def __init__(self, urdf_path, port=8080):
        self._server = viser.ViserServer(port=port)
        urdf_dir = Path(urdf_path).parent
        urdf = yourdfpy.URDF.load(
            urdf_path,
            filename_handler=partial(_package_uri_handler, dir=str(urdf_dir)),
            build_scene_graph=True,
            load_meshes=True,
            load_collision_meshes=False,
        )
        self._viser_urdf = ViserUrdf(self._server, urdf_or_path=urdf)
        self._joint_names = self._viser_urdf.get_actuated_joint_names()
        self._viser_urdf.update_cfg(np.zeros(len(self._joint_names)))
        self._server.scene.add_grid("/ground", width=2.0, height=2.0)

        self._mano_visible = self._server.gui.add_checkbox(
            "Show MANO Points", initial_value=True,
        )
        self._stop_button = self._server.gui.add_button("Stop")

        self._stopped = False
        self._has_connected = False
        self._start_time = time.time()
        self._mano_points_handle = None
        self._mano_lines_handle = None

        @self._stop_button.on_click
        def _(_):
            self._stopped = True

        @self._server.on_client_connect
        def _(client):
            self._has_connected = True

        @self._server.on_client_disconnect
        def _(client):
            if self._has_connected and len(self._server.get_clients()) == 0 and time.time() - self._start_time > 3.0:
                self._stopped = True

        webbrowser.open(f"http://localhost:{port}")

    @property
    def stopped(self):
        return self._stopped

    def update(self, joint_angles):
        cfg = np.array([joint_angles.get(name, 0.0) for name in self._joint_names])
        self._viser_urdf.update_cfg(cfg)

    def update_mano_points(self, points):
        if not self._mano_visible.value:
            if self._mano_points_handle is not None:
                self._mano_points_handle.remove()
                self._mano_points_handle = None
            if self._mano_lines_handle is not None:
                self._mano_lines_handle.remove()
                self._mano_lines_handle = None
            return

        points = points.astype(np.float32)
        colors = FINGER_COLORS[:len(points)]

        self._mano_points_handle = self._server.scene.add_point_cloud(
            "/mano_points",
            points=points,
            colors=colors,
            point_size=0.005,
        )

        line_points = np.array(
            [[points[i], points[j]] for i, j in HAND_CONNECTIONS if i < len(points) and j < len(points)],
            dtype=np.float32,
        )
        line_colors = np.full((len(line_points), 2, 3), 150, dtype=np.uint8)
        self._mano_lines_handle = self._server.scene.add_line_segments(
            "/mano_lines",
            points=line_points,
            colors=line_colors,
        )

    def close(self):
        self._server.stop()
