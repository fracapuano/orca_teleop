import re
import webbrowser
from pathlib import Path
from functools import partial

import numpy as np
import viser
from viser.extras import ViserUrdf
import yourdfpy


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
        webbrowser.open(f"http://localhost:{port}")

    def update(self, joint_angles):
        cfg = np.array([joint_angles.get(name, 0.0) for name in self._joint_names])
        self._viser_urdf.update_cfg(cfg)

    def close(self):
        self._server.close()
