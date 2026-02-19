import numpy as np
import threading
import zmq
from typing import Optional, Callable
from orca_core import OrcaHand


class ManusIngress:
    """Manus Quantum Metaglove ingress via ZMQ raw skeleton stream."""

    NODES_PER_HAND = 25
    FLOATS_PER_NODE = 7  # x, y, z, qx, qy, qz, qw
    VALUES_PER_HAND = NODES_PER_HAND * FLOATS_PER_NODE  # 175 floats + 1 ID string = 176 fields

    def __init__(self, model_path: str, glove_id: str, callback: Optional[Callable] = None,
                 zmq_addr: str = "tcp://localhost:8000"):
        self.callback = callback
        self.glove_id = glove_id.lower()
        self.zmq_addr = zmq_addr

        hand = OrcaHand(model_path)
        if hand.type not in ["left", "right"]:
            raise ValueError("hand.type must be 'left' or 'right'. Update config.yaml with type field.")
        self.hand_type = hand.type

        self.latest_skeleton = None
        self.frame_lock = threading.Lock()
        self.running = False
        self._thread = None

        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.PULL)
        self._socket.setsockopt(zmq.CONFLATE, 1)

    def _parse_hand_block(self, values):
        """Parse 176 CSV fields (1 ID + 175 floats) into glove_id and (25, 7) array."""
        glove_id = values[0].lower()
        floats = np.array([float(v) for v in values[1:]], dtype=np.float32)
        skeleton = floats.reshape(self.NODES_PER_HAND, self.FLOATS_PER_NODE)
        return glove_id, skeleton

    def _receive_loop(self):
        self._socket.connect(self.zmq_addr)
        while self.running:
            try:
                msg = self._socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                threading.Event().wait(0.001)
                continue

            fields = msg.decode("utf-8").split(",")
            n = len(fields)

            # 176 = single hand, 352 = two hands
            hand_fields = 1 + self.NODES_PER_HAND * self.FLOATS_PER_NODE  # 176
            if n == hand_fields:
                blocks = [fields]
            elif n == hand_fields * 2:
                blocks = [fields[:hand_fields], fields[hand_fields:]]
            else:
                continue

            for block in blocks:
                glove_id, skeleton = self._parse_hand_block(block)
                if glove_id != self.glove_id:
                    continue

                with self.frame_lock:
                    self.latest_skeleton = skeleton

                if self.callback:
                    self.callback({"skeleton": skeleton})

    def start(self):
        if not self.running:
            self.running = True
            self._thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._thread.start()

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def display_frame(self):
        pass

    def cleanup(self):
        self.stop()
        self._socket.close()
        self._ctx.term()
