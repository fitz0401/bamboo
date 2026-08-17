import threading

import msgpack
import numpy as np
import zmq

from bamboo import BambooFrankaClient


def test_streaming_client_protocol() -> None:
    context = zmq.Context()
    server = context.socket(zmq.REP)
    port = server.bind_to_random_port("tcp://127.0.0.1")
    requests: list[dict] = []

    def serve() -> None:
        while len(requests) < 5:
            request = msgpack.unpackb(server.recv(), raw=False)
            requests.append(request)
            if request["command"] == "get_robot_state":
                response = {
                    "success": True,
                    "data": {
                        "q": [0.0] * 7,
                        "dq": [0.0] * 7,
                        "tau_J": [0.0] * 7,
                        "O_T_EE": np.eye(4).T.reshape(-1).tolist(),
                        "time_sec": 1.0,
                    },
                }
            elif request["command"] == "get_capabilities":
                response = {
                    "success": True,
                    "protocol_version": 1,
                    "features": ["stream_joint_velocity", "watchdog"],
                }
            else:
                response = {"success": True, "error": ""}
            server.send(msgpack.packb(response, use_bin_type=True))

    thread = threading.Thread(target=serve)
    thread.start()
    client = BambooFrankaClient(
        server_ip="127.0.0.1",
        control_port=port,
        enable_gripper=False,
    )
    assert client.supports_streaming()
    assert client.start_streaming()["success"]
    assert client.stream_joint_velocity(np.arange(7) * 0.01)["success"]
    assert client.stop_streaming()["success"]
    client.close()

    thread.join(timeout=1.0)
    server.close()
    context.term()
    assert not thread.is_alive()
    assert [request["command"] for request in requests] == [
        "get_robot_state",
        "get_capabilities",
        "start_stream",
        "stream_joint_velocity",
        "stop_stream",
    ]
