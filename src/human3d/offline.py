"""Block Python socket operations before loading numerical/vision packages.

This audit guard is not an OS sandbox and does not cover native-library I/O.
"""

import os
import sys

BLOCKED = frozenset(
    {
        "socket.connect",
        "socket.connect_ex",
        "socket.getaddrinfo",
        "socket.gethostbyname",
        "socket.sendto",
        "socket.sendmsg",
        "socket.bind",
    }
)


def install_guard():
    def audit(event, args):
        if event in BLOCKED:
            raise PermissionError(
                f"Local reconstruction forbids network activity: {event}"
            )

    sys.addaudithook(audit)
    for name, value in {
        "DO_NOT_TRACK": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "GRADIO_ANALYTICS_ENABLED": "False",
        "WANDB_DISABLED": "true",
    }.items():
        os.environ[name] = value
