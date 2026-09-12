"""
Receive a file over the rdt protocol.

    python recv_file.py received.bin --verbose

Start this before the sender: it binds the port the channel forwards to.
"""

import argparse
import hashlib
import os
import sys

from rdt.constants import DEFAULT_RECEIVER_PORT
from rdt.receiver import GoBackNReceiver, StopAndWaitReceiver

# Adding Go-Back-N and Selective Repeat later is one line each here.
PROTOCOLS = {
    "sw": StopAndWaitReceiver,
    "gbn": GoBackNReceiver,
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description="Receive a file over UDP, reliably.")
    p.add_argument("output", help="path to write the received file to")
    p.add_argument("--protocol", choices=sorted(PROTOCOLS), default="sw",
                   help="sw = stop-and-wait (default), gbn = go-back-n")
    p.add_argument("--port", type=int, default=DEFAULT_RECEIVER_PORT,
                   help=f"port to bind (default: {DEFAULT_RECEIVER_PORT})")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--json", action="store_true",
                   help="print metrics as JSON instead of a summary")
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args(argv)

    receiver = PROTOCOLS[args.protocol](
        output_path=args.output,
        port=args.port,
        host=args.host,
        verbose=args.verbose,
    )

    print(f"listening on {args.host}:{args.port}, writing to {args.output}")
    metrics = receiver.run()

    if args.json:
        print(metrics.to_json())
    else:
        print()
        print(metrics.summary(f"receiver ({args.protocol})"))
        print(f"  sha256             {sha256(args.output)}")
        print(f"  size               {os.path.getsize(args.output)} B")

    return 0


if __name__ == "__main__":
    sys.exit(main())