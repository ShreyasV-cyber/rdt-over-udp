"""
Send a file over the rdt protocol.

    python send_file.py samples/test_file.bin --verbose

Point this at the channel's port (8000), not the receiver's (9000), so the
traffic goes through the impairments.
"""

import argparse
import hashlib
import os
import sys

from rdt.constants import DEFAULT_CHANNEL_PORT, DEFAULT_WINDOW, INITIAL_TIMEOUT
from rdt.sender import (
    GoBackNSender,
    SelectiveRepeatSender,
    StopAndWaitSender,
    TransferFailed,
)

# Adding Go-Back-N and Selective Repeat later is one line each here.
PROTOCOLS = {
    "sw": StopAndWaitSender,
    "gbn": GoBackNSender,
    "sr": SelectiveRepeatSender,
}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description="Send a file over UDP, reliably.")
    p.add_argument("file", help="path to the file to send")
    p.add_argument("--protocol", choices=sorted(PROTOCOLS), default="sw",
                   help="sw = stop-and-wait (default), gbn = go-back-n, sr = selective repeat")
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                   help=f"window size, gbn/sr only (default: {DEFAULT_WINDOW})")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=DEFAULT_CHANNEL_PORT,
                   help=f"port to send to — the channel (default: {DEFAULT_CHANNEL_PORT})")
    p.add_argument("--timeout", type=float, default=INITIAL_TIMEOUT,
                   help=f"retransmission timeout in seconds (default: {INITIAL_TIMEOUT})")
    p.add_argument("--json", action="store_true",
                   help="print metrics as JSON instead of a summary")
    p.add_argument("--verbose", action="store_true")

    args = p.parse_args(argv)

    if not os.path.isfile(args.file):
        p.error(f"no such file: {args.file}")

    kwargs = dict(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        verbose=args.verbose,
    )
    if args.protocol != "sw":
        kwargs["window"] = args.window

    sender = PROTOCOLS[args.protocol](**kwargs)

    print(
        f"sending {args.file} ({os.path.getsize(args.file)} B) "
        f"to {args.host}:{args.port} via {args.protocol}"
        + (f" (window {args.window})" if args.protocol != "sw" else "")
    )

    try:
        metrics = sender.send_file(args.file)
    except TransferFailed as exc:
        print(f"transfer failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(metrics.to_json())
    else:
        print()
        print(metrics.summary(f"sender ({args.protocol})"))
        print(f"  sha256             {sha256(args.file)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())