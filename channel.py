"""
An unreliable network, on purpose.

channel.py is a UDP proxy that sits between the sender and the receiver and
degrades the traffic passing through it: dropping datagrams, flipping bits,
adding delay and jitter, and reordering packets.

    sender  --->  :8000  channel  :ephemeral  --->  :9000  receiver
    sender  <---  :8000  channel  :ephemeral  <---  :9000  receiver

Run it in its own terminal:

    python channel.py --loss 0.2 --delay 20 --jitter 5 --verbose

Why a separate process rather than `if random() < p: skip` inside the sender:
a sender that simulates its own packet loss is testing a fake network it
controls. Here the datagrams genuinely leave the process and genuinely fail to
arrive, so the protocol code is unaware it is being abused — and the same
sender can be pointed at a real remote host later without changing a line.
"""

import argparse
import heapq
import itertools
import random
import select
import socket
import sys
import threading
import time
from dataclasses import dataclass, field

# Only used to pretty-print sequence numbers in --verbose mode. The channel
# never acts on packet contents; it is a dumb pipe with a grudge.
try:
    from rdt.packet import Packet, PacketError
except ImportError:  # channel.py must still run if rdt/ is missing
    Packet = None
    PacketError = Exception


FORWARD = "-->"  # sender to receiver (data)
REVERSE = "<--"  # receiver to sender (acks)


# ---------------------------------------------------------------------------
# Impairment settings
# ---------------------------------------------------------------------------


@dataclass
class Impairments:
    """Probabilities and timings for one direction of travel."""

    loss: float = 0.0
    corrupt: float = 0.0
    reorder: float = 0.0
    delay_ms: float = 0.0
    jitter_ms: float = 0.0
    # Extra hold time applied to a reordered packet, letting the packet behind
    # it overtake. Reordering is just differential delay.
    reorder_ms: float = 60.0


@dataclass
class Stats:
    """What actually happened, per direction."""

    received: int = 0
    forwarded: int = 0
    dropped: int = 0
    corrupted: int = 0
    reordered: int = 0
    delayed: int = 0

    def line(self, label: str) -> str:
        pct = (100.0 * self.dropped / self.received) if self.received else 0.0
        return (
            f"  {label}  in={self.received:<6} out={self.forwarded:<6} "
            f"dropped={self.dropped:<5} ({pct:5.1f}%)  "
            f"corrupted={self.corrupted:<5} reordered={self.reordered:<5}"
        )


# ---------------------------------------------------------------------------
# Delayed delivery
# ---------------------------------------------------------------------------


@dataclass(order=True)
class _Scheduled:
    due: float
    tiebreak: int
    sock: socket.socket = field(compare=False)
    data: bytes = field(compare=False)
    addr: tuple = field(compare=False)


class Scheduler:
    """A min-heap of pending sends, drained by a background thread.

    Delay has to be asynchronous: if the proxy slept in the receive loop it
    would stall every other packet too, turning independent per-packet delay
    into a queue. That would quietly change the shape of the network being
    simulated.
    """

    def __init__(self):
        self._heap: list[_Scheduled] = []
        self._counter = itertools.count()
        self._cv = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        with self._cv:
            self._running = False
            self._cv.notify_all()

    def send_at(self, due: float, sock: socket.socket, data: bytes, addr: tuple):
        with self._cv:
            heapq.heappush(
                self._heap, _Scheduled(due, next(self._counter), sock, data, addr)
            )
            self._cv.notify()

    def _run(self):
        while True:
            with self._cv:
                if not self._running:
                    return
                if not self._heap:
                    self._cv.wait(timeout=0.2)
                    continue

                wait = self._heap[0].due - time.monotonic()
                if wait > 0:
                    self._cv.wait(timeout=wait)
                    continue

                item = heapq.heappop(self._heap)

            try:
                item.sock.sendto(item.data, item.addr)
            except OSError:
                pass  # socket closed during shutdown


# ---------------------------------------------------------------------------
# The channel
# ---------------------------------------------------------------------------


class Channel:
    def __init__(
        self,
        listen_port: int,
        forward_host: str,
        forward_port: int,
        fwd: Impairments,
        rev: Impairments,
        verbose: bool = False,
        seed: int | None = None,
    ):
        self.receiver_addr = (forward_host, forward_port)
        self.sender_addr: tuple | None = None  # learned from the first datagram
        self.fwd = fwd
        self.rev = rev
        self.verbose = verbose
        self.rng = random.Random(seed)

        self.stats = {FORWARD: Stats(), REVERSE: Stats()}

        # Faces the sender.
        self.sock_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_sender.bind(("0.0.0.0", listen_port))

        # Faces the receiver. Bound to an ephemeral port; the receiver replies
        # to whatever source address it sees, so ACKs come back here.
        self.sock_receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_receiver.bind(("0.0.0.0", 0))

        self.scheduler = Scheduler()

    # -- helpers ------------------------------------------------------------

    def _describe(self, data: bytes) -> str:
        if Packet is None:
            return f"{len(data)}B"
        try:
            return repr(Packet.from_bytes(data))
        except PacketError:
            return f"<unparseable {len(data)}B>"

    def _corrupt(self, data: bytes) -> bytes:
        """Flip one random bit. This is what the checksum exists to catch."""
        idx = self.rng.randrange(len(data))
        bit = self.rng.randrange(8)
        mutable = bytearray(data)
        mutable[idx] ^= 1 << bit
        return bytes(mutable)

    def _hold_time(self, imp: Impairments, reordering: bool) -> float:
        held = imp.delay_ms
        if imp.jitter_ms:
            held += self.rng.uniform(-imp.jitter_ms, imp.jitter_ms)
        if reordering:
            held += imp.reorder_ms
        return max(0.0, held) / 1000.0

    # -- the interesting part -----------------------------------------------

    def _handle(self, data: bytes, direction: str, out_sock, out_addr):
        imp = self.fwd if direction == FORWARD else self.rev
        st = self.stats[direction]
        st.received += 1

        note = ""

        if self.rng.random() < imp.loss:
            st.dropped += 1
            if self.verbose:
                print(f"{direction} DROP      {self._describe(data)}")
            return

        if self.rng.random() < imp.corrupt:
            data = self._corrupt(data)
            st.corrupted += 1
            note += " [corrupted]"

        reordering = self.rng.random() < imp.reorder
        if reordering:
            st.reordered += 1
            note += " [reordered]"

        hold = self._hold_time(imp, reordering)
        st.forwarded += 1

        if hold > 0:
            st.delayed += 1
            note += f" [+{hold * 1000:.0f}ms]"
            self.scheduler.send_at(time.monotonic() + hold, out_sock, data, out_addr)
        else:
            out_sock.sendto(data, out_addr)

        if self.verbose:
            print(f"{direction} forward   {self._describe(data)}{note}")

    def run(self):
        self.scheduler.start()
        print(
            f"channel listening on :{self.sock_sender.getsockname()[1]}, "
            f"forwarding to {self.receiver_addr[0]}:{self.receiver_addr[1]}"
        )
        print(
            f"  data  loss={self.fwd.loss}  corrupt={self.fwd.corrupt}  "
            f"reorder={self.fwd.reorder}  delay={self.fwd.delay_ms}ms"
        )
        print(
            f"  acks  loss={self.rev.loss}  corrupt={self.rev.corrupt}  "
            f"reorder={self.rev.reorder}  delay={self.rev.delay_ms}ms"
        )
        print("Ctrl+C to stop.\n")

        socks = [self.sock_sender, self.sock_receiver]
        try:
            while True:
                # Short timeout so Ctrl+C is responsive on Windows, where a
                # blocking select is not interrupted by the signal.
                ready, _, _ = select.select(socks, [], [], 0.5)

                for sock in ready:
                    data, addr = sock.recvfrom(65535)

                    if sock is self.sock_sender:
                        self.sender_addr = addr
                        self._handle(
                            data, FORWARD, self.sock_receiver, self.receiver_addr
                        )
                    else:
                        if self.sender_addr is None:
                            continue  # nothing has spoken to us yet
                        self._handle(
                            data, REVERSE, self.sock_sender, self.sender_addr
                        )
        except KeyboardInterrupt:
            pass
        finally:
            self.scheduler.stop()
            self.summary()

    def summary(self):
        print("\n--- channel summary ---")
        print(self.stats[FORWARD].line("data -->"))
        print(self.stats[REVERSE].line("acks <--"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _pick(specific, general):
    """A direction-specific flag wins; otherwise fall back to the shared one."""
    return general if specific is None else specific


def main(argv=None):
    p = argparse.ArgumentParser(
        description="UDP proxy that drops, corrupts, delays and reorders packets."
    )
    p.add_argument("--listen-port", type=int, default=8000,
                   help="port the sender talks to (default: 8000)")
    p.add_argument("--forward-host", default="127.0.0.1")
    p.add_argument("--forward-port", type=int, default=9000,
                   help="port the receiver is bound to (default: 9000)")

    # Shared settings, applied to both directions.
    p.add_argument("--loss", type=float, default=0.0,
                   help="drop probability, 0.0-1.0")
    p.add_argument("--corrupt", type=float, default=0.0,
                   help="bit-flip probability, 0.0-1.0")
    p.add_argument("--reorder", type=float, default=0.0,
                   help="probability a packet is held so the next overtakes it")
    p.add_argument("--delay", type=float, default=0.0,
                   help="one-way delay in ms")
    p.add_argument("--jitter", type=float, default=0.0,
                   help="+/- random variation on the delay, in ms")

    # Per-direction overrides. Losing ACKs behaves very differently from
    # losing data: a lost ACK costs a retransmission of a packet that already
    # arrived, which is how duplicates appear at the receiver.
    p.add_argument("--ack-loss", type=float, default=None)
    p.add_argument("--ack-corrupt", type=float, default=None)
    p.add_argument("--ack-reorder", type=float, default=None)
    p.add_argument("--ack-delay", type=float, default=None)
    p.add_argument("--ack-jitter", type=float, default=None)

    p.add_argument("--seed", type=int, default=None,
                   help="seed the RNG for reproducible runs")
    p.add_argument("--verbose", action="store_true",
                   help="log every packet and what was done to it")

    args = p.parse_args(argv)

    for name in ("loss", "corrupt", "reorder"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            p.error(f"--{name} must be between 0.0 and 1.0")

    fwd = Impairments(
        loss=args.loss,
        corrupt=args.corrupt,
        reorder=args.reorder,
        delay_ms=args.delay,
        jitter_ms=args.jitter,
    )
    rev = Impairments(
        loss=_pick(args.ack_loss, args.loss),
        corrupt=_pick(args.ack_corrupt, args.corrupt),
        reorder=_pick(args.ack_reorder, args.reorder),
        delay_ms=_pick(args.ack_delay, args.delay),
        jitter_ms=_pick(args.ack_jitter, args.jitter),
    )

    Channel(
        listen_port=args.listen_port,
        forward_host=args.forward_host,
        forward_port=args.forward_port,
        fwd=fwd,
        rev=rev,
        verbose=args.verbose,
        seed=args.seed,
    ).run()


if __name__ == "__main__":
    sys.exit(main())
