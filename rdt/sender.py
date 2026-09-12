"""
Senders.

Stop-and-wait and Go-Back-N; Selective Repeat joins them here later. All
share BaseSender.

The sender carries the hard half of reliability. Four things in here are
easy to get subtly wrong:

  1. An ACK for the wrong sequence number is not the ACK you are waiting
     for. Late ACKs from earlier retransmissions turn up constantly once
     the link is lossy; treating any ACK as confirmation delivers a file
     with holes in it.

  2. The timeout is an absolute deadline, not a fresh countdown. Restarting
     the clock every time some unrelated datagram arrives means a busy link
     can keep the sender waiting indefinitely.

  3. Karn's algorithm: never measure RTT from a packet that was
     retransmitted. When you have sent seq=7 twice and an ACK for 7 comes
     back, there is no way to know which transmission it answers. The
     sample is ambiguous, so it is discarded and the timeout is doubled
     instead.

  4. The transfer has to end. A FIN flag on the last data packet tells the
     receiver where the file stops; without it the receiver would wait
     forever for a packet that is never coming.
"""

import os
import socket
import threading
import time

from rdt.constants import (
    DEFAULT_CHANNEL_PORT,
    DEFAULT_WINDOW,
    MAX_GBN_WINDOW,
    FLAG_DATA,
    FLAG_FIN,
    INITIAL_TIMEOUT,
    MAX_RETRIES,
    MSS,
    RECV_BUFFER,
)
from rdt.metrics import Metrics
from rdt.packet import Packet, PacketError
from rdt.rtt import RTTEstimator


def _suppress_icmp_reset(sock: socket.socket):
    """Stop Windows raising ConnectionResetError on UDP sockets.

    When a datagram reaches a host with nothing bound to the target port, the
    host replies with ICMP port-unreachable. Windows reports that as an
    exception on the socket's next recv, which is surprising on a connectionless
    protocol — Linux ignores it entirely. SIO_UDP_CONNRESET turns the behaviour
    off so the code behaves the same on both platforms.
    """
    if hasattr(socket, "SIO_UDP_CONNRESET"):  # Windows only
        try:
            sock.ioctl(socket.SIO_UDP_CONNRESET, False)
        except OSError:
            pass


class TransferFailed(RuntimeError):
    """Gave up after MAX_RETRIES consecutive attempts on one packet."""


class BaseSender:
    """Socket plumbing, file chunking, ACK reception, and RTT estimation."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_CHANNEL_PORT,
        timeout: float = INITIAL_TIMEOUT,
        adaptive: bool = True,
        verbose: bool = False,
    ):
        self.dest = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _suppress_icmp_reset(self.sock)
        self.verbose = verbose
        self.metrics = Metrics()

        # adaptive=False pins the timeout, which is what the benchmark uses
        # to show what the estimator is actually buying.
        self.adaptive = adaptive
        self.fixed_timeout = timeout
        self.rtt = RTTEstimator(rto=timeout)

    # -- timeout ------------------------------------------------------------

    @property
    def timeout(self) -> float:
        return self.rtt.timeout if self.adaptive else self.fixed_timeout

    def _on_rtt_sample(self, sample: float):
        self.metrics.record_rtt(sample)
        if self.adaptive:
            self.rtt.update(sample)

    def _on_timeout(self):
        self.metrics.timeouts += 1
        if self.adaptive:
            self.rtt.backoff()

    # -- io -----------------------------------------------------------------

    def _log(self, msg: str):
        if self.verbose:
            print(f"[send] {msg}")

    def _chunks(self, path: str):
        """Yield (seq, payload, is_last). Reads one chunk ahead so the final
        packet can be flagged FIN without loading the whole file."""
        size = os.path.getsize(path)
        total = max(1, (size + MSS - 1) // MSS)  # an empty file is one packet

        with open(path, "rb") as fh:
            for seq in range(total):
                yield seq, fh.read(MSS), seq == total - 1

    def _transmit(self, pkt: Packet, *, retransmission: bool) -> int:
        wire = pkt.to_bytes()
        self.sock.sendto(wire, self.dest)
        self.metrics.record_sent(len(wire), retransmission=retransmission)
        return len(wire)

    def _await_ack(self, wanted: int, deadline: float) -> bool:
        """Wait for an ACK matching `wanted`, until the absolute `deadline`.

        Returns True if it arrived. Datagrams that are corrupt, are not ACKs,
        or acknowledge a different sequence number consume time but do not
        extend the deadline — see trap 2.
        """
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False

            self.sock.settimeout(remaining)
            try:
                raw, _ = self.sock.recvfrom(RECV_BUFFER)
            except socket.timeout:
                return False
            except ConnectionResetError:
                # Windows surfaces an ICMP "port unreachable" from a previous
                # send as an exception on the *next* recv. Nothing is wrong
                # with this socket — the destination just was not listening.
                # Treat it as silence and let the deadline decide.
                self._log("icmp port-unreachable (is the channel running?)")
                continue

            try:
                pkt = Packet.from_bytes(raw)
            except PacketError:
                self.metrics.corrupt_received += 1
                continue

            if not pkt.is_ack:
                continue

            self.metrics.acks_received += 1

            if pkt.ack == wanted:
                return True

            # Trap 1: a stale ACK from an earlier retransmission.
            self.metrics.duplicate_acks += 1
            self._log(f"stale ack={pkt.ack} while waiting for {wanted}")

    def close(self):
        self.sock.close()


class StopAndWaitSender(BaseSender):
    """One packet in flight at a time.

    Utilisation is dismal on any link with real latency: the sender spends
    one RTT idle for every packet it sends. On a 1 Gbps link with 30 ms RTT
    and 1 KB packets, that is roughly 0.03% of capacity. Everything Go-Back-N
    and Selective Repeat do is an attempt to fix precisely this.
    """

    def send_file(self, path: str) -> Metrics:
        self.metrics.start()

        try:
            for seq, payload, is_last in self._chunks(path):
                flags = FLAG_DATA | (FLAG_FIN if is_last else 0)
                pkt = Packet(seq=seq, flags=flags, payload=payload)

                for attempt in range(MAX_RETRIES):
                    timeout = self.timeout
                    sent_at = time.monotonic()
                    self._transmit(pkt, retransmission=attempt > 0)
                    self._log(
                        f"seq={seq} {'FIN ' if is_last else ''}"
                        f"{len(payload)}B attempt={attempt + 1} "
                        f"rto={timeout * 1000:.0f}ms"
                    )

                    if self._await_ack(seq, sent_at + timeout):
                        # Trap 3: only an unambiguous transmission yields a
                        # usable RTT sample.
                        if attempt == 0:
                            self._on_rtt_sample(time.monotonic() - sent_at)
                        self.metrics.payload_bytes_delivered += len(payload)
                        self._log(f"seq={seq} acked  [{self.rtt}]")
                        break

                    self._on_timeout()
                    self._log(
                        f"seq={seq} timeout, rto now {self.timeout * 1000:.0f}ms"
                    )
                else:
                    raise TransferFailed(
                        f"no ack for seq={seq} after {MAX_RETRIES} attempts"
                    )
        finally:
            self.metrics.stop()
            self.close()

        return self.metrics


class GoBackNSender(BaseSender):
    """A window of N packets in flight, cumulative ACKs, one timer.

    The sender keeps two pointers into the sequence space:

        base      oldest packet sent but not yet acknowledged
        next_seq  next packet to be sent

    and may transmit freely while next_seq < base + N. Everything between
    base and next_seq is "in flight".

        already acked | . . . in flight . . . | not yet sent
                      ^base                   ^next_seq
                      |<------ window N ----->|

    ACKs are CUMULATIVE: an ACK for 7 means "I have everything through 7",
    so it acknowledges 5 and 6 as well. That makes lost ACKs cheap — the
    next one covers them — which is a real advantage over stop-and-wait,
    where every ACK loss costs a retransmission.

    There is ONE timer, attached to base. When it expires, the sender
    retransmits the entire in-flight window, not just the missing packet.
    That is the "go back N" the protocol is named for, and it is the
    weakness: one loss forces the resending of up to N-1 packets that
    arrived perfectly well. At high loss rates it collapses.

    Structurally this is the first piece of the project that cannot be a
    simple blocking loop. The sender has to transmit and receive at the
    same time, so ACK handling runs on its own thread and the window state
    is guarded by a condition variable.
    """

    def __init__(self, *args, window: int = DEFAULT_WINDOW, **kwargs):
        super().__init__(*args, **kwargs)
        if not 1 <= window <= MAX_GBN_WINDOW:
            raise ValueError(f"window must be 1..{MAX_GBN_WINDOW}")
        self.window = window

        # Window state, all guarded by self._cv.
        self._cv = threading.Condition()
        self.base = 0
        self.next_seq = 0
        self._timer_start: float | None = None
        self._send_times: dict[int, float] = {}
        self._retransmitted: set[int] = set()
        self._consecutive_timeouts = 0
        self._running = True

    # -- ack thread ---------------------------------------------------------

    def _ack_listener(self, sizes: list[int]):
        """Drain ACKs continuously and slide the window forward."""
        while True:
            with self._cv:
                if not self._running:
                    return

            self.sock.settimeout(0.1)
            try:
                raw, _ = self.sock.recvfrom(RECV_BUFFER)
            except socket.timeout:
                continue
            except ConnectionResetError:
                continue  # Windows ICMP noise, see _suppress_icmp_reset
            except OSError:
                return  # socket closed during shutdown

            try:
                pkt = Packet.from_bytes(raw)
            except PacketError:
                with self._cv:
                    self.metrics.corrupt_received += 1
                continue

            if not pkt.is_ack:
                continue

            with self._cv:
                self.metrics.acks_received += 1

                if pkt.ack < self.base:
                    # Already covered by an earlier cumulative ACK. Harmless,
                    # but worth counting: a run of these is the signature of
                    # a lost packet at the head of the window.
                    self.metrics.duplicate_acks += 1
                    continue

                # Karn's algorithm: sample only if this exact packet went out
                # once. A cumulative ACK covers several packets, so check the
                # one it names directly.
                if pkt.ack not in self._retransmitted and pkt.ack in self._send_times:
                    self._on_rtt_sample(time.monotonic() - self._send_times[pkt.ack])

                for seq in range(self.base, pkt.ack + 1):
                    if seq < len(sizes):
                        self.metrics.payload_bytes_delivered += sizes[seq]

                self.base = pkt.ack + 1
                self._consecutive_timeouts = 0

                # Restart the timer for the new base, or stop it if the
                # window is empty.
                self._timer_start = (
                    time.monotonic() if self.base < self.next_seq else None
                )
                self._log(f"ack={pkt.ack} cumulative, base -> {self.base}")
                self._cv.notify_all()

    # -- main loop ----------------------------------------------------------

    def send_file(self, path: str) -> Metrics:
        chunks = list(self._chunks(path))
        sizes = [len(payload) for _, payload, _ in chunks]
        total = len(chunks)

        self.metrics.start()
        listener = threading.Thread(
            target=self._ack_listener, args=(sizes,), daemon=True
        )
        listener.start()

        try:
            with self._cv:
                while self.base < total:
                    # 1. Fill the window.
                    while self.next_seq < self.base + self.window and self.next_seq < total:
                        seq, payload, is_last = chunks[self.next_seq]
                        flags = FLAG_DATA | (FLAG_FIN if is_last else 0)
                        pkt = Packet(seq=seq, flags=flags, payload=payload)

                        self._transmit(pkt, retransmission=False)
                        self._send_times[seq] = time.monotonic()

                        if self._timer_start is None:
                            self._timer_start = time.monotonic()

                        self._log(
                            f"seq={seq} sent  [base={self.base} "
                            f"next={self.next_seq + 1} win={self.window}]"
                        )
                        self.next_seq += 1

                    if self.base >= total:
                        break

                    # 2. Wait for the timer or for the window to advance.
                    rto = self.timeout
                    elapsed = time.monotonic() - (self._timer_start or time.monotonic())
                    remaining = rto - elapsed

                    if remaining > 0:
                        self._cv.wait(remaining)
                        continue

                    # 3. Timer expired: go back N.
                    self._on_timeout()
                    self._consecutive_timeouts += 1
                    if self._consecutive_timeouts > MAX_RETRIES:
                        raise TransferFailed(
                            f"no progress past seq={self.base} after "
                            f"{MAX_RETRIES} window retransmissions"
                        )

                    self._log(
                        f"TIMEOUT base={self.base}, resending "
                        f"{self.next_seq - self.base} packets, "
                        f"rto now {self.timeout * 1000:.0f}ms"
                    )

                    for seq in range(self.base, self.next_seq):
                        _, payload, is_last = chunks[seq]
                        flags = FLAG_DATA | (FLAG_FIN if is_last else 0)
                        self._transmit(
                            Packet(seq=seq, flags=flags, payload=payload),
                            retransmission=True,
                        )
                        self._retransmitted.add(seq)

                    self._timer_start = time.monotonic()
        finally:
            with self._cv:
                self._running = False
                self._cv.notify_all()
            listener.join(timeout=1.0)
            self.metrics.stop()
            self.close()

        return self.metrics