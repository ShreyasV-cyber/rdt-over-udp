"""
Senders.

Stop-and-wait only for now; Go-Back-N and Selective Repeat join it here
later and share BaseSender.

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
import time

from rdt.constants import (
    BACKOFF_FACTOR,
    DEFAULT_CHANNEL_PORT,
    FLAG_DATA,
    FLAG_FIN,
    INITIAL_TIMEOUT,
    MAX_RETRIES,
    MAX_TIMEOUT,
    MSS,
    RECV_BUFFER,
)
from rdt.metrics import Metrics
from rdt.packet import Packet, PacketError


class TransferFailed(RuntimeError):
    """Gave up after MAX_RETRIES consecutive attempts on one packet."""


class BaseSender:
    """Socket plumbing, file chunking, and ACK reception."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_CHANNEL_PORT,
        timeout: float = INITIAL_TIMEOUT,
        verbose: bool = False,
    ):
        self.dest = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.base_timeout = timeout
        self.verbose = verbose
        self.metrics = Metrics()

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

                timeout = self.base_timeout

                for attempt in range(MAX_RETRIES):
                    sent_at = time.monotonic()
                    self._transmit(pkt, retransmission=attempt > 0)
                    self._log(
                        f"seq={seq} {'FIN ' if is_last else ''}"
                        f"{len(payload)}B attempt={attempt + 1} "
                        f"timeout={timeout:.3f}s"
                    )

                    if self._await_ack(seq, sent_at + timeout):
                        # Trap 3: only an unambiguous transmission yields a
                        # usable RTT sample.
                        if attempt == 0:
                            self.metrics.record_rtt(time.monotonic() - sent_at)
                        self.metrics.payload_bytes_delivered += len(payload)
                        self._log(f"seq={seq} acked")
                        break

                    self.metrics.timeouts += 1
                    timeout = min(timeout * BACKOFF_FACTOR, MAX_TIMEOUT)
                    self._log(f"seq={seq} timeout, backing off to {timeout:.3f}s")
                else:
                    raise TransferFailed(
                        f"no ack for seq={seq} after {MAX_RETRIES} attempts"
                    )
        finally:
            self.metrics.stop()
            self.close()

        return self.metrics