"""
Receivers.

Stop-and-wait only for now; Go-Back-N and Selective Repeat join it here
later and share BaseReceiver.

The receiver's job is smaller than the sender's but has three traps in it:

  1. A corrupted packet must be discarded *silently*. It is tempting to send
     a NAK or re-ACK, but the sequence number lives inside the corruption —
     there is nothing trustworthy to respond about.

  2. A duplicate must be re-ACKed, never ignored. A duplicate means the
     sender never heard the original ACK. Staying quiet leaves it
     retransmitting the same packet until it gives up.

  3. The last ACK cannot be confirmed. The receiver has no way to know
     whether its final ACK arrived, so it lingers, answering retransmissions
     for a while before closing. This is the same problem TCP solves with
     TIME_WAIT, and it has no perfect solution — see the Two Generals
     problem.
"""

import socket
import time

from rdt.constants import (
    DEFAULT_RECEIVER_PORT,
    RECV_BUFFER,
)
from rdt.metrics import Metrics
from rdt.packet import Packet, PacketError

# How long to keep answering retransmissions after the transfer looks done.
# Long enough to cover a few sender timeouts; short enough not to hang.
LINGER_SECONDS = 2.0


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


class BaseReceiver:
    """Socket plumbing, ACK sending, and output file handling."""

    def __init__(
        self,
        output_path: str,
        port: int = DEFAULT_RECEIVER_PORT,
        host: str = "0.0.0.0",
        verbose: bool = False,
    ):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _suppress_icmp_reset(self.sock)
        self.sock.bind((host, port))
        self.output_path = output_path
        self.verbose = verbose
        self.metrics = Metrics()
        self._out = open(output_path, "wb")

    # -- io -----------------------------------------------------------------

    def _log(self, msg: str):
        if self.verbose:
            print(f"[recv] {msg}")

    def _recv(self, timeout: float | None = None):
        """Return (packet, addr), or (None, addr) if the datagram was corrupt,
        or (None, None) on timeout."""
        self.sock.settimeout(timeout)
        try:
            data, addr = self.sock.recvfrom(RECV_BUFFER)
        except socket.timeout:
            return None, None
        except ConnectionResetError:
            # An ACK we sent bounced: the channel or sender has gone away.
            # Windows-only noise, not a failure of this socket.
            self._log("icmp port-unreachable on ack path")
            return None, None

        try:
            pkt = Packet.from_bytes(data)
        except PacketError as exc:
            self.metrics.corrupt_received += 1
            self._log(f"corrupt datagram discarded ({exc})")
            return None, addr

        self.metrics.packets_received += 1
        return pkt, addr

    def _send_ack(self, seq: int, addr):
        self.sock.sendto(Packet.make_ack(seq).to_bytes(), addr)
        self.metrics.acks_sent += 1

    def _deliver(self, payload: bytes):
        self._out.write(payload)
        self.metrics.payload_bytes_delivered += len(payload)

    def close(self):
        if not self._out.closed:
            self._out.close()
        self.sock.close()


class StopAndWaitReceiver(BaseReceiver):
    """Accept exactly one sequence number at a time.

    Window of 1, so the entire receiver state is a single integer: the
    sequence number it is waiting for. Anything else is a duplicate.
    """

    def run(self) -> Metrics:
        expected = 0
        last_addr = None
        finished = False

        self.metrics.start()

        try:
            while not finished:
                pkt, addr = self._recv(timeout=None)

                if pkt is None:
                    # Corrupt datagram. Say nothing — see trap 1 above.
                    continue

                last_addr = addr

                if not (pkt.is_data or pkt.is_fin):
                    continue  # not ours to handle

                if pkt.seq == expected:
                    self._deliver(pkt.payload)
                    self._send_ack(expected, addr)
                    self._log(
                        f"seq={pkt.seq} in order, delivered {len(pkt.payload)}B, "
                        f"ack={expected}"
                    )
                    expected += 1

                    if pkt.is_fin:
                        finished = True
                else:
                    # Trap 2: the sender never heard our previous ACK, so it
                    # resent. Re-ACK the last thing we did receive in order.
                    self.metrics.duplicates_received += 1
                    self._send_ack(expected - 1, addr)
                    self._log(
                        f"seq={pkt.seq} duplicate (expecting {expected}), "
                        f"re-ack={expected - 1}"
                    )

            self._linger(expected, last_addr)

        finally:
            self.metrics.stop()
            self.close()

        return self.metrics

    def _linger(self, expected: int, addr):
        """Trap 3: keep answering retransmissions for a while after the FIN.

        If our final ACK was lost, the sender is sitting there retransmitting
        the FIN. Closing immediately would leave it to time out and declare
        failure on a transfer that actually succeeded.
        """
        deadline = time.monotonic() + LINGER_SECONDS
        self._log(f"transfer complete, lingering {LINGER_SECONDS}s for retransmissions")

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            pkt, from_addr = self._recv(timeout=remaining)
            if pkt is None:
                continue

            if pkt.is_data or pkt.is_fin:
                self.metrics.duplicates_received += 1
                self._send_ack(min(pkt.seq, expected - 1), from_addr or addr)
                self._log(f"seq={pkt.seq} retransmission during linger, re-acked")