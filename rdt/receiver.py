"""
Receivers.

Stop-and-wait, Go-Back-N, and Selective Repeat, all sharing BaseReceiver.

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
    DEFAULT_WINDOW,
    MAX_SR_WINDOW,
    RECV_BUFFER,
)
from rdt.metrics import Metrics
from rdt.packet import Packet, PacketError

# How long to keep answering retransmissions after the transfer looks done.
#
# This is a deadline of SILENCE, not a fixed window: every retransmission we
# answer pushes it back. A sender that is backing off exponentially can take
# many seconds to exhaust its retries, and closing on a fixed 2s timer would
# fail a transfer that was one lost ACK away from finishing.
LINGER_SECONDS = 5.0


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
                deadline = time.monotonic() + LINGER_SECONDS  # they are still asking
                self._log(f"seq={pkt.seq} retransmission during linger, re-acked")


class GoBackNReceiver(BaseReceiver):
    """Accept only the next in-order packet. Buffer nothing.

    The Go-Back-N receiver is barely more complex than stop-and-wait, and
    that is the whole design trade: the sender does all the work so the
    receiver can stay cheap. Its entire state is one integer.

    Anything arriving out of order is DISCARDED, even though it is perfectly
    valid data that will have to be sent again. In exchange, the receiver
    needs no buffer and no bookkeeping — which mattered enormously when this
    protocol was designed and memory was the scarce resource.

    ACKs are cumulative: re-ACKing `expected - 1` tells the sender
    "everything through here is safe", whatever happened after it. So a lost
    ACK costs nothing as long as a later one gets through — unlike
    stop-and-wait, where every single ACK is load-bearing.

    Selective Repeat is what you get when you decide the discarding is too
    wasteful and give the receiver a buffer instead.
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
                    continue  # corrupt: stay silent, the sender will time out

                last_addr = addr

                if not (pkt.is_data or pkt.is_fin):
                    continue

                if pkt.seq == expected:
                    self._deliver(pkt.payload)
                    self._send_ack(expected, addr)
                    self._log(
                        f"seq={pkt.seq} in order, delivered "
                        f"{len(pkt.payload)}B, ack={expected}"
                    )
                    expected += 1

                    if pkt.is_fin:
                        finished = True

                elif pkt.seq > expected:
                    # Valid data, arrived too early. Thrown away — this is
                    # the waste Selective Repeat exists to eliminate.
                    self.metrics.out_of_order_received += 1
                    if expected > 0:
                        self._send_ack(expected - 1, addr)
                    self._log(
                        f"seq={pkt.seq} out of order (want {expected}), "
                        f"discarded, re-ack={expected - 1}"
                    )

                else:
                    # Already delivered. The sender never heard our ACK.
                    self.metrics.duplicates_received += 1
                    self._send_ack(expected - 1, addr)
                    self._log(
                        f"seq={pkt.seq} duplicate, re-ack={expected - 1}"
                    )

            self._linger(expected, last_addr)

        finally:
            self.metrics.stop()
            self.close()

        return self.metrics

    def _linger(self, expected: int, addr):
        """Same reasoning as stop-and-wait: our final ACK may not arrive."""
        deadline = time.monotonic() + LINGER_SECONDS
        self._log(f"transfer complete, lingering {LINGER_SECONDS}s")

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            pkt, from_addr = self._recv(timeout=remaining)
            if pkt is None:
                continue

            if pkt.is_data or pkt.is_fin:
                self.metrics.duplicates_received += 1
                self._send_ack(expected - 1, from_addr or addr)
                deadline = time.monotonic() + LINGER_SECONDS  # they are still asking
                self._log(f"seq={pkt.seq} retransmission during linger, re-acked")


class SelectiveRepeatReceiver(BaseReceiver):
    """Buffer out-of-order packets instead of discarding them.

    This is the only receiver in the project with real state. Go-Back-N
    throws away anything that is not the next packet it wants; this one
    keeps it, on the reasonable grounds that the sender went to the trouble
    of delivering it and will otherwise have to send it again.

        delivered | buffered window        | not yet acceptable
                  |  .  X  X  .  X  .  .   |
                  ^rcv_base
                  gap here blocks delivery, but X's are kept

    Data is still handed to the file strictly in order: when the gap at
    rcv_base is finally filled, that packet and every contiguous buffered
    packet behind it are written at once.

    THE SUBTLE PART -- re-ACKing the previous window:

    A packet whose sequence number falls in [rcv_base - N, rcv_base - 1] has
    already been delivered. The receiver must ACK it AGAIN anyway. If it
    stayed silent, the sender (whose ACK was lost) would keep retransmitting
    that packet until it gave up, and the transfer would fail on data that
    arrived correctly the first time.

    This is exactly why the window must be at most half the sequence space.
    The receiver has to be able to tell "old packet, already delivered,
    re-ACK it" apart from "new packet, accept it". With a window larger than
    half the space, those two ranges overlap after wraparound, and the
    receiver cannot distinguish them -- it delivers stale data as fresh.
    """

    def __init__(self, *args, window: int = DEFAULT_WINDOW, **kwargs):
        super().__init__(*args, **kwargs)
        if not 1 <= window <= MAX_SR_WINDOW:
            raise ValueError(f"window must be 1..{MAX_SR_WINDOW}")
        self.window = window

    def run(self) -> Metrics:
        rcv_base = 0
        buffer: dict[int, bytes] = {}
        fin_seq: int | None = None
        last_addr = None

        self.metrics.start()

        try:
            while True:
                pkt, addr = self._recv(timeout=None)

                if pkt is None:
                    continue  # corrupt: silence, the sender's timer handles it

                last_addr = addr

                if not (pkt.is_data or pkt.is_fin):
                    continue

                if pkt.is_fin:
                    fin_seq = pkt.seq

                seq = pkt.seq

                if rcv_base <= seq < rcv_base + self.window:
                    # Inside the window. ACK it individually whether or not
                    # it fills the gap -- the sender is tracking this exact
                    # packet's timer.
                    self._send_ack(seq, addr)

                    if seq in buffer:
                        self.metrics.duplicates_received += 1
                        self._log(f"seq={seq} already buffered, re-ack")
                        continue

                    buffer[seq] = pkt.payload

                    if seq != rcv_base:
                        self.metrics.out_of_order_received += 1
                        self._log(
                            f"seq={seq} buffered (want {rcv_base}), ack={seq}"
                        )
                    else:
                        # Gap filled: flush the contiguous run.
                        delivered = 0
                        while rcv_base in buffer:
                            self._deliver(buffer.pop(rcv_base))
                            rcv_base += 1
                            delivered += 1
                        self._log(
                            f"seq={seq} in order, flushed {delivered} "
                            f"packet(s), rcv_base -> {rcv_base}"
                        )

                elif rcv_base - self.window <= seq < rcv_base:
                    # Already delivered. Re-ACK anyway -- see the docstring.
                    self.metrics.duplicates_received += 1
                    self._send_ack(seq, addr)
                    self._log(f"seq={seq} already delivered, re-ack={seq}")

                else:
                    self._log(f"seq={seq} outside window, ignored")
                    continue

                if fin_seq is not None and rcv_base > fin_seq:
                    break

            self._linger(rcv_base, last_addr)

        finally:
            self.metrics.stop()
            self.close()

        return self.metrics

    def _linger(self, rcv_base: int, addr):
        deadline = time.monotonic() + LINGER_SECONDS
        self._log(f"transfer complete, lingering {LINGER_SECONDS}s")

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            pkt, from_addr = self._recv(timeout=remaining)
            if pkt is None:
                continue

            if pkt.is_data or pkt.is_fin:
                self.metrics.duplicates_received += 1
                self._send_ack(pkt.seq, from_addr or addr)
                deadline = time.monotonic() + LINGER_SECONDS  # they are still asking
                self._log(f"seq={pkt.seq} retransmission during linger, re-acked")