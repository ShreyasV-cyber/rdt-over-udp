"""
Counters for a transfer.

Both senders and receivers accumulate one of these. The benchmark scripts
read the derived properties at the bottom — those are what end up on the
axes of the plots in the README.

The distinction that matters here is goodput vs throughput:

    throughput = every byte pushed onto the wire / elapsed time
    goodput    = unique application bytes delivered / elapsed time

A protocol retransmitting furiously can have excellent throughput and
terrible goodput. Goodput is what the user actually gets, so it is the
number worth plotting.
"""

import json
import time
from dataclasses import asdict, dataclass, field


@dataclass
class Metrics:
    # -- sender side --------------------------------------------------------

    packets_sent: int = 0
    """Every transmission, first attempts and retransmissions alike."""

    retransmissions: int = 0
    """Subset of packets_sent that were repeat attempts."""

    bytes_sent: int = 0
    """Wire bytes, header included, counting retransmissions."""

    timeouts: int = 0

    acks_received: int = 0
    duplicate_acks: int = 0

    # -- receiver side ------------------------------------------------------

    packets_received: int = 0
    """Datagrams that parsed cleanly, including duplicates."""

    corrupt_received: int = 0
    """Failed the checksum or the length guard."""

    duplicates_received: int = 0
    """Already-delivered sequence numbers arriving again — almost always the
    result of a lost ACK, not a lost data packet."""

    out_of_order_received: int = 0
    """Arrived inside the window but not at its base. Go-Back-N discards
    these; Selective Repeat buffers them. The gap between those two numbers
    is most of the difference between the protocols."""

    acks_sent: int = 0

    # -- payload accounting -------------------------------------------------

    payload_bytes_delivered: int = 0
    """Unique application bytes handed to the file. Excludes duplicates."""

    # -- timing -------------------------------------------------------------

    rtt_samples: list[float] = field(default_factory=list)

    _started: float | None = field(default=None, repr=False)
    _stopped: float | None = field(default=None, repr=False)

    # -- recording ----------------------------------------------------------

    def start(self) -> "Metrics":
        self._started = time.monotonic()
        return self

    def stop(self) -> "Metrics":
        self._stopped = time.monotonic()
        return self

    def record_sent(self, wire_bytes: int, *, retransmission: bool = False):
        self.packets_sent += 1
        self.bytes_sent += wire_bytes
        if retransmission:
            self.retransmissions += 1

    def record_rtt(self, sample: float):
        """Only called for unambiguous samples. Karn's algorithm forbids
        taking an RTT measurement from a retransmitted packet, because there
        is no way to tell which transmission the ACK belongs to."""
        self.rtt_samples.append(sample)

    # -- derived ------------------------------------------------------------

    @property
    def elapsed(self) -> float:
        if self._started is None:
            return 0.0
        end = self._stopped if self._stopped is not None else time.monotonic()
        return max(end - self._started, 1e-9)

    @property
    def goodput_bps(self) -> float:
        """Unique application bits per second — what the user actually got."""
        return 8 * self.payload_bytes_delivered / self.elapsed

    @property
    def throughput_bps(self) -> float:
        """All wire bits per second, retransmissions included."""
        return 8 * self.bytes_sent / self.elapsed

    @property
    def efficiency(self) -> float:
        """Fraction of transmissions that were first attempts. 1.0 on a
        perfect link; falls as loss forces repeat work."""
        if self.packets_sent == 0:
            return 0.0
        return (self.packets_sent - self.retransmissions) / self.packets_sent

    @property
    def retransmission_ratio(self) -> float:
        if self.packets_sent == 0:
            return 0.0
        return self.retransmissions / self.packets_sent

    @property
    def avg_rtt(self) -> float:
        return sum(self.rtt_samples) / len(self.rtt_samples) if self.rtt_samples else 0.0

    @property
    def min_rtt(self) -> float:
        return min(self.rtt_samples) if self.rtt_samples else 0.0

    @property
    def max_rtt(self) -> float:
        return max(self.rtt_samples) if self.rtt_samples else 0.0

    # -- output -------------------------------------------------------------

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        d.pop("rtt_samples", None)
        d.update(
            elapsed=round(self.elapsed, 4),
            goodput_bps=round(self.goodput_bps, 1),
            throughput_bps=round(self.throughput_bps, 1),
            efficiency=round(self.efficiency, 4),
            retransmission_ratio=round(self.retransmission_ratio, 4),
            avg_rtt_ms=round(self.avg_rtt * 1000, 3),
            min_rtt_ms=round(self.min_rtt * 1000, 3),
            max_rtt_ms=round(self.max_rtt * 1000, 3),
            rtt_sample_count=len(self.rtt_samples),
        )
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def summary(self, label: str = "transfer") -> str:
        kbps = self.goodput_bps / 1000
        lines = [
            f"--- {label} ---",
            f"  elapsed            {self.elapsed:.3f} s",
            f"  delivered          {self.payload_bytes_delivered} B",
            f"  goodput            {kbps:.1f} kbps",
            f"  throughput         {self.throughput_bps / 1000:.1f} kbps",
        ]
        if self.packets_sent:
            lines += [
                f"  packets sent       {self.packets_sent}"
                f"  (retransmits {self.retransmissions},"
                f" {self.retransmission_ratio:.1%})",
                f"  efficiency         {self.efficiency:.1%}",
                f"  timeouts           {self.timeouts}",
                f"  acks received      {self.acks_received}"
                f"  (duplicate {self.duplicate_acks})",
            ]
        if self.packets_received:
            lines += [
                f"  packets received   {self.packets_received}",
                f"  corrupt            {self.corrupt_received}",
                f"  duplicates         {self.duplicates_received}",
                f"  out of order       {self.out_of_order_received}",
                f"  acks sent          {self.acks_sent}",
            ]
        if self.rtt_samples:
            lines.append(
                f"  rtt min/avg/max    {self.min_rtt * 1000:.1f} /"
                f" {self.avg_rtt * 1000:.1f} /"
                f" {self.max_rtt * 1000:.1f} ms"
                f"  ({len(self.rtt_samples)} samples)"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()