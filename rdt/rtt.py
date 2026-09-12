"""
Adaptive retransmission timeout.

A fixed timeout is always wrong. Too short and the sender retransmits packets
that are merely in flight, adding load to a link that is already struggling.
Too long and every loss costs a long stall — which is exactly what made the
stop-and-wait run take 30 seconds instead of 3.

The right timeout is "a bit longer than a round trip," and since the round
trip changes constantly, it has to be measured continuously. This is the
Jacobson/Karels algorithm, as specified in RFC 6298 and used by every TCP
stack in existence:

    RTTVAR = (1 - BETA)  * RTTVAR + BETA  * |SRTT - sample|
    SRTT   = (1 - ALPHA) * SRTT   + ALPHA * sample
    RTO    = SRTT + K * RTTVAR

Two things are easy to get wrong:

  * RTTVAR must be updated BEFORE SRTT, because it needs the *old* SRTT to
    measure the deviation against. Swap the two lines and the deviation term
    collapses toward zero, the timeout hugs the mean, and any normal jitter
    triggers spurious retransmissions.

  * The K * RTTVAR term is what makes this work at all. A timeout set to the
    average RTT would fire on roughly half of all packets. Adding four
    standard deviations' worth of headroom is what keeps that from happening
    while still reacting quickly when the network genuinely slows down.
"""

from dataclasses import dataclass, field

from rdt.constants import (
    BACKOFF_FACTOR,
    INITIAL_TIMEOUT,
    MAX_TIMEOUT,
    MIN_TIMEOUT,
    RTT_ALPHA,
    RTT_BETA,
    RTT_K,
)


@dataclass
class RTTEstimator:
    """Tracks smoothed RTT and produces a retransmission timeout."""

    srtt: float = 0.0
    """Smoothed round-trip time."""

    rttvar: float = 0.0
    """Smoothed mean deviation — a cheap stand-in for standard deviation.
    Jacobson chose mean deviation over standard deviation specifically
    because it needs no square root, which mattered on 1980s hardware
    computing this for every segment."""

    rto: float = INITIAL_TIMEOUT
    """Current retransmission timeout, in seconds."""

    samples: int = 0

    history: list[tuple[float, float, float]] = field(default_factory=list)
    """(sample, srtt, rto) per update — useful for plotting convergence."""

    # -- updates ------------------------------------------------------------

    def update(self, sample: float):
        """Feed in one unambiguous RTT measurement.

        Only call this for packets that were transmitted exactly once. Karn's
        algorithm forbids sampling from a retransmitted packet: if seq=7 went
        out twice, an ACK for 7 could be answering either transmission, and
        guessing wrong corrupts the estimator in whichever direction you
        guessed.
        """
        if self.samples == 0:
            # RFC 6298 §2.2: seed from the first measurement rather than
            # from zero, so the estimator does not spend its first dozen
            # packets crawling up from nothing.
            self.srtt = sample
            self.rttvar = sample / 2
        else:
            # Order matters — rttvar uses the previous srtt.
            self.rttvar = (1 - RTT_BETA) * self.rttvar + RTT_BETA * abs(
                self.srtt - sample
            )
            self.srtt = (1 - RTT_ALPHA) * self.srtt + RTT_ALPHA * sample

        self.samples += 1
        self.rto = self._clamp(self.srtt + RTT_K * self.rttvar)
        self.history.append((sample, self.srtt, self.rto))

    def backoff(self):
        """Double the timeout after a loss.

        This is the other half of Karn's algorithm. A timeout means either
        the network is congested or the path got slower — retransmitting
        just as eagerly would make either situation worse. The doubling
        persists until a clean measurement arrives to replace it.
        """
        self.rto = self._clamp(self.rto * BACKOFF_FACTOR)

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _clamp(value: float) -> float:
        """Keep the timeout inside sane bounds.

        The floor matters more than it looks. On loopback the RTT is well
        under a millisecond, so the formula produces a timeout near zero and
        the sender retransmits packets that have not even arrived yet.
        Windows makes this worse: its timer granularity is around 15 ms, so
        sub-millisecond measurements are mostly noise.
        """
        return max(MIN_TIMEOUT, min(value, MAX_TIMEOUT))

    @property
    def timeout(self) -> float:
        return self.rto

    def summary(self) -> str:
        return (
            f"srtt={self.srtt * 1000:.1f}ms "
            f"rttvar={self.rttvar * 1000:.1f}ms "
            f"rto={self.rto * 1000:.1f}ms "
            f"({self.samples} samples)"
        )

    def __str__(self) -> str:
        return self.summary()