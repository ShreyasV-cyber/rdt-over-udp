"""
Sliding window arithmetic and the sequence-space constraints.

Run from the repo root:  pytest tests/test_window.py -v

The interesting tests here are not the ones that check the code works. They
are the ones that demonstrate WHY the window size limits exist, by building
the failure and watching it happen.

Note on modelling: the real receivers use a 32-bit sequence space that no
test could realistically exhaust, so the wraparound tests below reimplement
the receiver's window classification as small pure functions over a tiny
sequence space. That is deliberate — the point is to show the arithmetic
breaking, and you cannot do that with numbers that never wrap.
"""

import pytest

from rdt.constants import MAX_GBN_WINDOW, MAX_SR_WINDOW, SEQ_SPACE
from rdt.receiver import SelectiveRepeatReceiver
from rdt.sender import GoBackNSender, SelectiveRepeatSender


# ---------------------------------------------------------------------------
# Models of the receiver's window classification, over a tiny sequence space
# ---------------------------------------------------------------------------


def is_new(seq: int, rcv_base: int, window: int, space: int) -> bool:
    """Would the receiver treat this as a packet it has not yet delivered?

    Acceptable range is [rcv_base, rcv_base + window - 1], modulo the space.
    """
    return (seq - rcv_base) % space < window


def is_already_delivered(seq: int, rcv_base: int, window: int, space: int) -> bool:
    """Would the receiver treat this as an old packet needing a re-ACK?

    Previous-window range is [rcv_base - window, rcv_base - 1], modulo the
    space. The receiver must answer these: staying silent leaves a sender
    whose ACK was lost retransmitting forever.
    """
    return 0 < (rcv_base - seq) % space <= window


# ---------------------------------------------------------------------------
# THE constraint: window <= half the sequence space
# ---------------------------------------------------------------------------


def test_sr_window_at_half_is_unambiguous():
    """With window == space / 2, every sequence number means exactly one thing."""
    space, window = 8, 4
    rcv_base = 0

    for seq in range(space):
        new = is_new(seq, rcv_base, window, space)
        old = is_already_delivered(seq, rcv_base, window, space)
        assert not (new and old), (
            f"seq={seq} classified as BOTH new and already-delivered"
        )


def test_sr_window_above_half_is_ambiguous():
    """With window > space / 2 the two ranges overlap, and the receiver has
    no way to tell a retransmission of an old packet from a new one.

    This is the failure the constraint exists to prevent. Concretely, with
    space=8 and window=5 at rcv_base=0:

        acceptable-as-new:   0 1 2 3 4
        already-delivered:   3 4 5 6 7      (that is, -5..-1 mod 8)
                             ^^^
        3 and 4 are in both. A retransmission of old packet 4 gets accepted
        as new data and written to the file. The output is corrupt, and
        nothing in the protocol detects it.
    """
    space, window = 8, 5
    rcv_base = 0

    ambiguous = [
        seq
        for seq in range(space)
        if is_new(seq, rcv_base, window, space)
        and is_already_delivered(seq, rcv_base, window, space)
    ]

    assert ambiguous, "expected an overlap — the constraint would be pointless"
    assert ambiguous == [3, 4]


@pytest.mark.parametrize("space", [4, 8, 16, 32, 64])
def test_half_the_space_is_exactly_the_boundary(space):
    """Sweep every window size: ambiguity appears the moment window exceeds
    space / 2, and never before. That is where the rule comes from — it is
    not a safety margin someone chose, it is the exact point of failure."""
    for window in range(1, space):
        overlap = any(
            is_new(seq, 0, window, space) and is_already_delivered(seq, 0, window, space)
            for seq in range(space)
        )
        assert overlap == (window > space // 2), (
            f"space={space} window={window}: overlap={overlap}, "
            f"expected {window > space // 2}"
        )


def test_gbn_tolerates_a_larger_window_than_sr():
    """Go-Back-N only needs window <= space - 1.

    Its receiver buffers nothing and accepts only the single next in-order
    packet, so it never has to distinguish an old retransmission from a new
    arrival — anything that is not exactly `expected` is discarded either
    way. Buffering is what costs Selective Repeat half its sequence space.
    """
    assert MAX_GBN_WINDOW == SEQ_SPACE - 1
    assert MAX_SR_WINDOW == SEQ_SPACE // 2
    assert MAX_GBN_WINDOW > MAX_SR_WINDOW


# ---------------------------------------------------------------------------
# Window advance
# ---------------------------------------------------------------------------


def slide_cumulative(base: int, ack: int) -> int:
    """Go-Back-N: a cumulative ACK moves base past everything it covers."""
    return ack + 1 if ack >= base else base


def slide_contiguous(base: int, acked: set[int]) -> int:
    """Selective Repeat: base advances only across an unbroken run."""
    while base in acked:
        base += 1
    return base


def test_cumulative_ack_covers_everything_below_it():
    """One ACK for 7 retires 5 and 6 as well — which is why Go-Back-N barely
    cares about lost ACKs."""
    assert slide_cumulative(base=5, ack=7) == 8


def test_stale_cumulative_ack_does_not_move_base_backwards():
    assert slide_cumulative(base=10, ack=4) == 10


def test_selective_repeat_base_stalls_on_a_gap():
    """Acks for 6 and 7 arrive, 5 is still missing. base must NOT move: the
    file is written in order, so packet 5 blocks everything behind it."""
    assert slide_contiguous(base=5, acked={6, 7}) == 5


def test_selective_repeat_base_jumps_when_the_gap_fills():
    """The moment 5 arrives, base leaps past the whole buffered run. This is
    the flush that Go-Back-N throws away instead."""
    assert slide_contiguous(base=5, acked={5, 6, 7}) == 8


def test_selective_repeat_stops_at_the_next_hole():
    assert slide_contiguous(base=5, acked={5, 6, 8, 9}) == 7


# ---------------------------------------------------------------------------
# Constructors enforce the limits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, MAX_SR_WINDOW + 1])
def test_sr_sender_rejects_bad_window(bad):
    with pytest.raises(ValueError):
        SelectiveRepeatSender(window=bad)


@pytest.mark.parametrize("bad", [0, -1, MAX_SR_WINDOW + 1])
def test_sr_receiver_rejects_bad_window(bad, tmp_path):
    with pytest.raises(ValueError):
        SelectiveRepeatReceiver(str(tmp_path / "out.bin"), port=0, window=bad)


@pytest.mark.parametrize("bad", [0, -1, MAX_GBN_WINDOW + 1])
def test_gbn_sender_rejects_bad_window(bad):
    with pytest.raises(ValueError):
        GoBackNSender(window=bad)


def test_window_of_one_is_legal():
    """Window 1 reduces both protocols to stop-and-wait — a useful sanity
    baseline for the benchmarks."""
    GoBackNSender(window=1).close()
    SelectiveRepeatSender(window=1).close()