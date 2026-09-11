"""
Tests for rdt.packet.

Run from the repo root:  pytest tests/test_packet.py -v
"""

import random
import struct

import pytest

from rdt.constants import (
    FLAG_ACK,
    FLAG_DATA,
    FLAG_FIN,
    HEADER_FORMAT,
    HEADER_SIZE,
    MSS,
    SEQ_SPACE,
)
from rdt.packet import (
    ChecksumError,
    Packet,
    PacketError,
    TruncatedPacketError,
    checksum,
)


# ---------------------------------------------------------------------------
# Round-trip: what goes in comes out
# ---------------------------------------------------------------------------


def test_data_packet_round_trip():
    original = Packet.make_data(seq=42, payload=b"hello world")
    restored = Packet.from_bytes(original.to_bytes())

    assert restored.seq == 42
    assert restored.payload == b"hello world"
    assert restored.is_data
    assert not restored.is_ack
    assert not restored.is_fin


def test_ack_packet_round_trip():
    restored = Packet.from_bytes(Packet.make_ack(ack=99).to_bytes())

    assert restored.ack == 99
    assert restored.payload == b""
    assert restored.is_ack
    assert not restored.is_data


def test_fin_packet_round_trip():
    restored = Packet.from_bytes(Packet.make_fin(seq=7).to_bytes())

    assert restored.seq == 7
    assert restored.is_fin


def test_empty_payload_is_legal():
    restored = Packet.from_bytes(Packet.make_data(seq=1, payload=b"").to_bytes())
    assert restored.payload == b""


def test_full_mss_payload():
    payload = bytes(range(256)) * (MSS // 256)
    assert len(payload) == MSS

    restored = Packet.from_bytes(Packet.make_data(seq=1, payload=payload).to_bytes())
    assert restored.payload == payload


def test_binary_payload_survives():
    """Null bytes and high bytes must not be mangled anywhere in the path."""
    payload = b"\x00\xff\x00\xff\x80\x7f\x00"
    restored = Packet.from_bytes(Packet.make_data(seq=3, payload=payload).to_bytes())
    assert restored.payload == payload


def test_header_size_is_as_declared():
    """Guards against someone editing HEADER_FORMAT without HEADER_SIZE."""
    assert struct.calcsize(HEADER_FORMAT) == HEADER_SIZE
    assert len(Packet.make_ack(0).to_bytes()) == HEADER_SIZE


def test_odd_length_payload():
    """Exercises the zero-pad branch in the checksum."""
    restored = Packet.from_bytes(Packet.make_data(seq=1, payload=b"abc").to_bytes())
    assert restored.payload == b"abc"


# ---------------------------------------------------------------------------
# Field limits
# ---------------------------------------------------------------------------


def test_oversized_payload_rejected():
    with pytest.raises(ValueError):
        Packet.make_data(seq=1, payload=b"x" * (MSS + 1))


def test_sequence_number_wraps():
    """Constructors take seq modulo the sequence space, so callers can just
    increment a counter forever without overflowing the 32-bit field."""
    p = Packet.make_data(seq=SEQ_SPACE + 5, payload=b"x")
    assert p.seq == 5


# ---------------------------------------------------------------------------
# Corruption detection
# ---------------------------------------------------------------------------


def test_payload_bit_flip_is_caught():
    wire = bytearray(Packet.make_data(seq=1, payload=b"hello").to_bytes())
    wire[-1] ^= 0x01

    with pytest.raises(ChecksumError):
        Packet.from_bytes(bytes(wire))


def test_seq_bit_flip_is_caught():
    """Corruption in the header is as dangerous as corruption in the payload:
    a flipped seq would deliver correct data to the wrong slot."""
    wire = bytearray(Packet.make_data(seq=1, payload=b"hello").to_bytes())
    wire[0] ^= 0x80

    with pytest.raises(ChecksumError):
        Packet.from_bytes(bytes(wire))


def test_every_single_bit_flip_is_caught():
    """The Internet checksum detects all single-bit errors. Verify that
    property holds across every bit of a real packet, header included.

    The exception type depends on which field was hit: a flip in the length
    field trips the length guard before the checksum is ever compared, so
    this expects the PacketError base class rather than ChecksumError.
    """
    wire = Packet.make_data(seq=12345, payload=b"the quick brown fox").to_bytes()

    for byte_index in range(len(wire)):
        for bit in range(8):
            corrupted = bytearray(wire)
            corrupted[byte_index] ^= 1 << bit

            with pytest.raises(PacketError):
                Packet.from_bytes(bytes(corrupted))


def test_truncated_datagram_rejected():
    wire = Packet.make_data(seq=1, payload=b"hello").to_bytes()

    with pytest.raises(TruncatedPacketError):
        Packet.from_bytes(wire[:HEADER_SIZE - 1])


def test_lying_length_field_rejected():
    """Header claims more payload than the datagram actually carries."""
    p = Packet.make_data(seq=1, payload=b"hello")
    wire = bytearray(p.to_bytes())
    struct.pack_into("!I", wire, 12, 9999)  # payload_length lives at offset 12

    with pytest.raises(TruncatedPacketError):
        Packet.from_bytes(bytes(wire))


def test_random_multi_bit_corruption_mostly_caught():
    """Not a correctness guarantee — the Internet checksum is weak and some
    multi-bit errors cancel out. This pins the empirical detection rate so a
    regression in the checksum would show up as a sharp drop."""
    random.seed(1234)
    payload = bytes(random.getrandbits(8) for _ in range(200))
    wire = Packet.make_data(seq=77, payload=payload).to_bytes()

    all_bits = [(i, b) for i in range(len(wire)) for b in range(8)]
    trials, caught = 400, 0

    for _ in range(trials):
        corrupted = bytearray(wire)
        # Sample distinct (byte, bit) pairs. Flipping the same bit twice would
        # undo itself, leaving an uncorrupted packet — that is not a missed
        # detection, so it must not count against the rate.
        for idx, bit in random.sample(all_bits, random.randint(2, 6)):
            corrupted[idx] ^= 1 << bit

        try:
            Packet.from_bytes(bytes(corrupted))
        except PacketError:
            caught += 1

    # Not 100%, and deliberately so: flipping bit k of one 16-bit word up and
    # bit k of another word down cancels exactly in the one's-complement sum.
    # A genuinely broken checksum would score near zero, so this still catches
    # regressions.
    assert caught / trials > 0.95


# ---------------------------------------------------------------------------
# The checksum function itself
# ---------------------------------------------------------------------------


def test_checksum_is_deterministic():
    assert checksum(b"abcdef") == checksum(b"abcdef")


def test_checksum_fits_in_16_bits():
    for data in (b"", b"\x00", b"\xff" * 100, bytes(range(256))):
        assert 0 <= checksum(data) <= 0xFFFF


def test_checksum_detects_word_change():
    assert checksum(b"\x00\x01") != checksum(b"\x00\x02")


def test_checksum_carry_folding():
    """All-ones input maximises the carry chain — the case a single fold
    would get wrong."""
    assert 0 <= checksum(b"\xff" * 1024) <= 0xFFFF


def test_checksum_trailing_zero_byte_ambiguity():
    """Known property, not a bug: odd-length input is zero-padded, so 'abc'
    and 'abc\\x00' checksum identically. The length field in the header is
    what disambiguates them on the wire."""
    assert checksum(b"abc") == checksum(b"abc\x00")


# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------


def test_flags_are_a_bitfield():
    """A packet can carry more than one flag — the final DATA packet of a
    transfer will be DATA|FIN."""
    p = Packet(seq=1, flags=FLAG_DATA | FLAG_FIN, payload=b"last")
    restored = Packet.from_bytes(p.to_bytes())

    assert restored.is_data
    assert restored.is_fin
    assert not restored.is_ack


def test_flag_values_are_distinct_bits():
    assert FLAG_DATA & FLAG_ACK == 0
    assert FLAG_DATA & FLAG_FIN == 0
    assert FLAG_ACK & FLAG_FIN == 0