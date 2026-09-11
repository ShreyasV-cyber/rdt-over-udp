"""
Packet encoding and decoding for rdt-over-udp.

Wire format (16-byte header, big-endian):

     0                   1                   2                   3
    +-------------------------------+-------------------------------+
    |                        Sequence Number                        |
    +---------------------------------------------------------------+
    |                     Acknowledgment Number                     |
    +---------------+---------------+-------------------------------+
    |     Flags     |   Reserved    |           Checksum            |
    +---------------+---------------+-------------------------------+
    |                        Payload Length                         |
    +---------------------------------------------------------------+
    |                          Payload ...                          |
"""

import struct
from dataclasses import dataclass, field

from rdt.constants import (
    FLAG_ACK,
    FLAG_DATA,
    FLAG_FIN,
    HEADER_FORMAT,
    HEADER_SIZE,
    MSS,
    SEQ_SPACE,
)


class PacketError(ValueError):
    """Base class for anything wrong with a received datagram."""


class TruncatedPacketError(PacketError):
    """Datagram is too short to contain a valid header + payload."""


class ChecksumError(PacketError):
    """Header and payload do not match the checksum field — bits were flipped.

    Raised separately from TruncatedPacketError so the receiver can count
    corruption and loss as distinct events.
    """


def checksum(data: bytes) -> int:
    """16-bit one's-complement Internet checksum (RFC 1071).

    The same algorithm IP, TCP and UDP use. Sum the data as big-endian 16-bit
    words, fold any carry back into the low 16 bits, then invert.

    Note this is a *detection* code, not a correction code, and a weak one:
    it cannot catch a reordering of 16-bit words, or two errors that cancel
    out. That is a known and accepted property of the Internet checksum — it
    is cheap enough to compute in software on every hop, which is why it won
    over stronger codes like CRC-32 at this layer.
    """
    # An odd-length buffer is padded with a zero byte for the purposes of the
    # calculation only. The pad is not transmitted.
    if len(data) % 2:
        data += b"\x00"

    total = 0
    for i in range(0, len(data), 2):
        total += (data[i] << 8) | data[i + 1]

    # Fold carries. Twice, because the first fold can itself produce a carry.
    total = (total & 0xFFFF) + (total >> 16)
    total = (total & 0xFFFF) + (total >> 16)

    return (~total) & 0xFFFF


@dataclass
class Packet:
    """One protocol data unit.

    seq is meaningful on DATA packets, ack on ACK packets. Both fields are
    always present on the wire; the unused one is zero.
    """

    seq: int = 0
    ack: int = 0
    flags: int = 0
    payload: bytes = field(default=b"")

    # -- constructors -------------------------------------------------------

    @classmethod
    def make_data(cls, seq: int, payload: bytes) -> "Packet":
        if len(payload) > MSS:
            raise ValueError(f"payload {len(payload)}B exceeds MSS {MSS}B")
        return cls(seq=seq % SEQ_SPACE, flags=FLAG_DATA, payload=payload)

    @classmethod
    def make_ack(cls, ack: int) -> "Packet":
        return cls(ack=ack % SEQ_SPACE, flags=FLAG_ACK)

    @classmethod
    def make_fin(cls, seq: int) -> "Packet":
        return cls(seq=seq % SEQ_SPACE, flags=FLAG_FIN)

    # -- flag predicates ----------------------------------------------------

    @property
    def is_data(self) -> bool:
        return bool(self.flags & FLAG_DATA)

    @property
    def is_ack(self) -> bool:
        return bool(self.flags & FLAG_ACK)

    @property
    def is_fin(self) -> bool:
        return bool(self.flags & FLAG_FIN)

    # -- serialisation ------------------------------------------------------

    def to_bytes(self) -> bytes:
        """Serialise to a datagram, computing the checksum over the result."""
        # Build the header once with the checksum field zeroed, checksum the
        # whole thing, then rebuild with the real value. Zeroing the field is
        # what makes the checksum verifiable at the other end: the receiver
        # performs the identical zeroing before recomputing.
        blank_header = struct.pack(
            HEADER_FORMAT,
            self.seq,
            self.ack,
            self.flags,
            0,  # reserved
            0,  # checksum placeholder
            len(self.payload),
        )
        csum = checksum(blank_header + self.payload)

        header = struct.pack(
            HEADER_FORMAT,
            self.seq,
            self.ack,
            self.flags,
            0,
            csum,
            len(self.payload),
        )
        return header + self.payload

    @classmethod
    def from_bytes(cls, data: bytes) -> "Packet":
        """Parse a datagram. Raises PacketError if it is malformed or corrupt."""
        if len(data) < HEADER_SIZE:
            raise TruncatedPacketError(
                f"got {len(data)}B, need at least {HEADER_SIZE}B for a header"
            )

        seq, ack, flags, reserved, csum, length = struct.unpack(
            HEADER_FORMAT, data[:HEADER_SIZE]
        )
        payload = data[HEADER_SIZE:]

        # A corrupted length field can claim more payload than arrived. Check
        # before trusting it, so a bit flip is reported as corruption rather
        # than silently producing a short packet.
        if len(payload) != length:
            raise TruncatedPacketError(
                f"header claims {length}B payload, datagram carries {len(payload)}B"
            )

        blank_header = struct.pack(
            HEADER_FORMAT, seq, ack, flags, reserved, 0, length
        )
        if checksum(blank_header + payload) != csum:
            raise ChecksumError(f"checksum mismatch on seq={seq} ack={ack}")

        return cls(seq=seq, ack=ack, flags=flags, payload=payload)

    # -- debugging ----------------------------------------------------------

    def __repr__(self) -> str:
        names = []
        if self.is_data:
            names.append("DATA")
        if self.is_ack:
            names.append("ACK")
        if self.is_fin:
            names.append("FIN")
        tag = "|".join(names) or "NONE"

        if self.is_ack and not self.is_data:
            return f"<{tag} ack={self.ack}>"
        return f"<{tag} seq={self.seq} len={len(self.payload)}>"