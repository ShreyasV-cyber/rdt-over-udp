"""
Shared constants for the rdt-over-udp project.

Every other module imports from here. If a number appears in two places,
it belongs in this file instead.
"""

# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------

# struct format string for the 16-byte header.
#   !  network byte order (big-endian), no padding
#   I  seq            (4 bytes, unsigned)
#   I  ack            (4 bytes, unsigned)
#   B  flags          (1 byte)
#   B  reserved       (1 byte, always 0 — keeps the header 4-byte aligned)
#   H  checksum       (2 bytes)
#   I  payload_length (4 bytes)
HEADER_FORMAT = "!IIBBHI"
HEADER_SIZE = 16

# Maximum segment size: bytes of payload per packet.
# 16 + 1024 = 1040 bytes on the wire, comfortably under a 1500-byte Ethernet
# MTU, so no IP fragmentation. Fragmentation would make loss measurements
# lie to us: one lost fragment kills the whole datagram.
MSS = 1024

# Largest datagram we ever expect to receive.
RECV_BUFFER = HEADER_SIZE + MSS

# ---------------------------------------------------------------------------
# Flags (bitfield — a packet can be both DATA and FIN)
# ---------------------------------------------------------------------------

FLAG_DATA = 0b0001
FLAG_ACK = 0b0010
FLAG_FIN = 0b0100

# ---------------------------------------------------------------------------
# Sequence number space
# ---------------------------------------------------------------------------

# Sequence numbers are per-packet (not per-byte like real TCP). Simpler to
# reason about and it matches how sliding windows are taught.
SEQ_BITS = 32
SEQ_SPACE = 2 ** SEQ_BITS

# Selective Repeat requires window <= SEQ_SPACE // 2, otherwise the receiver
# cannot distinguish a retransmission of an old packet from a new one after
# wraparound. Go-Back-N requires window <= SEQ_SPACE - 1.
# Tests override SEQ_BITS with a small value (e.g. 4) so wraparound is
# actually reachable in a short transfer.
MAX_SR_WINDOW = SEQ_SPACE // 2
MAX_GBN_WINDOW = SEQ_SPACE - 1

# ---------------------------------------------------------------------------
# Timers
# ---------------------------------------------------------------------------

# Used until the RTT estimator has its first sample.
INITIAL_TIMEOUT = 0.5  # seconds

# Clamps on the adaptive timeout. Too low and we retransmit packets that are
# merely in flight; too high and a real loss stalls the sender for seconds.
MIN_TIMEOUT = 0.05
MAX_TIMEOUT = 4.0

# Jacobson/Karels smoothing constants (RFC 6298).
#   EstimatedRTT = (1 - ALPHA) * EstimatedRTT + ALPHA * SampleRTT
#   DevRTT       = (1 - BETA)  * DevRTT       + BETA  * |SampleRTT - EstimatedRTT|
#   Timeout      = EstimatedRTT + K * DevRTT
RTT_ALPHA = 0.125
RTT_BETA = 0.25
RTT_K = 4

# Karn's algorithm: on timeout, double the timeout rather than taking a new
# RTT sample from an ambiguous (retransmitted) packet.
BACKOFF_FACTOR = 2.0

# Give up on the transfer after this many consecutive retransmissions of the
# same packet. Prevents an infinite loop when the receiver is simply gone.
MAX_RETRIES = 12

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_WINDOW = 8

DEFAULT_RECEIVER_PORT = 9000
DEFAULT_CHANNEL_PORT = 8000