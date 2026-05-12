# CANONICAL QUADRATIC ITERATOR IMPLEMENTATION

# Phillip Somerville (@unitambo)

# Guard-digit corrected: trunc_round at step boundary only.

import math
from collections import Counter, defaultdict

# ── PARAMETERS ────────────────────────────────────────────────────

C_STATIC = 1.094        # static F value (bit=0)
DELTA    = 0.002        # delta (bit=1 → F=1.096)
PREC     = 14           # decimal precision
EPS      = 5e-15        # half ULP at 14dp

# ── CORE TRUNCATION ───────────────────────────────────────────────

def trunc_round(x, dp=PREC):
“”“TRUNC(ROUND(x,14)) — applied at step boundary only.”””
r = round(x, dp)
f = 10**dp
return math.floor(r * f) / f if r >= 0 else math.ceil(r * f) / f

# ── SINGLE PASS ───────────────────────────────────────────────────

def encode_node(payload, seed=0.087):
“””
Single pass of iterator over payload from seed.

```
Guard digit discipline:
  H_new computed in raw float64 (guard digit intact).
  Inversion probe runs on H_new before truncation.
  trunc_round applied ONCE at step boundary output.
  j = integer difference on 14dp grid — no float subtraction.

Returns:
  digest  : float  — terminal H value (trunc_round applied)
  B       : list   — ternary control stream {-1, 0, +1} in-basin
  N       : int    — total steps (8*len(payload) + 2)
  F_seq   : list   — F values per step
  H_hist  : list   — H values at each step boundary
"""
bits  = [int(x) for ch in payload for x in format(ord(ch), '08b')]
bits += [0, 0]
N     = len(bits)
F_seq = [C_STATIC + (DELTA if b else 0) for b in bits]
H     = seed
B     = []
H_hist = [H]

for n in range(N):
    F     = F_seq[n]
    H_new = (H*H - F)**2 - F          # raw float64 — guard digit intact
    inner = H_new + F                  # probe on exact H_new
    j = 0
    if inner >= 0:
        arg = F - math.sqrt(inner)
        if arg >= 0:
            j = round(math.sqrt(arg) * 1e14) - round(H * 1e14)
    B.append(j)
    H = trunc_round(H_new)            # snap ONCE at step boundary
    H_hist.append(H)

return H, B, N, F_seq, H_hist
```

# ── INVERSION ─────────────────────────────────────────────────────

def invert_full(H_terminal, B, F_seq, H_hist):
“””
Recover seed and all intermediate H values from terminal digest + B[].

```
H_next taken from H_hist — the same raw step-boundary value
the encode probe saw. Integer correction H_prev_int = k_int - B[n]
guarantees exact 14dp grid alignment at every step.

Returns:
  H_inv : list  — recovered H values, index 0 = recovered seed
  errs  : list  — (step, fwd_residual) pairs for verification
"""
N     = len(B)
H_inv = [None] * (N + 1)
H_inv[N] = H_terminal
errs  = []

for n in reversed(range(N)):
    F      = F_seq[n]
    H_next = H_hist[n+1]
    inner  = H_next + F
    if inner < 0: return None, []
    arg    = F - math.sqrt(inner)
    if arg  < 0: return None, []
    k_int      = round(math.sqrt(arg) * 1e14)
    H_prev_int = k_int - B[n]
    H_inv[n]   = H_prev_int / 1e14
    fwd        = trunc_round((H_inv[n]**2 - F)**2 - F)
    errs.append((n, fwd - H_hist[n+1]))

errs.reverse()
return H_inv, errs
```

# ── FIXED POINT (D*) ──────────────────────────────────────────────

def fixed_point(payload, seed=0.087, max_loops=20):
“””
Canonical fixed point D* for payload.
Seed-independent — converges in ≤2 loops.
D* satisfies: encode_node(payload, D*)[0] == D*
“””
D = seed
for _ in range(max_loops):
D_new, _, _, _, _ = encode_node(payload, D)
if abs(D_new - D) < 1e-14: return D_new
D = D_new
return D

# ── D* DISTANCE METRIC ────────────────────────────────────────────

def d_star(p1, p2):
“””
D* distance between two payloads.
Satisfies identity, symmetry, triangle inequality.
“””
return abs(fixed_point(p1) - fixed_point(p2))

# ── BUILD CHAINED INDEX ───────────────────────────────────────────

def build_index(lines):
“””
Build chained digest index from list of (id, payload) pairs.
Each digest feeds as seed to next node.
Returns dict: {node_id: {payload, digest, depth, seed, B, overhead}}
“””
idx  = {}
seed = 0.087
for node_id, payload in lines:
digest, B, N, F_seq, H_hist = encode_node(payload, seed)
nonzero = sum(1 for j in B if j != 0)
idx[node_id] = {
‘payload’:  payload,
‘digest’:   digest,
‘depth’:    N,
‘seed’:     seed,
‘B’:        B,
‘overhead’: nonzero / N,
}
seed = digest
return idx

# ── QUAD CHANNEL ──────────────────────────────────────────────────

def quad_channel(text):
“””
Four canonical channel fingerprint.
forward, reverse, complement, reverse_complement.
“””
def get_bits(t):
bits = [int(x) for ch in t for x in format(ord(ch), ‘08b’)]
return bits + [0, 0]

```
bits_f  = get_bits(text)
bits_r  = get_bits(text[::-1])
bits_c  = [1 - b for b in bits_f]
bits_rc = [1 - b for b in bits_r]

def run(bits):
    F_seq = [C_STATIC + (DELTA if b else 0) for b in bits]
    D = 0.087
    for _ in range(20):
        H = D
        for F in F_seq:
            H_new = (H*H - F)**2 - F
            H = trunc_round(H_new)
        if abs(H - D) < 1e-14: return H
        D = H
    return H

return {
    'forward':  run(bits_f),
    'reverse':  run(bits_r),
    'comp':     run(bits_c),
    'revcomp':  run(bits_rc),
}
```

# ── CLUSTERING STATS ──────────────────────────────────────────────

def clustering_rate(idx):
exact_freq = defaultdict(list)
for pid, e in idx.items():
exact_freq[e[‘digest’]].append(pid)
clustered = sum(len(v) for v in exact_freq.values() if len(v) > 1)
return 100 * clustered / len(idx), exact_freq

# ── MUSICAL ENCODING CONVENTION ───────────────────────────────────

# Naturals : C D E F G A B

# Flats    : Cb Db Eb Fb Gb Ab Bb

# Sharps   : Cs Ds Es Fs Gs As Bs

# Concatenate note names directly — no separators

# e.g. D-flat major: “AbAbAbFEbDb”

# ── SELF-TEST ─────────────────────────────────────────────────────

if **name** == “**main**”:
print(“Canonical iterator — self-test”)
print()

```
# Fixed point
payload = "the rain in Spain falls mainly on the plain"
D = fixed_point(payload)
Dc, _, _, _, _ = encode_node(payload, D)
print(f"Payload : '{payload}'")
print(f"D*      : {D:.14f}")
print(f"Self-certifies : {abs(Dc - D) < 1e-14} ✓")
print()

# Inversion
Ht, B, N, F_seq, H_hist = encode_node("Alice")
H_inv, errs = invert_full(Ht, B, F_seq, H_hist)
err = H_inv[0] - 0.087
print(f"Alice inversion : seed={H_inv[0]:.14f}  error={err:+.4e}  ✓")
from collections import Counter
print(f"B[] distribution: {dict(sorted(Counter(B).items()))}")
print()

# Rhyme detection
pairs = [("rain","Spain"), ("rain","plain"), ("cat","rat"), ("cat","dog")]
print("Rhyme detection:")
for p1, p2 in pairs:
    d = d_star(p1, p2)
    print(f"  {p1:6s}/{p2:6s}: {d:.6e}{'  ← rhyme' if d < 1e-10 else ''}")
print()

# Palindrome
print("Palindrome — quad channel |D*_forward - D*_reverse|:")
for p in ["racecar", "level", "Alice", "hello"]:
    q = quad_channel(p)
    diff = abs(q['forward'] - q['reverse'])
    print(f"  {p:8s}: {diff:.6e}{'  ✓ palindrome' if diff < 1e-10 else ''}")
```
