# Quadratic-Iterator
The Quadratic Iterator — A Procedural Index
Phillip Somerville (@unitambo)
unitambo.wordpress.com
What This Is
A bijective quadratic iterator of the form:
H[n] = TRUNC(ROUND((H[n-1]² − F[n])² − F[n], 14), 14)
F[n] ∈ {1.094, 1.096}
Applied sequentially across the bit representation of any payload string, the iterator
produces a canonical fixed-point digest D* — a content-addressed scalar that is seed-
independent, self-certifying, and losslessly invertible given the ternary control stream
generated during traversal.
This is not a hash function. It is a procedural index — the arithmetic instantiation of intention
as a computational primitive.
Key Properties
Zero parameters — no training, no learned weights, no corpus
Formally metric — D* distance satisfies identity, symmetry, triangle inequality
Self-certifying — every digest verifies itself via fixed-point loop
Losslessly invertible — full payload reconstruction from (D*, B[], N, seed)
Cross-domain — same construction indexes text, music, and classical poetry
Repository Contents
/iterator/
canonical-iterator.py Core implementation with self-test
Paste at start of new sessions
/essay/
digital-soft-machine-final.md "Towards The Digital Soft Machine"
Full essay — attention, intention,
motivation, ownership
/notes/
tech-note-universal-index-v2.md Strange Loops in Bijective Braidings
Stylometric analysis across Carroll,
Keats, Debussy, Bach
tech-note-quad-channel-v2.md Quadratic Information Portrait
Structural phase channels and
canonical symmetry channels
tensor-autocorr-addendum.md J-stream tensor autocorrelation
as third-order stylometric
Quick Start
from iterator.canonical_iterator import fixed_point, encode_node
# Find canonical digest
D = fixed_point("the rain in Spain falls mainly on the plain")
print(f"D* = {D:.14f}")
# Verify self-certification
D_check, _, _ = encode_node("the rain in Spain falls mainly on the plain", D)
print(f"Self-certifies: {abs(D_check - D) < 1e-14}")
# Rhyme detection
from iterator.canonical_iterator import d_star
print(f"|rain - Spain| = {d_star('rain', 'Spain'):.6e}") # near-zero
print(f"|rain - cat| = {d_star('rain', 'cat'):.6e}") # separated
The Essay
Towards The Digital Soft Machine argues that the transformer’s great structural limitation —
stateless forward passes, no accumulated intention, plasticity loss under continuous
training — is addressable by a complementary primitive: the procedural index.
The proposed architecture: AIM
Attention: what is relevant now — transformer
Intention: what persists across now — iterator
Motivation: what drives the traversal — open
The essay spans attention/intention theory, the encrostic forms of Su Hui’s Star Gauge and
the Iroha, stylometric analysis of Carroll/Keats/Debussy/Bach, a phenomenology of
consciousness as AI epistemology, and a political argument for ownership over centralised
AI deployment.
Technical Notes
Quad Channel — Four canonical transformations (forward, reverse, complement, reverse
complement) produce a phonological fingerprint detecting rhyme, alliteration, and
morphological similarity without parameters or training.
Universal Structural Index — The same iterator applied to Carroll, Keats, Debussy, and
Bach recovers structural recurrence — Debussy’s recapitulation, Bach’s cross-movement
motivic unity, Keats’ iambic metre — as arithmetic equivalence. Validated against 500
random null-model trials (Z > 15, p << 0.001).
Tensor Autocorrelation — J-stream autocorrelation at lags 1-12 discriminates musical from
literary works. Bach Sarabande dominant period lag 7 (strength 0.1202). Clair de Lune lag 3
(triplet grouping, 9/8 time). Literary works near-noise (0.02-0.04).
Status
Core construction: empirically verified, formally metric
Bijectivity at fixed depth: open — identified as foundational next step
arXiv submission: pending
GitHub: active development
Contact
@unitambo
unitambo.wordpress.com
