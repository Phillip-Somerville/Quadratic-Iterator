import { useState, useCallback } from “react”;

// ── CANONICAL QUADRATIC ITERATOR ──────────────────────────────────────
// Phillip Somerville (@unitambo)
// f(x) = (x² − F)² − F   F ∈ {1.094, 1.096}

const C_STATIC = 1.094;
const DELTA    = 0.002;
const PREC     = 14;
const EPS      = 5e-15;

function truncRound(x, dp = PREC) {
const r = Math.round(x * 10 ** dp) / 10 ** dp;
const f = 10 ** dp;
return r >= 0 ? Math.floor(r * f) / f : Math.ceil(r * f) / f;
}

function encodeNode(payload, seed = 0.087) {
const bits = [];
for (const ch of payload) {
const code = ch.charCodeAt(0) & 0xff;
for (let i = 7; i >= 0; i–) bits.push((code >> i) & 1);
}
bits.push(0, 0);
const F_seq = bits.map(b => C_STATIC + (b === 1 ? DELTA : 0));
let H = seed;
const B = [];
for (let n = 0; n < bits.length; n++) {
const H_new = truncRound((H * H - F_seq[n]) ** 2 - F_seq[n]);
const inner = H_new + F_seq[n];
let j = 0;
if (inner >= 0 && F_seq[n] >= Math.sqrt(inner)) {
const k = Math.round(Math.sqrt(F_seq[n] - Math.sqrt(inner)) * 10 ** PREC) / 10 ** PREC;
j = Math.max(-1, Math.min(1, Math.round((k - H) * 1e14)));
}
B.push(j);
H = H_new;
}
return { digest: H, B, depth: bits.length };
}

function fixedPoint(payload, seed = 0.087, maxLoops = 20) {
let D = seed;
for (let i = 0; i < maxLoops; i++) {
const { digest } = encodeNode(payload, D);
if (Math.abs(digest - D) < EPS) return digest;
D = digest;
}
return D;
}

function analyseStream(payload, seed = 0.087) {
const { digest, B, depth } = encodeNode(payload, seed);
const D = fixedPoint(payload);
const { digest: verify } = encodeNode(payload, D);
const selfCertifies = Math.abs(verify - D) < EPS;

const neg   = B.filter(j => j === -1).length;
const zero  = B.filter(j => j === 0).length;
const pos   = B.filter(j => j === 1).length;
const total = B.length;

// Sign-flip autocorrelation (lag 1)
const nonzero = B.filter(j => j !== 0);
let flips = 0;
for (let i = 1; i < nonzero.length; i++)
if (nonzero[i] * nonzero[i-1] < 0) flips++;
const flipRate = nonzero.length > 1 ? flips / (nonzero.length - 1) : 0;

// Overhead
const overhead = (total - zero) / total;

return { D, digest, selfCertifies, B, depth: total, neg, zero, pos, flipRate, overhead };
}

// ── COMPONENT ─────────────────────────────────────────────────────────
const MONO = “‘JetBrains Mono’,‘Fira Code’,‘Courier New’,monospace”;
const SERIF = “‘Palatino Linotype’,‘Book Antiqua’,Palatino,Georgia,serif”;

const EXAMPLES = [
“the rain in Spain falls mainly on the plain”,
“hamlet”,
“To be, or not to be”,
“of”,
“’Twas brillig, and the slithy toves”,
];

export default function App() {
const [input, setInput]   = useState(“the rain in Spain falls mainly on the plain”);
const [result, setResult] = useState(null);
const [seed,   setSeed]   = useState(“0.087”);

const run = useCallback(() => {
const s = parseFloat(seed);
if (isNaN(s)) return;
const r = analyseStream(input, s);
setResult(r);
}, [input, seed]);

const r = result;

// J-stream visualisation — 60 chars wide
const jViz = r ? r.B.slice(0, 60).map(j =>
j === -1 ? ‘▼’ : j === 1 ? ‘▲’ : ‘·’
).join(’’) : null;

return (
<div style={{
background: ‘#0a0b0e’,
minHeight: ‘100vh’,
padding: ‘40px 24px’,
fontFamily: MONO,
color: ‘#c8d0dc’,
}}>
<div style={{ maxWidth: 680, margin: ‘0 auto’ }}>

```
    {/* Header */}
    <div style={{ marginBottom: 40 }}>
      <div style={{ fontSize: 10, letterSpacing: 6, color: '#2a3040', marginBottom: 10 }}>
        QUADRATIC ITERATOR · CANONICAL DEMO
      </div>
      <h1 style={{
        fontFamily: SERIF,
        fontSize: 26,
        fontWeight: 400,
        color: '#e8edf5',
        lineHeight: 1.2,
        margin: '0 0 8px',
      }}>
        f(x) = (x² − F)² − F
      </h1>
      <div style={{ fontSize: 11, color: '#3a4555', lineHeight: 1.7 }}>
        F ∈ {'{1.094, 1.096}'}  ·  PREC = 14  ·  ε = 5×10⁻¹⁵
      </div>
    </div>

    {/* Input */}
    <div style={{ marginBottom: 24 }}>
      <div style={{ fontSize: 10, letterSpacing: 3, color: '#2a3545', marginBottom: 8 }}>
        PAYLOAD
      </div>
      <textarea
        value={input}
        onChange={e => setInput(e.target.value)}
        rows={3}
        style={{
          width: '100%',
          background: '#0d1018',
          border: '1px solid #1e2535',
          borderRadius: 6,
          padding: '12px 14px',
          fontFamily: MONO,
          fontSize: 13,
          color: '#c8d0dc',
          resize: 'vertical',
          boxSizing: 'border-box',
          outline: 'none',
          lineHeight: 1.6,
        }}
      />
    </div>

    {/* Seed + Run */}
    <div style={{ display: 'flex', gap: 12, marginBottom: 32, alignItems: 'center' }}>
      <div>
        <div style={{ fontSize: 10, letterSpacing: 3, color: '#2a3545', marginBottom: 6 }}>SEED</div>
        <input
          value={seed}
          onChange={e => setSeed(e.target.value)}
          style={{
            background: '#0d1018',
            border: '1px solid #1e2535',
            borderRadius: 5,
            padding: '8px 12px',
            fontFamily: MONO,
            fontSize: 12,
            color: '#c8d0dc',
            width: 120,
            outline: 'none',
          }}
        />
      </div>
      <button
        onClick={run}
        style={{
          marginTop: 22,
          background: '#0d1a2a',
          border: '1px solid #3a6090',
          borderRadius: 5,
          padding: '9px 24px',
          fontFamily: MONO,
          fontSize: 12,
          color: '#7ab0e0',
          cursor: 'pointer',
          letterSpacing: 2,
        }}
      >
        ENCODE
      </button>
      <div style={{ marginTop: 22, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {EXAMPLES.map(ex => (
          <button
            key={ex}
            onClick={() => { setInput(ex); }}
            style={{
              background: 'transparent',
              border: '1px solid #1e2535',
              borderRadius: 4,
              padding: '5px 10px',
              fontFamily: MONO,
              fontSize: 10,
              color: '#3a4555',
              cursor: 'pointer',
              letterSpacing: 0,
            }}
          >
            {ex.length > 20 ? ex.slice(0, 20) + '…' : ex}
          </button>
        ))}
      </div>
    </div>

    {/* Results */}
    {r && (
      <div>
        {/* Fixed point */}
        <div style={{
          background: '#0d1018',
          border: `1px solid ${r.selfCertifies ? '#1e4a2e' : '#4a1e1e'}`,
          borderRadius: 8,
          padding: '20px 24px',
          marginBottom: 16,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 12 }}>
            <div style={{ fontSize: 10, letterSpacing: 3, color: '#2a4040' }}>CANONICAL FIXED POINT D*</div>
            <div style={{
              fontSize: 10, letterSpacing: 2,
              color: r.selfCertifies ? '#4caf82' : '#ef9a9a',
              border: `1px solid ${r.selfCertifies ? '#1e4a2e' : '#4a1e1e'}`,
              borderRadius: 3, padding: '2px 8px',
            }}>
              {r.selfCertifies ? '✓ SELF-CERTIFIES' : '✗ DOES NOT CERTIFY'}
            </div>
          </div>
          <div style={{ fontFamily: MONO, fontSize: 18, color: '#7ab0e0', letterSpacing: 1 }}>
            {r.D.toFixed(14)}
          </div>
          <div style={{ marginTop: 8, fontSize: 11, color: '#2a3545' }}>
            Seed-independent · same payload always produces same D*
          </div>
        </div>

        {/* Stats grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 8, marginBottom: 16 }}>
          {[
            { label: 'DEPTH', val: r.depth, unit: 'steps' },
            { label: 'OVERHEAD', val: (r.overhead * 100).toFixed(1), unit: '%' },
            { label: 'FLIP RATE', val: (r.flipRate * 100).toFixed(1), unit: '%' },
            { label: 'PAYLOAD', val: input.length, unit: 'chars' },
          ].map(({ label, val, unit }) => (
            <div key={label} style={{
              background: '#0d1018', border: '1px solid #1e2535',
              borderRadius: 6, padding: '12px 14px',
            }}>
              <div style={{ fontSize: 9, letterSpacing: 3, color: '#2a3545', marginBottom: 6 }}>{label}</div>
              <div style={{ fontSize: 20, fontWeight: 700, color: '#c8d0dc' }}>{val}</div>
              <div style={{ fontSize: 10, color: '#2a3545', marginTop: 2 }}>{unit}</div>
            </div>
          ))}
        </div>

        {/* J-stream distribution */}
        <div style={{
          background: '#0d1018', border: '1px solid #1e2535',
          borderRadius: 8, padding: '16px 20px', marginBottom: 16,
        }}>
          <div style={{ fontSize: 10, letterSpacing: 3, color: '#2a3545', marginBottom: 14 }}>
            J-STREAM DISTRIBUTION  ·  ternary control {'{−1, 0, +1}'}
          </div>
          <div style={{ display: 'flex', gap: 24, marginBottom: 14 }}>
            {[
              { label: '▼  −1', val: r.neg,  color: '#ef9a9a', pct: (r.neg/r.depth*100).toFixed(1) },
              { label: '·   0', val: r.zero, color: '#4a5565', pct: (r.zero/r.depth*100).toFixed(1) },
              { label: '▲  +1', val: r.pos,  color: '#81c784', pct: (r.pos/r.depth*100).toFixed(1) },
            ].map(({ label, val, color, pct }) => (
              <div key={label}>
                <div style={{ fontSize: 11, color, marginBottom: 4, letterSpacing: 1 }}>{label}</div>
                <div style={{ fontSize: 18, fontWeight: 700, color }}>{val}</div>
                <div style={{ fontSize: 10, color: '#2a3545' }}>{pct}%</div>
              </div>
            ))}
          </div>
          {/* Bar */}
          <div style={{ display: 'flex', height: 6, borderRadius: 3, overflow: 'hidden', marginBottom: 14 }}>
            <div style={{ flex: r.neg,  background: '#ef9a9a40' }}/>
            <div style={{ flex: r.zero, background: '#1e2535' }}/>
            <div style={{ flex: r.pos,  background: '#81c78440' }}/>
          </div>
          {/* Visualisation */}
          <div style={{ fontSize: 10, color: '#2a3545', marginBottom: 6 }}>
            FIRST 60 STEPS  ·  ▲=+1  ·=0  ▼=−1
          </div>
          <div style={{ fontFamily: MONO, fontSize: 12, color: '#5a7090', letterSpacing: 2, lineHeight: 1.8 }}>
            {jViz}
          </div>
        </div>

        {/* Raw stream */}
        <div style={{
          background: '#0d1018', border: '1px solid #1e2535',
          borderRadius: 8, padding: '14px 18px',
        }}>
          <div style={{ fontSize: 10, letterSpacing: 3, color: '#2a3545', marginBottom: 10 }}>
            CONTROL STREAM B[]  ·  first 80 values
          </div>
          <div style={{
            fontFamily: MONO, fontSize: 11, color: '#3a5070',
            lineHeight: 2, wordBreak: 'break-all', letterSpacing: 1,
          }}>
            {r.B.slice(0, 80).map((j, i) => (
              <span key={i} style={{
                color: j === -1 ? '#ef9a9a80' : j === 1 ? '#81c78480' : '#2a3545',
                marginRight: 4,
              }}>
                {j === -1 ? '−1' : j === 0 ? ' 0' : '+1'}
              </span>
            ))}
            {r.B.length > 80 && <span style={{ color: '#2a3545' }}>…+{r.B.length - 80}</span>}
          </div>
        </div>

        <div style={{ marginTop: 20, fontSize: 10, color: '#1e2535', lineHeight: 1.8 }}>
          unitambo.wordpress.com · @unitambo · Phillip Somerville 2026
        </div>
      </div>
    )}

  </div>
</div>
```

);
}
