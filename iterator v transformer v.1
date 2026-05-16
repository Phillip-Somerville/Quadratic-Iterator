// ════════════════════════════════════════════════════════════════════
// ITERATOR CONTROL-TARGET STREAM vs RAW BITS — A vs C COMPARISON
// Phillip Somerville (@unitambo) · 2026
//
// CONSTRUCTION
//   Bijective quadratic iterator: f(x) = (x² − F)² − F
//   F ∈ {1.094, 1.096}  PREC=14  ε=5×10⁻¹⁵
//
//   Per word, the iterator produces a control-target stream:
//     Target channel:  raw bit expansion of ASCII characters {0,1}
//     Control channel: ternary J-events from inverse interrogation {−1,0,+1}
//     Interleaved:     [b₀,j₀][b₁,j₁]···[bₙ,jₙ]
//   Word boundaries marked by iterator state digests: [H_seed, H_terminal]
//   Chain seed flows word-to-word — inter-word context carried implicitly
//
// EXPERIMENT
//   Condition A: raw bits | J=0 | [H_seed,H_terminal] dummy boundary
//   Condition C: iterator stream | J∈{−1,0,+1} | [H_seed,H_terminal] boundary
//   Both: identical QKV architecture — CTX=32 QKV(8/8) flat(256d) 128×2 MLP
//   Corpus: Alice in Wonderland Ch.I — 871 word pairs — vocab 309
//   Seeded init (LCG seed=42) — identical weight geometry both conditions
//   lr=2e-4  batch=16  100 epochs
//
// RESULTS (3 runs, consistent)
//   C final loss: ~0.22   signal 5.51 nats below random (log 309 = 5.73)
//   A final loss: ~1.95   signal 3.78 nats below random
//   C diverges from A at epoch 3
//   C surpasses A’s 100-epoch ceiling at ~epoch 25 (4× faster convergence)
//   Attention entropy drop: C=0.49 vs A=0.35 — deeper structural exploitation
//
// NOTE
//   Prior versions used tanh(j×10⁻⁶) — attenuated J-signal by ×10⁶
//   This version uses max(−1, min(1, j)) — correct ternary encoding
//   The WOW signal appeared only after this fix
// ════════════════════════════════════════════════════════════════════

import { useState, useRef, useCallback } from “react”;

const C_STATIC = 1.094, DELTA = 0.002, PREC = 14;

function truncRound(x, dp = PREC) {
const r = Math.round(x * 10 ** dp) / 10 ** dp;
const f = 10 ** dp;
return r >= 0 ? Math.floor(r * f) / f : Math.ceil(r * f) / f;
}

function iterateWord(word, seed) {
const bits = [];
for (const ch of word) {
const code = ch.charCodeAt(0) & 0xff;
for (let i = 7; i >= 0; i–) bits.push((code >> i) & 1);
}
bits.push(0, 0);
const F_seq = bits.map(b => C_STATIC + (b === 1 ? DELTA : 0));
let H = seed;
const stream = new Float32Array(bits.length * 2);
for (let n = 0; n < bits.length; n++) {
const H_new = truncRound((H * H - F_seq[n]) ** 2 - F_seq[n]);
const inner = H_new + F_seq[n];
let j = 0;
if (inner >= 0 && F_seq[n] >= Math.sqrt(inner)) {
const k = Math.round(Math.sqrt(F_seq[n] - Math.sqrt(inner)) * 10 ** PREC) / 10 ** PREC;
j = Math.round((k - H) * 1e14);
}
stream[2 * n]     = bits[n];
stream[2 * n + 1] = Math.max(-1, Math.min(1, j));
H = H_new;
}
// No D* convergence loop — terminal H is the digest
// Boundary sentinel: [H_seed, H_terminal]
// encodes both ends of the word’s traversal path
return { stream, H_seed: seed, H_terminal: H, nextSeed: H };
}

function rawWordStream(word, seed) {
const bits = [];
for (const ch of word) {
const code = ch.charCodeAt(0) & 0xff;
for (let i = 7; i >= 0; i–) bits.push((code >> i) & 1);
}
bits.push(0, 0);
const stream = new Float32Array(bits.length * 2);
let H = seed;
for (let n = 0; n < bits.length; n++) {
stream[2 * n]     = bits[n];
stream[2 * n + 1] = 0.0;
// A also advances a dummy seed for positional parity
H = truncRound((H * H - C_STATIC) ** 2 - C_STATIC);
}
// A boundary: [H_seed, H_terminal] from raw traversal
// J-channel still zero throughout — only boundary uses terminal
return { stream, H_seed: seed, H_terminal: H, nextSeed: H };
}

// ── SEEDED RNG ────────────────────────────────────────────────────────
let _rngState = 42;
function resetRng(seed = 42) { _rngState = seed >>> 0; }
function seededRand() {
_rngState = (Math.imul(_rngState, 1664525) + 1013904223) >>> 0;
return _rngState / 0x100000000;
}

// ── MATH ──────────────────────────────────────────────────────────────
const heInit = (r, c) =>
Array.from({ length: r * c }, () => (seededRand() * 2 - 1) * Math.sqrt(2 / c));
const zeros  = n => new Array(n).fill(0);
const relu   = x => Math.max(0, x);
const reluG  = x => x > 0 ? 1 : 0;

function softmax(arr) {
const max = Math.max(…arr);
const exp = arr.map(x => Math.exp(x - max));
const s   = exp.reduce((a, b) => a + b, 0);
return exp.map(x => x / s);
}

function attentionEntropy(w) {
return -w.reduce((s, v) => s + (v > 1e-10 ? v * Math.log(v) : 0), 0);
}

function matmul(W, x, rows, cols) {
const out = new Array(rows).fill(0);
for (let i = 0; i < rows; i++)
for (let j = 0; j < cols; j++) out[i] += W[i * cols + j] * x[j];
return out;
}

function adamStep(p, g, m, v, t, lr, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
const bc1 = 1 - b1 ** t, bc2 = 1 - b2 ** t;
for (let i = 0; i < p.length; i++) {
if (g[i] === 0) continue;
m[i] = b1 * m[i] + (1 - b1) * g[i];
v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
p[i] -= lr * (m[i] / bc1) / (Math.sqrt(v[i] / bc2) + eps);
}
}

const D_IN = 2, D_K = 8, D_V = 8, CTX = 32;
const IN_DIM = CTX * D_V;

function buildAttn(lr) {
return {
lr, t: 0,
WQ: heInit(D_K, D_IN), mWQ: zeros(D_K*D_IN), vWQ: zeros(D_K*D_IN),
WK: heInit(D_K, D_IN), mWK: zeros(D_K*D_IN), vWK: zeros(D_K*D_IN),
WV: heInit(D_V, D_IN), mWV: zeros(D_V*D_IN), vWV: zeros(D_V*D_IN),
};
}

function attnForward(a, seq) {
const N = seq.length;
const Q = seq.map(x => matmul(a.WQ, x, D_K, D_IN));
const K = seq.map(x => matmul(a.WK, x, D_K, D_IN));
const V = seq.map(x => matmul(a.WV, x, D_V, D_IN));
const scores  = Q.map(q => K.map(k => q.reduce((s,v,i) => s+v*k[i], 0) / Math.sqrt(D_K)));
const weights = scores.map(row => softmax(row));
const entropies = weights.map(attentionEntropy);
const attended = weights.map(wRow =>
wRow.reduce((acc, w, j) => acc.map((v, d) => v + w * V[j][d]), new Array(D_V).fill(0))
);
return { Q, K, V, weights, entropies, attended };
}

function attnBackward(a, seq, fwd, dAttended) {
const N = seq.length;
const { Q, K, V, weights } = fwd;
const dWQ = zeros(D_K*D_IN), dWK = zeros(D_K*D_IN), dWV = zeros(D_V*D_IN);
for (let i = 0; i < N; i++) {
for (let j = 0; j < N; j++)
for (let d = 0; d < D_V; d++) {
const grad = weights[i][j] * dAttended[i][d];
for (let k = 0; k < D_IN; k++) dWV[d*D_IN+k] += grad * seq[j][k];
}
const dW = new Array(N).fill(0);
for (let j = 0; j < N; j++)
for (let d = 0; d < D_V; d++) dW[j] += dAttended[i][d] * V[j][d];
const dScores = weights[i].map((w, j) => {
let s = 0;
for (let k = 0; k < N; k++) s += weights[i][k] * dW[k];
return w * (dW[j] - s) / Math.sqrt(D_K);
});
const dQi = new Array(D_K).fill(0);
for (let j = 0; j < N; j++)
for (let d = 0; d < D_K; d++) dQi[d] += dScores[j] * K[j][d];
for (let d = 0; d < D_K; d++)
for (let k = 0; k < D_IN; k++) dWQ[d*D_IN+k] += dQi[d] * seq[i][k];
for (let j = 0; j < N; j++)
for (let d = 0; d < D_K; d++)
for (let k = 0; k < D_IN; k++) dWK[d*D_IN+k] += dScores[j] * Q[i][d] * seq[j][k];
}
a.t++;
adamStep(a.WQ, dWQ, a.mWQ, a.vWQ, a.t, a.lr);
adamStep(a.WK, dWK, a.mWK, a.vWK, a.t, a.lr);
adamStep(a.WV, dWV, a.mWV, a.vWV, a.t, a.lr);
}

function buildMLP(inDim, hDim, outDim, lr) {
return {
in: inDim, h: hDim, out: outDim, lr, t: 0,
W1: heInit(hDim,inDim),  b1: zeros(hDim),
W2: heInit(hDim,hDim),   b2: zeros(hDim),
W3: heInit(outDim,hDim), b3: zeros(outDim),
mW1: zeros(hDim*inDim), vW1: zeros(hDim*inDim),
mb1: zeros(hDim),       vb1: zeros(hDim),
mW2: zeros(hDim*hDim),  vW2: zeros(hDim*hDim),
mb2: zeros(hDim),       vb2: zeros(hDim),
mW3: zeros(outDim*hDim),vW3: zeros(outDim*hDim),
mb3: zeros(outDim),     vb3: zeros(outDim),
};
}

function mlpForwardBack(m, x, target) {
const z1 = matmul(m.W1,x,m.h,m.in).map((v,i)=>v+m.b1[i]);
const a1 = z1.map(relu);
const z2 = matmul(m.W2,a1,m.h,m.h).map((v,i)=>v+m.b2[i]);
const a2 = z2.map(relu);
const z3 = matmul(m.W3,a2,m.out,m.h).map((v,i)=>v+m.b3[i]);
const probs = softmax(z3);
const loss  = -Math.log(probs[target]+1e-10);
const dz3 = probs.map((p,i)=>p-(i===target?1:0));
const dW3 = zeros(m.out*m.h);
for (let i=0;i<m.out;i++) for (let j=0;j<m.h;j++) dW3[i*m.h+j]=dz3[i]*a2[j];
const da2 = zeros(m.h);
for (let j=0;j<m.h;j++) for (let i=0;i<m.out;i++) da2[j]+=m.W3[i*m.h+j]*dz3[i];
const dz2 = da2.map((v,i)=>v*reluG(z2[i]));
const dW2 = zeros(m.h*m.h);
for (let i=0;i<m.h;i++) for (let j=0;j<m.h;j++) dW2[i*m.h+j]=dz2[i]*a1[j];
const da1 = zeros(m.h);
for (let j=0;j<m.h;j++) for (let i=0;i<m.h;i++) da1[j]+=m.W2[i*m.h+j]*dz2[i];
const dz1 = da1.map((v,i)=>v*reluG(z1[i]));
const dW1 = zeros(m.h*m.in);
for (let i=0;i<m.h;i++) for (let j=0;j<m.in;j++) dW1[i*m.in+j]=dz1[i]*x[j];
const dx = zeros(m.in);
for (let j=0;j<m.in;j++) for (let i=0;i<m.h;i++) dx[j]+=m.W1[i*m.in+j]*dz1[i];
m.t++;
adamStep(m.W1,dW1,m.mW1,m.vW1,m.t,m.lr);
adamStep(m.b1,dz1,m.mb1,m.vb1,m.t,m.lr);
adamStep(m.W2,dW2,m.mW2,m.vW2,m.t,m.lr);
adamStep(m.b2,dz2,m.mb2,m.vb2,m.t,m.lr);
adamStep(m.W3,dW3,m.mW3,m.vW3,m.t,m.lr);
adamStep(m.b3,dz3,m.mb3,m.vb3,m.t,m.lr);
return { loss, dx };
}

const CORPUS = `Alice was beginning to get very tired of sitting by her sister on the bank and of having nothing to do once or twice she had peeped into the book her sister was reading but it had no pictures or conversations in it and what is the use of a book thought Alice without pictures or conversations so she was considering in her own mind as well as she could for the hot day made her feel very sleepy and stupid whether the pleasure of making a daisy chain would be worth the trouble of getting up and picking the daisies when suddenly a White Rabbit with pink eyes ran close by her there was nothing so very remarkable in that nor did Alice think it so very much out of the way to hear the Rabbit say to itself oh dear oh dear I shall be late but when the Rabbit actually took a watch out of its waistcoat pocket and looked at it and then hurried on Alice started to her feet for it flashed across her mind that she had never before seen a rabbit with either a waistcoat pocket or a watch to take out of it and burning with curiosity she ran across the field after it and was just in time to see it pop down a large rabbit hole under the hedge in another moment down went Alice after it the rabbit hole went straight on like a tunnel for some way and then dipped suddenly down so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well either the well was very deep or she fell very slowly for she had plenty of time as she went down to look about her and to wonder what was going to happen next first she tried to look down and make out what she was coming to but it was too dark to see anything then she looked at the sides of the well and noticed that they were filled with cupboards and book shelves here and there she saw maps and pictures hung upon pegs she took down a jar from one of the shelves as she passed it was labelled orange marmalade but to her great disappointment it was empty she did not like to drop the jar for fear of killing somebody underneath so managed to put it into one of the cupboards as she fell past it down down down would the fall never come to an end there was nothing else to do so Alice soon began talking again Dinah will miss me very much tonight I should think Dinah was the cat I hope they will remember her saucer of milk at teatime Dinah my dear I wish you were down here with me there are no mice in the air I am afraid but you might catch a bat and that is very like a mouse you know but do cats eat bats I wonder and here Alice began to get rather sleepy and went on saying to herself in a dreamy sort of way do cats eat bats do cats eat bats and sometimes do bats eat cats for you see as she could not answer either question it did not much matter which way she put it she felt that she was dozing off and had just begun to dream that she was walking hand in hand with Dinah when suddenly thump thump down she came upon a heap of sticks and dry leaves and the fall was over Alice was not a bit hurt and she jumped up on to her feet in a moment she looked up but it was all dark overhead before her was another long passage and the White Rabbit was still in sight hurrying down it there was not a moment to be lost away went Alice like the wind and was just in time to hear it say as it turned a corner oh my ears and whiskers how late it is getting she was close behind it when she turned the corner but the Rabbit was no longer to be seen she found herself in a long low hall which was lit up by a row of lamps hanging from the roof there were doors all round the hall but they were all locked and when Alice had been all the way down one side and up the other trying every door she walked sadly down the middle wondering how she was ever to get out again suddenly she came upon a little table all made of solid glass there was nothing on it but a tiny golden key and Alice thought it might belong to one of the doors of the hall but the locks were too large or the key was too small and it would not open any of them however on the second time round she came upon a low curtain she had not noticed before and behind it was a little door about fifteen inches high she tried the little golden key in the lock and to her great delight it fitted`;

function buildStreams(corpus) {
const rawWords = corpus.toLowerCase().split(/\s+/).filter(Boolean);
const vocab    = […new Set(rawWords)].sort();
const wordToIdx = Object.fromEntries(vocab.map((w,i)=>[w,i]));
const V = vocab.length;
const seqC = [], seqA = [], pairs = [];
let seedC = 0.087, seedA = 0.087;

for (let wi = 0; wi < rawWords.length; wi++) {
const word = rawWords[wi];

```
// C: full iterator stream + [H_seed, H_terminal] boundary
const { stream: cStream, H_seed: cSeed, H_terminal: cTerm, nextSeed: cNext } =
  iterateWord(word, seedC);
seedC = cNext;
for (let n = 0; n < cStream.length; n += 2)
  seqC.push([cStream[n], cStream[n+1]]);
const bpC = seqC.length;
seqC.push([cSeed, cTerm]); // [H_seed, H_terminal] — gradient across word

// A: raw bits + J=0 + [H_seed, H_terminal] from dummy traversal
const { stream: aStream, H_seed: aSeed, H_terminal: aTerm, nextSeed: aNext } =
  rawWordStream(word, seedA);
seedA = aNext;
for (let n = 0; n < aStream.length; n += 2)
  seqA.push([aStream[n], aStream[n+1]]);
const bpA = seqA.length;
seqA.push([aSeed, aTerm]); // same structure — boundary encodes traversal range

if (wi < rawWords.length - 1)
  pairs.push({ posC: bpC, posA: bpA, target: wordToIdx[rawWords[wi+1]] });
```

}
return { seqC, seqA, pairs, V, rawWords };
}

const EPOCHS = 100;
const RNG_SEED = 42;

export default function App() {
const [status,  setStatus]  = useState(‘idle’);
const [lossA,   setLA]      = useState([]);
const [lossC,   setLC]      = useState([]);
const [entA,    setEA]      = useState([]);
const [entC,    setEC]      = useState([]);
const [finals,  setFinals]  = useState(null);
const [epoch,   setEpoch]   = useState(0);
const [log,     setLog]     = useState([]);
const [copied,  setCopied]  = useState(false);
const [info,    setInfo]    = useState(null);
const running = useRef(false);
const fullLog = useRef([]);

const addLog = msg => { fullLog.current.push(msg); setLog(p=>[…p.slice(-100),msg]); };
const copyResults = () =>
navigator.clipboard.writeText(fullLog.current.join(’\n’))
.then(()=>{ setCopied(true); setTimeout(()=>setCopied(false),2000); });

const run = useCallback(async () => {
if (running.current) return;
running.current = true;
setStatus(‘running’);
setLA([]); setLC([]); setEA([]); setEC([]);
setFinals(null); setEpoch(0);
setLog([]); fullLog.current = [];

```
const { seqC, seqA, pairs, V, rawWords } = buildStreams(CORPUS);
setInfo({ words: rawWords.length, vocab: V, pairs: pairs.length });

const HIDDEN = 128, LR = 2e-4, BATCH = 16;
const logV = Math.log(V);

addLog(`A vs C v2 — SEEDED(${RNG_SEED}) — ${EPOCHS} EPOCHS`);
addLog(`A: raw bits | J=0 | boundary=[H_seed, H_terminal] dummy traversal`);
addLog(`C: iterator stream | J={-1,0,1} ternary | boundary=[H_seed, H_terminal]`);
addLog(`No D* convergence loop — terminal H is the digest, EoW`);
addLog(`Both: CTX=${CTX} QKV(${D_K}/${D_V}) flat(${IN_DIM}d) ${HIDDEN}x2 vocab=${V}`);
addLog(`lr=${LR} batch=${BATCH} random=log(${V})=${logV.toFixed(4)}`);
addLog(`─────────────────────────────────────────────`);

resetRng(RNG_SEED);
const attnA = buildAttn(LR);
const mlpA  = buildMLP(IN_DIM, HIDDEN, V, LR);

resetRng(RNG_SEED);
const attnC = buildAttn(LR);
const mlpC  = buildMLP(IN_DIM, HIDDEN, V, LR);

const laArr=[], lcArr=[], eaArr=[], ecArr=[];

for (let ep = 0; ep < EPOCHS; ep++) {
  if (!running.current) break;
  const shuffled = [...pairs].sort(()=>Math.random()-0.5);
  let sumLA=0, sumLC=0, sumEA=0, sumEC=0, cnt=0;

  for (let b=0; b<shuffled.length; b+=BATCH) {
    if (!running.current) break;
    const batch = shuffled.slice(b, Math.min(b+BATCH, shuffled.length));

    for (const { posC, posA, target } of batch) {
      // A
      const ctxA = seqA.slice(Math.max(0,posA-CTX+1), posA+1);
      while (ctxA.length < CTX) ctxA.unshift([0,0]);
      const fwdA = attnForward(attnA, ctxA);
      sumEA += fwdA.entropies.reduce((a,b)=>a+b,0)/fwdA.entropies.length;
      const { loss: lA, dx: dxA } = mlpForwardBack(mlpA, fwdA.attended.flat(), target);
      sumLA += isFinite(lA) ? lA : 0;
      const dAttnA = [];
      for (let k=0;k<CTX;k++) dAttnA.push(dxA.slice(k*D_V,(k+1)*D_V));
      attnBackward(attnA, ctxA, fwdA, dAttnA);

      // C
      const ctxC = seqC.slice(Math.max(0,posC-CTX+1), posC+1);
      while (ctxC.length < CTX) ctxC.unshift([0,0]);
      const fwdC = attnForward(attnC, ctxC);
      sumEC += fwdC.entropies.reduce((a,b)=>a+b,0)/fwdC.entropies.length;
      const { loss: lC, dx: dxC } = mlpForwardBack(mlpC, fwdC.attended.flat(), target);
      sumLC += isFinite(lC) ? lC : 0;
      const dAttnC = [];
      for (let k=0;k<CTX;k++) dAttnC.push(dxC.slice(k*D_V,(k+1)*D_V));
      attnBackward(attnC, ctxC, fwdC, dAttnC);

      cnt++;
    }
    if (b % (BATCH*8) === 0) await new Promise(r=>setTimeout(r,0));
  }

  const avgLA=sumLA/cnt, avgLC=sumLC/cnt;
  const avgEA=sumEA/cnt, avgEC=sumEC/cnt;
  laArr.push(avgLA); lcArr.push(avgLC);
  eaArr.push(avgEA); ecArr.push(avgEC);
  setLA([...laArr]); setLC([...lcArr]);
  setEA([...eaArr]); setEC([...ecArr]);
  setEpoch(ep+1);
  setFinals({ la:avgLA, lc:avgLC, ea:avgEA, ec:avgEC,
              sigA:logV-avgLA, sigC:logV-avgLC, delta:avgLA-avgLC });
  const m = (ep+1)%10===0;
  addLog(`Ep${String(ep+1).padStart(2)}  A=${avgLA.toFixed(4)} C=${avgLC.toFixed(4)}  eA=${avgEA.toFixed(4)} eC=${avgEC.toFixed(4)}  Δ=${(avgLA-avgLC).toFixed(4)}${m?' ◆':''}`);
}

const fLA=laArr.at(-1), fLC=lcArr.at(-1);
addLog(`─────────────────────────────────────────────`);
addLog(`FINAL  A=${fLA.toFixed(4)}  C=${fLC.toFixed(4)}  Δ=${(fLA-fLC).toFixed(4)}`);
addLog(`Signal A=${(logV-fLA).toFixed(4)}  C=${(logV-fLC).toFixed(4)}`);
addLog(`Ent drop A=${(eaArr[0]-eaArr.at(-1)).toFixed(4)}  C=${(ecArr[0]-ecArr.at(-1)).toFixed(4)}`);
addLog(`Winner: ${fLC<fLA?'C (iterator)':'A (raw ASCII)'}`);
running.current=false; setStatus('done');
```

}, []);

const renderCurves = () => {
if (!lossA.length) return null;
const W=540, H=210, PAD=48;
const all=[…lossA,…lossC];
const minL=Math.min(…all)*0.97, maxL=Math.max(…all)*1.03;
const allE=[…entA,…entC];
const minE=Math.min(…allE)*0.97, maxE=Math.max(…allE)*1.03;
const n=lossA.length;
const xS  = i => PAD+(i/Math.max(n-1,1))*(W-PAD*2);
const ySL = v => PAD+(1-(v-minL)/(maxL-minL))*(H-PAD*2);
const ySE = v => PAD+(1-(v-minE)/(maxE-minE))*(H-PAD*2);
const path = (data, ySc, color, dash) => {
if (!data.length) return null;
const d = data.map((v,i)=>`${i===0?'M':'L'}${xS(i).toFixed(1)},${ySc(v).toFixed(1)}`).join(’ ’);
return <path d={d} fill="none" stroke={color} strokeWidth="2.5" strokeDasharray={dash} strokeLinecap="round"/>;
};
const ticks = [0,0.25,0.5,0.75,1].map(t=>{
const v=minL+(maxL-minL)*t, y=ySL(v);
return <g key={t}><line x1={PAD-3} y1={y} x2={W-PAD} y2={y} stroke="#1a1c22"/><text x={PAD-6} y={y+4} textAnchor="end" fontSize="9" fill="#4a5060">{v.toFixed(2)}</text></g>;
});
const eM=[10,20,30,40,50,60,70,80,90].map(e=>e<n?<line key={e} x1={xS(e-1)} y1={PAD} x2={xS(e-1)} y2={H-PAD} stroke="#1a2030" strokeDasharray="2,4"/>:null);
return (
<svg width={W} height={H} style={{display:‘block’,maxWidth:‘100%’}}>
<rect width={W} height={H} fill="#080a0d" rx="8"/>
{ticks}{eM}
<line x1={PAD} y1={PAD} x2={PAD} y2={H-PAD} stroke="#1a1c22"/>
<line x1={PAD} y1={H-PAD} x2={W-PAD} y2={H-PAD} stroke="#1a1c22"/>
{path(lossA, ySL, ‘#64b5f6’)}
{path(lossC, ySL, ‘#ff8a65’)}
{path(entA,  ySE, ‘#64b5f6’, ‘3,3’)}
{path(entC,  ySE, ‘#ff8a65’, ‘3,3’)}
{lossA.length>0&&<circle cx={xS(lossA.length-1)} cy={ySL(lossA.at(-1))} r="4" fill="#64b5f6"/>}
{lossC.length>0&&<circle cx={xS(lossC.length-1)} cy={ySL(lossC.at(-1))} r="4" fill="#ff8a65"/>}
</svg>
);
};

const winner = finals ? finals.lc < finals.la ? ‘C’ : finals.la < finals.lc ? ‘A’ : ‘tie’ : null;

return (
<div style={{background:’#06080b’,color:’#b8c4d4’,fontFamily:”‘JetBrains Mono’,‘Fira Code’,monospace”,fontSize:12,minHeight:‘100vh’,padding:28}}>
<div style={{maxWidth:620,margin:‘0 auto’}}>

```
    <div style={{marginBottom:20}}>
      <div style={{fontSize:9,color:'#1e2530',letterSpacing:4,marginBottom:6}}>SEEDED v3 TERNARY · [H_seed, H_terminal] BOUNDARY · 100 EPOCHS</div>
      <h1 style={{fontFamily:'Georgia,serif',fontSize:18,fontWeight:300,color:'#dde4f0',lineHeight:1.3,marginBottom:6}}>
        Word Traversal Gradient as Boundary
      </h1>
      <div style={{fontSize:11,color:'#3a4455',lineHeight:1.7}}>
        Boundary sentinel: [H_seed, H_terminal] — both ends of word traversal<br/>
        No D* convergence loop — terminal H is the digest, EoW<br/>
        J-channel: C carries J∈{-1,0,1} · A carries zero
      </div>
      {info&&<div style={{marginTop:6,fontSize:11,color:'#2a3545'}}>{info.words} words · vocab {info.vocab} · {info.pairs} pairs</div>}
    </div>

    <div style={{background:'#0d1017',border:'1px solid #1a2030',borderRadius:8,padding:'10px 14px',marginBottom:14,fontSize:11,color:'#2a3545',lineHeight:1.8}}>
      <div>stream: <span style={{color:'#ff8a65'}}>[bit, j]···[bit, j]</span> boundary: <span style={{color:'#ff8a65'}}>[H_seed, H_term]</span> → predict next word</div>
      <div>A same structure — J=0 throughout · boundary from dummy traversal</div>
    </div>

    <div style={{display:'flex',gap:16,marginBottom:10,fontSize:11}}>
      <span style={{color:'#64b5f6'}}>── A loss  ╌╌ A entropy</span>
      <span style={{color:'#ff8a65'}}>── C loss  ╌╌ C entropy</span>
      {epoch>0&&<span style={{marginLeft:'auto',color:'#3a4a60'}}>ep {epoch}/{EPOCHS}</span>}
    </div>

    <div style={{marginBottom:14}}>
      {renderCurves()||<div style={{background:'#080a0d',border:'1px dashed #1a2030',borderRadius:8,height:210,display:'flex',alignItems:'center',justifyContent:'center',color:'#1a2030',fontSize:11}}>curves appear here</div>}
    </div>

    {finals&&(
      <div style={{display:'grid',gridTemplateColumns:'repeat(5,1fr)',gap:8,marginBottom:14}}>
        {[
          {label:'A loss',  val:finals.la.toFixed(4), color:'#64b5f6'},
          {label:'C loss',  val:finals.lc.toFixed(4), color:'#ff8a65'},
          {label:'Δ loss',  val:finals.delta.toFixed(4), color:finals.delta>0?'#81c784':'#ef9a9a'},
          {label:'A signal',val:finals.sigA.toFixed(4), color:'#64b5f6'},
          {label:'C signal',val:finals.sigC.toFixed(4), color:'#ff8a65'},
        ].map(({label,val,color})=>(
          <div key={label} style={{background:'#0d1017',border:'1px solid #1a2030',borderRadius:8,padding:'10px 12px'}}>
            <div style={{fontSize:10,color:'#2a3a50',marginBottom:5}}>{label}</div>
            <div style={{color,fontSize:16,fontWeight:700}}>{val}</div>
          </div>
        ))}
      </div>
    )}

    {winner&&(
      <div style={{background:'#0d1017',border:`1px solid ${winner==='C'?'#ff8a65':'#64b5f6'}`,borderRadius:8,padding:'10px 16px',marginBottom:14,fontSize:12,color:winner==='C'?'#ff8a65':'#64b5f6',fontWeight:600}}>
        {winner==='C'?'▲ C wins — J-stream + traversal gradient carry signal':'▲ A wins — raw bits + traversal range sufficient'}
      </div>
    )}

    <div style={{display:'flex',gap:10,marginBottom:14,flexWrap:'wrap'}}>
      <button onClick={run} disabled={status==='running'} style={{background:status==='running'?'#0d1017':'#0d1a2a',border:`1px solid ${status==='running'?'#1a2030':'#64b5f6'}`,color:status==='running'?'#2a3a50':'#64b5f6',fontFamily:'inherit',fontSize:12,padding:'9px 20px',borderRadius:6,cursor:status==='running'?'default':'pointer',letterSpacing:1}}>
        {status==='idle'?'RUN':status==='running'?'RUNNING…':'RUN AGAIN'}
      </button>
      {status==='running'&&<button onClick={()=>{running.current=false;}} style={{background:'transparent',border:'1px solid #1a2030',color:'#3a4455',fontFamily:'inherit',fontSize:12,padding:'9px 16px',borderRadius:6,cursor:'pointer'}}>STOP</button>}
      {log.length>0&&<button onClick={copyResults} style={{background:copied?'#0d1a14':'transparent',border:`1px solid ${copied?'#4caf82':'#1a2030'}`,color:copied?'#4caf82':'#3a4455',fontFamily:'inherit',fontSize:12,padding:'9px 16px',borderRadius:6,cursor:'pointer',marginLeft:'auto'}}>{copied?'✓ COPIED':'COPY'}</button>}
    </div>

    {log.length>0&&(
      <div style={{background:'#080a0d',border:'1px solid #1a2030',borderRadius:8,padding:'12px 14px',maxHeight:380,overflowY:'auto',fontSize:11,lineHeight:1.85}}>
        {log.map((l,i)=>{
          const isFinal=l.startsWith('FINAL')||l.startsWith('Signal')||l.startsWith('Winner')||l.startsWith('Ent');
          const milestone=l.includes('◆');
          return <div key={i} style={{color:isFinal?'#dde4f0':l.startsWith('─')?'#1a2030':milestone?'#8aabcc':'#2a3a50',fontWeight:isFinal||milestone?600:400}}>{l}</div>;
        })}
      </div>
    )}

    <div style={{marginTop:12,fontSize:10,color:'#1a2030',lineHeight:1.6}}>
      v3 · ternary J · [H_seed, H_terminal] boundary · no D* loop · EoW terminal is digest
    </div>
  </div>
</div>
```

);
}
