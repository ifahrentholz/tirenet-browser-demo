// TireNet-Demo: Live-Reifenerkennung im Browser, komplett auf dem Gerät.
// Kamera → quadratischer Ausschnitt um den Kreis → TireNet (ONNX, ~4 MB) → Kreis grün, wenn der Reifen passt.
import * as ort from './vendor/ort.wasm.min.mjs';

const MODEL = { file: './models/tirenet.onnx', size: 224 };
const CIRCLE_R = 0.45;          // Kreisradius relativ zur Bühnenbreite
const CENTER_TOL = 0.20;        // max. Abstand Box-Mitte ↔ Kreis-Mitte, relativ zum Radius
const SIZE_MIN = 0.70, SIZE_MAX = 1.15; // Reifengröße relativ zum Kreisdurchmesser
const ROI_FACTOR = 1.6;         // das Modell sieht einen Ausschnitt von 1,6 × Kreisdurchmesser
const HYSTERESIS = 3;           // so viele Frames in Folge, bis der Zustand wechselt (gegen Flackern)

const $ = (id) => document.getElementById(id);
const els = {
  stage: $('stage'), video: $('video'), testimg: $('testimg'), overlay: $('overlay'),
  ring: $('ring'), holeCircle: $('holeCircle'), badge: $('badge'), dbg: $('dbg'), hint: $('hint'),
  start: $('start'), btnStart: $('btnStart'), fileInput: $('fileInput'), shutter: $('shutter'),
  chkBoxes: $('chkBoxes'), rConf: $('rConf'), examples: $('examples'),
  err: $('err'), shot: $('shot'), shotImg: $('shotImg'), shotInfo: $('shotInfo'),
};
const out = (id, v) => { $(id).value = v; $(id).textContent = v; };

ort.env.wasm.wasmPaths = new URL('./vendor/', location.href).href;
// Mehrere Threads nur mit COOP/COEP-Headern (lokaler server.mjs). Auf GitHub Pages läuft es mit 1 Thread –
// bei ~5 ms pro Bild reicht das locker.
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

let session = null;
let source = null;               // HTMLVideoElement | HTMLImageElement
let stream = null;
let busy = false, okStreak = 0, badStreak = 0, isOk = false;
let fpsCount = 0, fpsT0 = performance.now();
const size = MODEL.size;
const roiCanvas = Object.assign(document.createElement('canvas'), { width: size, height: size });
const roiCtx = roiCanvas.getContext('2d', { willReadFrequently: true });

function showError(e) { console.error(e); els.err.textContent = String(e?.message || e); }

async function ensureSession() {
  if (session) return;
  out('oBackend', 'lädt…');
  const t0 = performance.now();
  session = await ort.InferenceSession.create(MODEL.file, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  out('oBackend', `wasm ×${ort.env.wasm.numThreads}`);
  out('oModel', `TireNet ${size}px, geladen in ${Math.round(performance.now() - t0)} ms`);
}

// --- Geometrie --------------------------------------------------------------
function stageGeom() {
  const W = els.stage.clientWidth, H = els.stage.clientHeight;
  return { W, H, cx: W / 2, cy: H / 2, r: W * CIRCLE_R };
}
function srcDims() {
  return source instanceof HTMLVideoElement
    ? { sw: source.videoWidth, sh: source.videoHeight }
    : { sw: source.naturalWidth, sh: source.naturalHeight };
}
// object-fit: cover – Umrechnung Quelle ↔ Bühne
function coverTransform(W, H, sw, sh) {
  const s = Math.max(W / sw, H / sh);
  return { s, ox: (W - sw * s) / 2, oy: (H - sh * s) / 2 };
}

function layoutOverlay() {
  const g = stageGeom();
  for (const c of [els.ring, els.holeCircle]) {
    c.setAttribute('cx', g.cx); c.setAttribute('cy', g.cy); c.setAttribute('r', g.r);
  }
  const bx = g.cx + g.r * Math.cos(-Math.PI / 4), by = g.cy + g.r * Math.sin(-Math.PI / 4);
  els.badge.setAttribute('transform', `translate(${bx} ${by})`);
}
addEventListener('resize', layoutOverlay);

// --- Inferenz ---------------------------------------------------------------
function preprocess() {
  const g = stageGeom();
  const { sw, sh } = srcDims();
  const t = coverTransform(g.W, g.H, sw, sh);
  const side = (2 * g.r * ROI_FACTOR) / t.s;           // Ausschnitt-Kantenlänge in Quellpixeln
  const sx = (g.cx - t.ox) / t.s - side / 2;
  const sy = (g.cy - t.oy) / t.s - side / 2;
  roiCtx.fillStyle = '#000';
  roiCtx.fillRect(0, 0, size, size);
  roiCtx.drawImage(source, sx, sy, side, side, 0, 0, size, size);
  const { data } = roiCtx.getImageData(0, 0, size, size);
  const n = size * size;
  const f = new Float32Array(3 * n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    f[i] = data[p] / 255; f[n + i] = data[p + 1] / 255; f[2 * n + i] = data[p + 2] / 255;
  }
  return { tensor: new ort.Tensor('float32', f, [1, 3, size, size]), roi: { sx, sy, side }, g, t };
}

// Modellausgabe [1, 5]: p (Reifen da?), cx, cy, w, h – relativ zum Ausschnitt
function detection(output, minConf) {
  const [p, cx, cy, w, h] = output.data;
  return p >= minConf ? { cx: cx * size, cy: cy * size, w: w * size, h: h * size, score: p } : null;
}

// Ausschnitt-Koordinaten → Bühnen-Koordinaten
function toStage(det, roi, t) {
  const k = roi.side / size;
  const sx = roi.sx + det.cx * k, sy = roi.sy + det.cy * k;
  return { ...det, x: sx * t.s + t.ox, y: sy * t.s + t.oy, bw: det.w * k * t.s, bh: det.h * k * t.s };
}

function evaluateFit(det, g) {
  if (!det) return { fits: false };
  const center = Math.hypot(det.x - g.cx, det.y - g.cy) / g.r;
  const sz = Math.max(det.bw, det.bh) / (2 * g.r);
  return { fits: center <= CENTER_TOL && sz >= SIZE_MIN && sz <= SIZE_MAX, center, size: sz };
}

function setState(fits) {
  if (fits) { okStreak++; badStreak = 0; } else { badStreak++; okStreak = 0; }
  const next = isOk ? badStreak < HYSTERESIS : okStreak >= HYSTERESIS;
  if (next !== isOk) {
    isOk = next;
    els.overlay.classList.toggle('ok', isOk);
    els.hint.textContent = isOk ? 'Reifen erkannt' : '';
    if (isOk && navigator.vibrate) navigator.vibrate(30);
  }
}

function drawDebug(det) {
  els.dbg.innerHTML = '';
  if (!det || !els.chkBoxes.checked) return;
  els.dbg.insertAdjacentHTML('beforeend',
    `<rect class="dbgbox" x="${det.x - det.bw / 2}" y="${det.y - det.bh / 2}" width="${det.bw}" height="${det.bh}"/>` +
    `<text class="dbglabel" x="${det.x - det.bw / 2 + 3}" y="${det.y - det.bh / 2 + 12}">Reifen ${det.score.toFixed(2)}</text>`);
}

// Schleife per setTimeout statt requestAnimationFrame: läuft so schnell, wie die Inferenz erlaubt.
async function loop() {
  for (;;) {
    await tick();
    await new Promise((r) => setTimeout(r, session && source ? 0 : 100));
  }
}

async function tick() {
  if (busy || !session || !source || document.hidden) return;
  if (!srcDims().sw) return;
  busy = true;
  try {
    const { tensor, roi, g, t } = preprocess();
    const t0 = performance.now();
    const res = await session.run({ [session.inputNames[0]]: tensor });
    const ms = performance.now() - t0;
    const raw = detection(res[session.outputNames[0]], Number(els.rConf.value));
    const det = raw && toStage(raw, roi, t);
    const fit = evaluateFit(det, g);
    setState(fit.fits);
    drawDebug(det);
    out('oInf', ms.toFixed(1));
    out('oBest', det ? `Reifen ${det.score.toFixed(2)}` : 'kein Reifen');
    out('oCenter', fit.center != null ? `${fit.center.toFixed(2)} (≤${CENTER_TOL})` : '–');
    out('oSize', fit.size != null ? `${fit.size.toFixed(2)} (${SIZE_MIN}–${SIZE_MAX})` : '–');
    out('oState', isOk ? 'GRÜN' : 'orange');
    fpsCount++;
    const now = performance.now();
    if (now - fpsT0 > 1000) { out('oFps', (fpsCount * 1000 / (now - fpsT0)).toFixed(1)); fpsCount = 0; fpsT0 = now; }
  } catch (e) {
    showError(e);
  } finally {
    busy = false;
  }
}

// --- Quellen ----------------------------------------------------------------
async function startCamera() {
  els.err.textContent = '';
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error(`Kamera-API nicht verfügbar (secure context: ${self.isSecureContext}). Die Seite braucht HTTPS.`);
  }
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 } },
  });
  els.video.srcObject = stream;
  await els.video.play();
  els.testimg.hidden = true; els.video.hidden = false;
  source = els.video;
  const s = stream.getVideoTracks()[0].getSettings();
  out('oVideo', `${els.video.videoWidth}×${els.video.videoHeight}${s.frameRate ? ` @${Math.round(s.frameRate)}fps` : ''}`);
}

document.addEventListener('visibilitychange', () => {
  if (!document.hidden && source === els.video && stream?.getVideoTracks()[0]?.readyState === 'ended') {
    startCamera().catch(showError);
  }
});

function loadImage(blob) {
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  return new Promise((resolve) => {
    els.testimg.onload = () => {
      els.video.hidden = true; els.testimg.hidden = false; source = els.testimg;
      out('oVideo', `Bild ${els.testimg.naturalWidth}×${els.testimg.naturalHeight}`);
      resolve();
    };
    els.testimg.src = URL.createObjectURL(blob);
  });
}

async function begin(fn) {
  els.start.hidden = true;
  layoutOverlay();
  try {
    await ensureSession();
    await fn();
  } catch (e) {
    showError(e);
    els.start.hidden = false;
  }
}

async function loadExample(url) {
  const blob = await (await fetch(url)).blob();
  await begin(() => loadImage(blob));
}

els.btnStart.addEventListener('click', () => begin(startCamera));
els.fileInput.addEventListener('change', () => els.fileInput.files[0] && begin(() => loadImage(els.fileInput.files[0])));
els.examples.addEventListener('click', (e) => {
  const url = e.target.closest('[data-src]')?.dataset.src;
  if (url) loadExample(url).catch(showError);
});
els.rConf.addEventListener('input', () => out('oConf', els.rConf.value));

// Auslöser: volles Kamerabild als JPEG (zeigt, welche Auflösung ein Upload bekäme)
els.shutter.addEventListener('click', () => {
  if (!source) return;
  const { sw, sh } = srcDims();
  const c = Object.assign(document.createElement('canvas'), { width: sw, height: sh });
  c.getContext('2d').drawImage(source, 0, 0);
  c.toBlob((blob) => {
    els.shotImg.src = URL.createObjectURL(blob);
    els.shotInfo.textContent = `${sw}×${sh}, ${(blob.size / 1024).toFixed(0)} KB, Status beim Auslösen: ${isOk ? 'GRÜN' : 'orange'}`;
    els.shot.hidden = false;
    els.shot.scrollIntoView({ behavior: 'smooth' });
  }, 'image/jpeg', 0.92);
});

layoutOverlay();
loop();
window.__demo = { loadExample };   // für automatisierte Prüfungen
