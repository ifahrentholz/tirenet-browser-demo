// Mini-Server: http://localhost:8080 (Desktop) und – falls cert/ existiert – https://<LAN-IP>:8443 (Handy).
// Zertifikat erzeugen: siehe README, Abschnitt "Lokal starten".
// COOP/COEP-Header aktivieren crossOriginIsolated → Multi-Thread-WASM für onnxruntime-web.
import http from 'node:http';
import https from 'node:https';
import { readFile, stat } from 'node:fs/promises';
import { existsSync, readFileSync } from 'node:fs';
import { extname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(fileURLToPath(new URL('.', import.meta.url)), 'web');
const types = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.wasm': 'application/wasm', '.onnx': 'application/octet-stream', '.png': 'image/png',
  '.jpg': 'image/jpeg', '.css': 'text/css',
};

async function handler(req, res) {
  const path = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)).replace(/^(\.\.[/\\])+/, '');
  let file = join(root, path);
  if (!file.startsWith(root)) { res.writeHead(403).end(); return; }
  try {
    if ((await stat(file)).isDirectory()) file = join(file, 'index.html');
    const body = await readFile(file);
    res.writeHead(200, {
      'Content-Type': types[extname(file)] || 'application/octet-stream',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cache-Control': 'no-store',
    });
    res.end(body);
  } catch {
    res.writeHead(404).end('not found');
  }
}

http.createServer(handler).listen(8080, '127.0.0.1', () => console.log('Desktop: http://localhost:8080'));

const certDir = fileURLToPath(new URL('./cert/', import.meta.url));
if (existsSync(join(certDir, 'key.pem'))) {
  https.createServer({ key: readFileSync(join(certDir, 'key.pem')), cert: readFileSync(join(certDir, 'cert.pem')) }, handler)
    .listen(8443, '0.0.0.0', () => console.log('Handy:   https://<LAN-IP>:8443 (Zertifikatswarnung bestätigen)'));
}
