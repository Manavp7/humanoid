"""Outputs: an interactive 3D HTML viewer (three.js) and a top-down PNG."""
import json

import cv2
import numpy as np

_RNG = np.random.default_rng(0)


def _subsample(a, n):
    a = np.asarray(a).reshape(-1, 3)
    if len(a) <= n:
        return a
    return a[_RNG.choice(len(a), n, replace=False)]


def _opt(c, key):
    return np.round(c[key], 3).tolist() if key in c else None


def write_outputs(out, changes, static, lo, hi, stats, lines, title, unit):
    data = dict(
        title=title, unit=unit, lo=np.round(lo, 3).tolist(), hi=np.round(hi, 3).tolist(),
        static=np.round(_subsample(static, 25000), 3).tolist(),
        changes=[dict(kind=c["kind"], label=c["label"], text=c["text"],
                      at=_opt(c, "at"), to=_opt(c, "to"), lo=_opt(c, "lo"), hi=_opt(c, "hi"),
                      to_lo=_opt(c, "to_lo"), to_hi=_opt(c, "to_hi")) for c in changes],
        stats=stats,
    )
    (out / "viewer.html").write_text(_html(data))
    _topdown_png(out / "topdown.png", changes, static, lo, hi, unit)
    _report_md(out / "report.md", title, stats, lines, unit)


def _html(data):
    payload = json.dumps(data).replace("</", "<\\/")
    return TEMPLATE.replace("__DATA__", payload)


def _topdown_png(path, changes, static, lo, hi, unit, px=110):
    w = int((hi[0] - lo[0]) * px)
    h = int((hi[1] - lo[1]) * px)
    img = np.full((h, w, 3), 255, np.uint8)

    def xy(p):
        return int((p[0] - lo[0]) * px), int(h - (p[1] - lo[1]) * px)

    for p in _subsample(static, 20000):
        cv2.circle(img, xy(p), 1, (170, 170, 170), -1)
    for c in changes:
        bgr = {"moved": (48, 160, 240), "added": (113, 179, 60), "removed": (74, 72, 224)}[c["kind"]]
        x0, y0 = xy(c["lo"])
        x1, y1 = xy(c["hi"])
        cv2.rectangle(img, (min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1)), bgr, 2)
        if c["kind"] == "moved":
            x0, y0 = xy(c["to_lo"])
            x1, y1 = xy(c["to_hi"])
            cv2.rectangle(img, (min(x0, x1), min(y0, y1)), (max(x0, x1), max(y0, y1)), bgr, 2)
            cv2.arrowedLine(img, xy(c["at"]), xy(c["to"]), bgr, 2, tipLength=0.15)
    cv2.putText(img, f"top-down, x right, y up ({unit})", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    cv2.imwrite(str(path), img)


def _report_md(path, title, stats, lines, unit):
    ev = stats.get("evaluation")
    out = [f"# {title}", "",
           f"Changes found: {stats['changes']} "
           f"(moved {stats['kinds']['moved']}, added {stats['kinds']['added']}, removed {stats['kinds']['removed']}).", ""]
    out += [f"- {line}" for line in lines] or ["No changes found."]
    out += ["", "## Coverage",
            f"- Voxel size {stats['voxel']} {unit}; grid {stats['grid_shape']}.",
            f"- {stats['voxels_seen_in_both_pct']}% of the box was seen by both recordings. Only that part is compared.",
            f"- {stats['voxels_unverified']} voxels are occupied in one recording and unseen in the other. "
            "They are reported as unverified, not as changes."]
    if ev:
        out += ["", "## Evaluation against ground truth",
                f"- Precision {ev['precision']:.2f}, recall {ev['recall']:.2f} "
                f"(TP {ev['true_positive']}, FP {ev['false_positive']}, FN {ev['false_negative']}, "
                f"tolerance {ev['tolerance']} {unit})."]
    path.write_text("\n".join(out) + "\n")


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Room change viewer</title>
<style>
:root{--bg:#f6f6f3;--fg:#1d1d1b;--panel:#ffffff;--line:#dcdcd6;--muted:#6b6b66}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#17181a;--fg:#ecece8;--panel:#202224;--line:#34373b;--muted:#9a9a94}}
:root[data-theme="dark"]{--bg:#17181a;--fg:#ecece8;--panel:#202224;--line:#34373b;--muted:#9a9a94}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,sans-serif;display:flex;height:100vh}
#side{width:320px;flex:none;padding:16px;background:var(--panel);border-right:1px solid var(--line);overflow:auto}
#view{flex:1;position:relative;min-width:0}
h1{font-size:17px;margin:0 0 8px} h2{font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin:18px 0 6px}
.legend span{display:inline-flex;align-items:center;gap:6px;margin-right:12px}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
ul{list-style:none;padding:0;margin:0} li{padding:8px 0;border-bottom:1px solid var(--line);cursor:pointer}
li:hover{color:var(--muted)} .k{font-weight:600;text-transform:capitalize}
#hint{position:absolute;left:12px;bottom:10px;color:var(--muted);font-size:12px}
canvas{display:block}
@media (max-width:700px){body{flex-direction:column;height:auto}#side{width:auto;border-right:none;border-bottom:1px solid var(--line)}#view{height:70vh}}
</style>
<script type="importmap">
{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.158.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.158.0/examples/jsm/"}}
</script>
</head>
<body>
<div id="side">
  <h1 id="title"></h1>
  <div class="legend">
    <span><i class="dot" style="background:#9a9a95"></i>unchanged</span>
    <span><i class="dot" style="background:#e0484a"></i>removed</span>
    <span><i class="dot" style="background:#3cb371"></i>added</span>
    <span><i class="dot" style="background:#f0a030"></i>moved</span>
  </div>
  <h2>Changes</h2>
  <ul id="list"></ul>
  <h2>Notes</h2>
  <p id="notes" style="color:var(--muted);margin:0"></p>
</div>
<div id="view"><div id="hint">Drag to orbit, scroll to zoom, click a change to focus it.</div></div>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const DATA = __DATA__;
const COL = {moved:0xf0a030, added:0x3cb371, removed:0xe0484a};

document.getElementById('title').textContent = DATA.title;
const s = DATA.stats;
document.getElementById('notes').textContent =
  `${s.voxels_seen_in_both_pct}% of the room was seen by both recordings. Only that part is compared. ` +
  `Units: ${DATA.unit}. Labels are shape-based guesses.`;

const view = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
view.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf6f6f3);
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 200);
camera.up.set(0, 0, 1);
const center = new THREE.Vector3(...DATA.lo.map((v, i) => (v + DATA.hi[i]) / 2));
camera.position.copy(center).add(new THREE.Vector3(-6, -6, 5));
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.copy(center);

function points(arr, color, size, opacity) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(arr.flat(), 3));
  const m = new THREE.PointsMaterial({color, size, transparent: true, opacity});
  return new THREE.Points(g, m);
}
function box(lo, hi, color) {
  const size = hi.map((v, i) => v - lo[i]);
  const mid = lo.map((v, i) => (v + hi[i]) / 2);
  const e = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(...size)),
                                   new THREE.LineBasicMaterial({color}));
  e.position.set(...mid);
  return e;
}

scene.add(points(DATA.static, 0x9a9a95, 0.035, 0.7));
const grid = new THREE.GridHelper(Math.max(DATA.hi[0] - DATA.lo[0], DATA.hi[1] - DATA.lo[1]), 10, 0x888888, 0xcccccc);
grid.rotation.x = Math.PI / 2;
grid.position.set(center.x, center.y, DATA.lo[2]);
scene.add(grid);

const list = document.getElementById('list');
DATA.changes.forEach((c) => {
  const col = COL[c.kind];
  scene.add(box(c.lo, c.hi, col));
  let mid = new THREE.Vector3(...c.at);
  if (c.kind === 'moved') {
    scene.add(box(c.to_lo, c.to_hi, col));
    const from = new THREE.Vector3(...c.at), to = new THREE.Vector3(...c.to);
    const dir = to.clone().sub(from);
    scene.add(new THREE.ArrowHelper(dir.clone().normalize(), from, dir.length(), col, 0.2, 0.12));
    mid = from.clone().add(to).multiplyScalar(0.5);
  }
  const li = document.createElement('li');
  li.innerHTML = `<span class="k">${c.kind}</span>: ${c.label}<br><span style="color:var(--muted)">${c.text}</span>`;
  li.onclick = () => { controls.target.copy(mid); camera.position.copy(mid).add(new THREE.Vector3(-3, -3, 2.5)); };
  list.appendChild(li);
});
if (!DATA.changes.length) list.innerHTML = '<li>No changes found.</li>';

function resize() {
  const w = view.clientWidth, h = view.clientHeight || 500;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize); resize();
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
</script>
</body></html>
"""
