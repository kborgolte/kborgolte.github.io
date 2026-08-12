// Punktwolken-Viewer auf three.js.
//
// Geladen wird points.bin, nicht das GLB: bei Millionen Punkten ist ein
// glTF-Parser reine Zeitverschwendung, wenn die Daten ohnehin schon als
// dichtes Float32-Array vorliegen. Das GLB gibt es zum Herunterladen für
// Blender und Co.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const MAGIC = 0x4350474b; // "KGPC" als Little-Endian-uint32

// LingBot-Map rechnet in Kamerakonvention: y zeigt nach unten, z nach vorn.
// three.js erwartet y nach oben. Eine halbe Drehung um X bringt beides in
// Deckung — ohne sie steht jede Szene auf dem Kopf.
const FLIP = new THREE.Euler(Math.PI, 0, 0);

function flipVector([x, y, z]) {
  return new THREE.Vector3(x, -y, -z);
}

export class SceneViewer {
  constructor(container) {
    this.container = container;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x090c10);

    this.camera = new THREE.PerspectiveCamera(60, 1, 0.01, 5000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.rotateSpeed = 0.7;

    this.group = new THREE.Group();
    this.group.rotation.copy(FLIP);
    this.scene.add(this.group);

    this.points = null;
    this.path = null;
    this.home = null;

    this._resize();
    new ResizeObserver(() => this._resize()).observe(container);
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  _resize() {
    const { clientWidth: width, clientHeight: height } = this.container;
    if (!width || !height) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  clear() {
    for (const object of [this.points, this.path]) {
      if (!object) continue;
      this.group.remove(object);
      object.geometry.dispose();
      object.material.dispose();
    }
    this.points = null;
    this.path = null;
  }

  /** points.bin einlesen: 12 Byte Kopf, dann xyz als float32, dann rgb als uint8. */
  static parsePoints(buffer) {
    const header = new DataView(buffer, 0, 12);
    if (header.getUint32(0, true) !== MAGIC) {
      throw new Error('Keine gültige points.bin (Magic passt nicht)');
    }

    const count = header.getUint32(8, true);
    const positions = new Float32Array(buffer, 12, count * 3);
    const rgb = new Uint8Array(buffer, 12 + count * 12, count * 3);

    // three.js will Farben als Float 0..1. Der Umweg kostet etwas Speicher,
    // spart aber einen eigenen Shader.
    const colors = new Float32Array(count * 3);
    for (let i = 0; i < colors.length; i++) colors[i] = rgb[i] / 255;

    return { count, positions, colors };
  }

  async load(jobId, meta) {
    const response = await fetch(`/api/scene/${jobId}/points.bin`);
    if (!response.ok) throw new Error(`Punktwolke nicht abrufbar (${response.status})`);

    const { count, positions, colors } = SceneViewer.parsePoints(await response.arrayBuffer());

    this.clear();

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const extent = meta?.extent || 10;

    // An die Szenengröße gekoppelt, sonst sind kleine Räume Konfetti und große
    // leer. Als Feld gemerkt, damit der Regler von einem festen Wert aus
    // skaliert und nicht von der aktuellen Kameraentfernung.
    this.baseSize = extent / 900;

    this.points = new THREE.Points(
      geometry,
      new THREE.PointsMaterial({
        size: this.baseSize,
        sizeAttenuation: true,
        vertexColors: true,
      }),
    );
    this.group.add(this.points);

    if (meta?.camera_path?.length > 1) this._addCameraPath(meta.camera_path);

    this._frame(meta, extent);
    return count;
  }

  _addCameraPath(pathPoints) {
    const geometry = new THREE.BufferGeometry().setFromPoints(pathPoints.map(flipVector));
    this.path = new THREE.Line(
      geometry,
      new THREE.LineBasicMaterial({ color: 0x4cc2ff, transparent: true, opacity: 0.75 }),
    );
    this.group.add(this.path);
  }

  /** Kamera so setzen, dass die ganze Szene ins Bild passt. */
  _frame(meta, extent) {
    const center = meta?.center ? flipVector(meta.center) : new THREE.Vector3();
    const distance = Math.max(extent * 0.75, 1.5);

    this.home = {
      position: center.clone().add(new THREE.Vector3(distance, distance * 0.45, distance)),
      target: center,
    };
    this.resetView();
  }

  resetView() {
    if (!this.home) return;
    this.camera.position.copy(this.home.position);
    this.controls.target.copy(this.home.target);
    this.controls.update();
  }

  setPointSize(scale) {
    if (!this.points) return;
    this.points.material.size = this.baseSize * scale;
  }

  setPathVisible(visible) {
    if (this.path) this.path.visible = visible;
  }
}
