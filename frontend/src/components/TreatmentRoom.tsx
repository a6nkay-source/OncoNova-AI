import { useEffect, useRef, useState, type MutableRefObject, type ReactNode } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { api } from "../api/client";
import { BREATH_AMPLITUDE_MM, gantryDisplay, type SimState } from "../sim/treatment";
import { ErrorState, Spinner } from "./ui";

/**
 * A stylised treatment room for the LINAC simulator page. SIMULATED - NON-CLINICAL.
 *
 * Everything is in the case's LPS millimetre frame (x = patient left, y = posterior,
 * z = superior), so the anatomy meshes from the API drop straight in. The patient lies
 * head-first supine; "up" in the room is anterior (-y). The gantry rotates about the
 * superior-inferior axis through the isocentre (the target centre) using the IEC
 * convention of the dose illustration: 0 deg = beam from anterior, 90 deg = from the
 * patient's left. Dimensions are typical of a C-arm linac (1000 mm source-axis
 * distance) but nothing is modelled on a specific machine.
 */

const SAD = 1000;

interface MeshPart {
  key: string;
  source: string;
  label: number;
  material: THREE.Material;
}

function mat(color: number, opacity = 1, extra: THREE.MeshStandardMaterialParameters = {}): THREE.MeshStandardMaterial {
  return new THREE.MeshStandardMaterial({
    color,
    roughness: 0.55,
    metalness: 0.05,
    transparent: opacity < 1,
    opacity,
    depthWrite: opacity >= 0.6,
    side: opacity < 1 ? THREE.DoubleSide : THREE.FrontSide,
    ...extra,
  });
}

function box(w: number, h: number, d: number, m: THREE.Material, x: number, y: number, z: number): THREE.Mesh {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), m);
  mesh.position.set(x, y, z);
  return mesh;
}

/** Capsule or cylinder with its long axis along z (superior-inferior). */
function alongZ(geom: THREE.BufferGeometry, m: THREE.Material, x: number, y: number, z: number): THREE.Mesh {
  geom.rotateX(Math.PI / 2);
  const mesh = new THREE.Mesh(geom, m);
  mesh.position.set(x, y, z);
  return mesh;
}

function geometryFrom(vertices: number[], faces: number[]): THREE.BufferGeometry {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  g.setIndex(faces);
  g.computeVertexNormals();
  return g;
}

export function TreatmentRoom({ caseId, tpId, targetSource, hasStructures, simRef, marginMm, height = 560, overlay }: {
  caseId: string;
  tpId: string;
  targetSource: string;
  hasStructures: boolean;
  simRef: MutableRefObject<SimState>;
  marginMm: number;
  height?: number;
  overlay?: ReactNode;
}) {
  const mountRef = useRef<HTMLDivElement>(null);
  const marginRef = useRef(marginMm);
  marginRef.current = marginMm;
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    let disposed = false;
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      setError(new Error("WebGL is not available in this browser, so the treatment room cannot be shown."));
      setLoading(false);
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, height);
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, mount.clientWidth / height, 10, 30000);
    camera.up.set(0, -1, 0); // anterior is up in the room
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.maxPolarAngle = Math.PI * 0.49;

    const hemi = new THREE.HemisphereLight(0xdfe8ff, 0x141a26, 1.1);
    hemi.position.set(0, -1, 0);
    scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 1.4);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x7fb8ff, 0.5);
    scene.add(rim);

    const materials = {
      machine: mat(0xe6ebf1, 1, { roughness: 0.45 }),
      machineDark: mat(0x2b3547, 1, { roughness: 0.7 }),
      accent: mat(0x2fd0b8, 1, { emissive: 0x0b4a42, roughness: 0.4 }),
      couch: mat(0x1b2231, 1, { roughness: 0.5, metalness: 0.25 }),
      floor: mat(0x1a2336, 1, { roughness: 0.95 }),
      dummy: mat(0xd3deee, 0.34, { roughness: 0.6 }),
      lungs: mat(0x62a8ff, 0.16),
      heart: mat(0xff6e82, 0.35),
      cord: mat(0xffe25a, 0.7),
      target: mat(0x2fd0b8, 0.95, { emissive: 0x1f8c7c, emissiveIntensity: 0.6 }),
    };
    const parts: MeshPart[] = [
      ...(hasStructures
        ? [
            { key: "body", source: "structures", label: 6, material: materials.dummy },
            { key: "lungL", source: "structures", label: 1, material: materials.lungs },
            { key: "lungR", source: "structures", label: 2, material: materials.lungs },
            { key: "heart", source: "structures", label: 3, material: materials.heart },
            { key: "cord", source: "structures", label: 5, material: materials.cord },
          ]
        : []),
      { key: "target", source: targetSource, label: 1, material: materials.target },
    ];

    // Groups that the animation loop moves.
    const chest = new THREE.Group(); // scales anteriorly about the couch surface as the dummy breathes
    const belly = new THREE.Group();
    const lungs = new THREE.Group();
    const heart = new THREE.Group();
    const target = new THREE.Group();
    const gantry = new THREE.Group();
    const beam = new THREE.Group();
    let lamp: THREE.MeshStandardMaterial | null = null;
    let beamMats: THREE.MeshBasicMaterial[] = [];
    let couchTopY = 0;
    let targetHalf = 20;
    let builtMargin = 5;
    let iso = new THREE.Vector3();

    setLoading(true);
    setError(null);
    Promise.allSettled(parts.map((p) => api.mesh(caseId, tpId, p.source, p.label))).then((results) => {
      if (disposed) return;
      const meshes: Record<string, THREE.Mesh> = {};
      results.forEach((r, i) => {
        if (r.status === "fulfilled" && r.value.vertices.length) {
          meshes[parts[i].key] = new THREE.Mesh(geometryFrom(r.value.vertices, r.value.faces), parts[i].material);
        }
      });
      if (!meshes.target) {
        setError(new Error("No target contour to aim at. The AI drawing or a reference outline of the primary lesion is needed."));
      }

      // Isocentre at the target centre; torso size from the body surface when there is one.
      const tBox = meshes.target ? new THREE.Box3().setFromObject(meshes.target) : null;
      const bBox = meshes.body ? new THREE.Box3().setFromObject(meshes.body) : null;
      iso = tBox ? tBox.getCenter(new THREE.Vector3()) : bBox ? bBox.getCenter(new THREE.Vector3()) : new THREE.Vector3();
      const torso =
        bBox ??
        new THREE.Box3(new THREE.Vector3(iso.x - 170, iso.y - 110, iso.z - 150), new THREE.Vector3(iso.x + 170, iso.y + 110, iso.z + 150));
      const cx = (torso.min.x + torso.max.x) / 2;
      const halfW = (torso.max.x - torso.min.x) / 2;
      const halfD = (torso.max.y - torso.min.y) / 2;
      const top = torso.max.z; // superior end of the scanned section
      const bottom = torso.min.z;
      couchTopY = torso.max.y + 2;
      const floorY = iso.y + 1300;

      // ---- dummy patient ------------------------------------------------------------
      const pivot = (g: THREE.Group, child: THREE.Object3D) => {
        g.position.set(0, couchTopY, 0);
        child.position.y -= couchTopY;
        g.add(child);
      };
      const chestInner = new THREE.Group();
      if (meshes.body) chestInner.add(meshes.body);
      else chestInner.add(alongZ(new THREE.CylinderGeometry(1, 1, 1, 48), materials.dummy, cx, couchTopY - halfD, (top + bottom) / 2));
      if (!meshes.body) chestInner.children[0].scale.set(halfW, halfD, top - bottom);
      chestInner.add(alongZ(new THREE.CapsuleGeometry(55, Math.max(halfW * 2 - 110, 40), 6, 16), materials.dummy, cx, couchTopY - 75, top - 30).rotateY(Math.PI / 2));
      pivot(chest, chestInner);
      scene.add(chest);

      const bellyInner = new THREE.Group();
      const abdomen = alongZ(new THREE.CylinderGeometry(1, 1, 1, 48), materials.dummy, cx, couchTopY - halfD * 0.9, bottom - 150);
      abdomen.scale.set(halfW * 0.93, halfD * 0.9, 300);
      const pelvis = alongZ(new THREE.CylinderGeometry(1, 1, 1, 48), materials.dummy, cx, couchTopY - halfD * 0.8, bottom - 420);
      pelvis.scale.set(halfW * 0.98, halfD * 0.8, 240);
      bellyInner.add(abdomen, pelvis);
      pivot(belly, bellyInner);
      scene.add(belly);

      const neckZ = top + 70;
      scene.add(alongZ(new THREE.CylinderGeometry(52, 58, 140, 24), materials.dummy, cx, couchTopY - 82, neckZ));
      const head = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 24), materials.dummy);
      head.scale.set(78, 92, 108);
      head.position.set(cx, couchTopY - 94, top + 140 + 108);
      scene.add(head);
      for (const side of [-1, 1]) {
        scene.add(alongZ(new THREE.CapsuleGeometry(42, 560, 6, 16), materials.dummy, cx + side * (halfW + 52), couchTopY - 48, top - 380));
        scene.add(alongZ(new THREE.CapsuleGeometry(80, 360, 6, 16), materials.dummy, cx + side * 98, couchTopY - 82, bottom - 540 - 260));
        scene.add(alongZ(new THREE.CapsuleGeometry(58, 360, 6, 16), materials.dummy, cx + side * 98, couchTopY - 60, bottom - 540 - 700));
        scene.add(box(92, 150, 100, materials.dummy, cx + side * 98, couchTopY - 120, bottom - 540 - 960));
      }

      if (meshes.lungL) lungs.add(meshes.lungL);
      if (meshes.lungR) lungs.add(meshes.lungR);
      if (meshes.cord) scene.add(meshes.cord);
      if (meshes.heart) heart.add(meshes.heart);
      if (meshes.target) target.add(meshes.target);
      scene.add(lungs, heart, target);

      // ---- couch ----------------------------------------------------------------------
      const headEnd = top + 420;
      const couchLen = 2300;
      scene.add(box(540, 50, couchLen, materials.couch, cx, couchTopY + 25, headEnd - couchLen / 2));
      scene.add(box(520, 6, couchLen - 40, materials.machineDark, cx, couchTopY + 1, headEnd - couchLen / 2));
      const pedestalTop = couchTopY + 50;
      scene.add(box(400, floorY - pedestalTop - 120, 640, materials.machine, cx, (pedestalTop + floorY - 120) / 2, headEnd - 1500));
      scene.add(box(760, 120, 1100, materials.machine, cx, floorY - 60, headEnd - 1500));

      // ---- room -----------------------------------------------------------------------
      const floor = new THREE.Mesh(new THREE.PlaneGeometry(9000, 9000), materials.floor);
      floor.rotation.x = Math.PI / 2; // normal points to -y (up)
      floor.position.set(iso.x, floorY, iso.z);
      scene.add(floor);
      const grid = new THREE.GridHelper(9000, 30, 0x34476a, 0x26344f);
      grid.position.set(iso.x, floorY - 1, iso.z);
      scene.add(grid);

      // ---- gantry: static stand, rotating drum + arm + head + imager --------------------
      const standZ = iso.z + 1420;
      scene.add(box(1900, 2700, 700, materials.machine, iso.x, floorY - 1350, standZ));
      scene.add(box(1920, 60, 720, materials.accent, iso.x, floorY - 2700 + 160, standZ));
      lamp = mat(0x553322, 1, { emissive: 0x000000 });
      const lampMesh = new THREE.Mesh(new THREE.CylinderGeometry(60, 60, 40, 24), lamp);
      lampMesh.position.set(iso.x + 700, floorY - 2700 - 20, standZ);
      scene.add(lampMesh);

      gantry.position.copy(iso);
      gantry.add(alongZ(new THREE.CylinderGeometry(830, 830, 60, 72), materials.machine, 0, 0, 1040));
      const ring = new THREE.Mesh(new THREE.TorusGeometry(830, 10, 8, 96), materials.accent); // lies in the gantry plane
      ring.position.set(0, 0, 1005);
      gantry.add(ring);
      gantry.add(box(520, 330, 1000, materials.machine, 0, -1090, 520)); // arm
      gantry.add(box(440, 520, 560, materials.machine, 0, -930, 0)); // treatment head
      gantry.add(box(446, 40, 566, materials.accent, 0, -1040, 0));
      const collimator = new THREE.Mesh(new THREE.CylinderGeometry(190, 220, 90, 48), materials.machineDark);
      collimator.position.set(0, -625, 0);
      gantry.add(collimator);
      gantry.add(box(140, 60, 1000, materials.machine, 0, 600, 520)); // imager arm
      gantry.add(box(430, 30, 430, materials.machineDark, 0, 545, 0)); // imaging panel opposite the head

      // Beam: diverging cone from the source through the isocentre, shown only while the beam is on.
      const size = tBox ? tBox.getSize(new THREE.Vector3()) : new THREE.Vector3(40, 40, 40);
      targetHalf = Math.max(size.x, size.y, size.z) / 2;
      builtMargin = marginRef.current;
      const r = targetHalf + builtMargin;
      const coneLen = SAD + 380;
      const makeCone = (scale: number, opacity: number) => {
        const geom = new THREE.CylinderGeometry(r * scale * 0.03, r * scale * (coneLen / SAD), coneLen, 40, 1, true);
        geom.rotateX(Math.PI); // narrow end towards the source at -y
        const m = new THREE.MeshBasicMaterial({
          color: 0xf08cff,
          transparent: true,
          opacity,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
          side: THREE.DoubleSide,
        });
        beamMats.push(m);
        const cone = new THREE.Mesh(geom, m);
        cone.position.set(0, -SAD + coneLen / 2, 0);
        return cone;
      };
      beamMats = [];
      beam.add(makeCone(1, 0.13), makeCone(0.55, 0.16));
      beam.visible = false;
      gantry.add(beam);
      scene.add(gantry);

      // Room lasers crossing at the isocentre.
      const laserGeom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-1100, 0, 0), new THREE.Vector3(1100, 0, 0),
        new THREE.Vector3(0, -900, 0), new THREE.Vector3(0, 400, 0),
        new THREE.Vector3(0, 0, -900), new THREE.Vector3(0, 0, 600),
      ]);
      const lasers = new THREE.LineSegments(laserGeom, new THREE.LineBasicMaterial({ color: 0x3dff9a, transparent: true, opacity: 0.55 }));
      lasers.position.copy(iso);
      scene.add(lasers);

      key.position.set(iso.x - 900, iso.y - 2600, iso.z - 1400);
      key.target.position.copy(iso);
      rim.target.position.copy(iso);
      scene.add(key.target, rim.target);
      rim.position.set(iso.x + 2000, iso.y - 600, iso.z + 1800);
      camera.position.set(iso.x - 3300, iso.y - 1700, iso.z - 2300);
      controls.target.set(iso.x, iso.y + 250, iso.z - 150);
      controls.update();
      setLoading(false);
    });

    let frame = 0;
    const animate = () => {
      frame = requestAnimationFrame(animate);
      const s = simRef.current;
      const b = s.breath;
      chest.scale.y = 1 + 0.035 * b;
      belly.scale.y = 1 + 0.07 * b;
      lungs.position.z = -0.5 * BREATH_AMPLITUDE_MM * b;
      heart.position.z = -0.25 * BREATH_AMPLITUDE_MM * b;
      target.position.set(0, -1.5 * b, -BREATH_AMPLITUDE_MM * b); // inhale moves the lesion inferiorly
      gantry.rotation.z = THREE.MathUtils.degToRad(gantryDisplay(s.g));
      beam.visible = s.beamOn;
      // Field opens with the scenario margin without rebuilding the scene.
      const open = (targetHalf + marginRef.current) / (targetHalf + builtMargin);
      beam.scale.set(open, 1, open);
      const flicker = 0.9 + 0.1 * Math.sin(performance.now() / 45);
      beamMats.forEach((m, i) => (m.opacity = (i === 0 ? 0.13 : 0.16) * flicker));
      if (lamp) {
        lamp.emissive.setHex(s.beamOn ? 0xff3b5c : s.hold ? 0xffb020 : 0x000000);
        lamp.emissiveIntensity = s.beamOn || s.hold ? 1.4 : 0;
      }
      controls.update();
      renderer.render(scene, camera);
    };
    animate();
    const ro = new ResizeObserver(() => {
      const w = mount.clientWidth;
      camera.aspect = w / height;
      camera.updateProjectionMatrix();
      renderer.setSize(w, height);
    });
    ro.observe(mount);
    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      ro.disconnect();
      controls.dispose();
      scene.traverse((o) => {
        if (o instanceof THREE.Mesh || o instanceof THREE.LineSegments) {
          o.geometry.dispose();
          const m = o.material as THREE.Material | THREE.Material[];
          (Array.isArray(m) ? m : [m]).forEach((x) => x.dispose());
        }
      });
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, [caseId, tpId, targetSource, hasStructures, height, simRef]);

  return (
    <div className="relative overflow-hidden rounded-lg border border-ink-700/70 bg-gradient-to-b from-[#121b2e] to-ink-950">
      <div ref={mountRef} style={{ height }} className="w-full" />
      {overlay}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <Spinner label="Setting up the room" />
        </div>
      )}
      {error && !loading ? (
        <div className="absolute inset-x-3 bottom-12">
          <ErrorState error={error} />
        </div>
      ) : null}
    </div>
  );
}
