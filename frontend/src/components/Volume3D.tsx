import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { api } from "../api/client";
import { ErrorState, Spinner } from "./ui";

export interface MeshSpec {
  id: string;
  source: string; // reference | annotation | pred:<model> | structures
  label: number;
  color: string;
  opacity: number;
  name: string;
}

export const DEFAULT_ANATOMY: MeshSpec[] = [
  { id: "lungL", source: "structures", label: 1, color: "#62a8ff", opacity: 0.1, name: "Left lung" },
  { id: "lungR", source: "structures", label: 2, color: "#7c8cff", opacity: 0.1, name: "Right lung" },
  { id: "heart", source: "structures", label: 3, color: "#ff6e82", opacity: 0.22, name: "Heart" },
  { id: "cord", source: "structures", label: 5, color: "#ffe25a", opacity: 0.6, name: "Spinal cord" },
];

export function lesionMeshes(source: string): MeshSpec[] {
  return [
    { id: `${source}-1`, source, label: 1, color: "#2fd0b8", opacity: 0.95, name: "Primary lesion" },
    { id: `${source}-2`, source, label: 2, color: "#f4b740", opacity: 0.95, name: "Node-region lesion" },
    { id: `${source}-3`, source, label: 3, color: "#e58cff", opacity: 0.95, name: "Other lesion" },
  ];
}

export function Volume3D({ caseId, tpId, meshes, version = 0, height = 360 }: {
  caseId: string;
  tpId: string;
  meshes: MeshSpec[];
  version?: number;
  height?: number;
}) {
  const mountRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [legend, setLegend] = useState<{ name: string; color: string }[]>([]);
  const key = meshes.map((m) => `${m.source}:${m.label}:${m.opacity}`).join("|");

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    let disposed = false;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, mount.clientWidth / height, 1, 5000);
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch (e) {
      setError(new Error("WebGL is not available in this browser, so the 3-D view cannot be shown."));
      setLoading(false);
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(mount.clientWidth, height);
    mount.appendChild(renderer.domElement);
    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key1 = new THREE.DirectionalLight(0xffffff, 1.1);
    key1.position.set(-200, -300, 250);
    scene.add(key1);
    const rim = new THREE.DirectionalLight(0x7fb8ff, 0.6);
    rim.position.set(250, 300, -100);
    scene.add(rim);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    // LPS: view from anterior (-y) with superior (+z) up, so patient left appears on the right.
    camera.up.set(0, 0, 1);
    camera.position.set(0, -520, 40);
    controls.update();

    const group = new THREE.Group();
    scene.add(group);
    setLoading(true);
    setError(null);
    Promise.allSettled(meshes.map((m) => api.mesh(caseId, tpId, m.source, m.label))).then((results) => {
      if (disposed) return;
      const shown: { name: string; color: string }[] = [];
      results.forEach((r, i) => {
        if (r.status !== "fulfilled" || !r.value.vertices.length) return;
        const spec = meshes[i];
        const geom = new THREE.BufferGeometry();
        geom.setAttribute("position", new THREE.Float32BufferAttribute(r.value.vertices, 3));
        geom.setIndex(r.value.faces);
        geom.computeVertexNormals();
        const mat = new THREE.MeshPhongMaterial({
          color: spec.color,
          transparent: spec.opacity < 1,
          opacity: spec.opacity,
          depthWrite: spec.opacity >= 0.5,
          shininess: 40,
          side: THREE.DoubleSide,
        });
        const meshObj = new THREE.Mesh(geom, mat);
        meshObj.renderOrder = spec.opacity < 0.5 ? 2 : 1;
        group.add(meshObj);
        shown.push({ name: spec.name, color: spec.color });
      });
      setLegend(shown);
      const box = new THREE.Box3().setFromObject(group);
      if (!box.isEmpty()) {
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3()).length();
        controls.target.copy(center);
        camera.position.set(center.x, center.y - size * 1.3, center.z + size * 0.15);
        controls.update();
      }
      setLoading(false);
      if (!shown.length) setError(new Error("No surfaces to display for this selection."));
    });

    let frame = 0;
    const animate = () => {
      frame = requestAnimationFrame(animate);
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
      group.traverse((o) => {
        if (o instanceof THREE.Mesh) {
          o.geometry.dispose();
          (o.material as THREE.Material).dispose();
        }
      });
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [caseId, tpId, key, version, height]);

  return (
    <div className="relative overflow-hidden rounded-lg border border-ink-700/70 bg-gradient-to-b from-ink-900 to-ink-950">
      <div ref={mountRef} style={{ height }} className="w-full" />
      <div className="pointer-events-none absolute left-2 top-2 flex flex-col gap-1">
        {legend.map((l) => (
          <span key={l.name} className="flex items-center gap-1.5 text-[11px] text-ink-300">
            <span className="h-2 w-2 rounded-full" style={{ background: l.color }} />
            {l.name}
          </span>
        ))}
      </div>
      <span className="pointer-events-none absolute bottom-2 right-2 text-[10.5px] text-ink-500">Anterior view · drag to rotate</span>
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center">
          <Spinner label="Building surfaces" />
        </div>
      )}
      {error && !loading ? (
        <div className="absolute inset-x-3 bottom-8">
          <ErrorState error={error} />
        </div>
      ) : null}
    </div>
  );
}
