/**
 * Overlay alignment. A scan with one bright voxel and a mask on that same voxel is drawn
 * through the real SliceView into a recording canvas; the test then checks that the
 * bright pixel, the mask outline, the crosshair and a click all land on the same screen
 * rectangle, in every plane, after resizing, zooming and panning. Pretty pictures would
 * not catch a half-voxel or flipped-axis error; this does.
 */
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import type { VolumeData } from "../api/client";
import { useVolume } from "../state/hooks";
import { CaseViewer } from "../viewer/CaseViewer";
import { geometryMismatch, patientAxis, voxelToPatient } from "../viewer/geometry";
import { SliceView } from "../viewer/SliceView";
import { fitView, panBy, screenToVoxel, voxelToScreen, zoomAt, type ViewState } from "../viewer/transform";
import { crosshair, planeInfo, sliceIndexFor, voxelIndex, type Plane } from "../viewer/volume";

type Pt = [number, number];
const PLANES: Plane[] = ["axial", "coronal", "sagittal"];

// ----- a canvas that records what is drawn -----

class FakePath {
  pts: Pt[] = [];
  moveTo(x: number, y: number) {
    this.pts.push([x, y]);
  }
  lineTo(x: number, y: number) {
    this.pts.push([x, y]);
  }
  closePath() {}
}

interface Recording {
  image: { width: number; height: number; data: Uint8ClampedArray } | null; // last putImageData (bitmaps)
  drawn: { src: HTMLCanvasElement; x: number; y: number; w: number; h: number }[];
  fills: Pt[][];
  strokes: Pt[][];
  lines: Pt[]; // crosshair segments (moveTo/lineTo outside a Path2D)
}
const recordings = new WeakMap<HTMLCanvasElement, Recording>();

function fakeContext(canvas: HTMLCanvasElement) {
  const rec: Recording = { image: null, drawn: [], fills: [], strokes: [], lines: [] };
  recordings.set(canvas, rec);
  let current: Pt[] = [];
  return {
    // Every frame of the main canvas starts with setTransform: keep only the latest frame.
    setTransform: () => Object.assign(rec, { drawn: [], fills: [], strokes: [], lines: [] }),
    fillRect: () => {},
    setLineDash: () => {},
    createImageData: (w: number, h: number) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) }),
    putImageData: (img: Recording["image"]) => {
      rec.image = img;
    },
    drawImage: (src: HTMLCanvasElement, x: number, y: number, w: number, h: number) => rec.drawn.push({ src, x, y, w, h }),
    fill: (p: FakePath) => rec.fills.push(p.pts),
    stroke: (p?: FakePath) => (p ? rec.strokes.push(p.pts) : rec.lines.push(...current)),
    beginPath: () => {
      current = [];
    },
    moveTo: (x: number, y: number) => current.push([x, y]),
    lineTo: (x: number, y: number) => current.push([x, y]),
  };
}

const observers: { cb: ResizeObserverCallback; el: Element }[] = [];
function resizeTo(width: number, height: number) {
  act(() => {
    for (const o of observers) o.cb([{ contentRect: { width, height } } as ResizeObserverEntry], {} as ResizeObserver);
  });
}

beforeAll(() => {
  globalThis.ResizeObserver = class {
    constructor(private cb: ResizeObserverCallback) {}
    observe(el: Element) {
      observers.push({ cb: this.cb, el });
    }
    unobserve() {}
    disconnect() {
      for (let i = observers.length - 1; i >= 0; i--) if (observers[i].cb === this.cb) observers.splice(i, 1);
    }
  } as unknown as typeof ResizeObserver;
  (globalThis as unknown as { Path2D: unknown }).Path2D = FakePath;
  HTMLCanvasElement.prototype.getContext = function (this: HTMLCanvasElement) {
    return fakeContext(this);
  } as unknown as HTMLCanvasElement["getContext"];
  if (!("PointerEvent" in window)) {
    class PointerEventStub extends MouseEvent {
      pointerId: number;
      constructor(type: string, init: PointerEventInit = {}) {
        super(type, init);
        this.pointerId = init.pointerId ?? 1;
      }
    }
    (window as unknown as { PointerEvent: unknown }).PointerEvent = PointerEventStub;
  }
});
afterEach(() => vi.unstubAllGlobals());

// ----- the scan: anisotropic, non-cubic, with the marked voxel away from every centre -----

const SHAPE: [number, number, number] = [6, 8, 10]; // z, y, x
const SPACING: [number, number, number] = [0.8, 1.5, 2.5]; // x, y, z mm
const MARK: [number, number, number] = [4, 2, 7];
const flat = (zyx: [number, number, number]) => (zyx[0] * SHAPE[1] + zyx[1]) * SHAPE[2] + zyx[2];

function scan(): { image: VolumeData; mask: VolumeData } {
  const n = SHAPE[0] * SHAPE[1] * SHAPE[2];
  const img = new Int16Array(n);
  const lab = new Uint8Array(n);
  img[flat(MARK)] = 1000;
  lab[flat(MARK)] = 1;
  const geo = { shape: SHAPE, spacing: SPACING, origin: [-4, 7, 12] as [number, number, number], scale: 1, offset: 0, frame: "C/T0" };
  return { image: { ...geo, data: img }, mask: { ...geo, data: lab } };
}

describe("voxel <-> screen mapping", () => {
  it("draws every voxel at the pixel that a click there selects, in all planes, at any size, zoom and pan", () => {
    const views: [number, number, (v: ViewState, info: ReturnType<typeof planeInfo>, vp: { width: number; height: number }) => ViewState][] = [
      [300, 300, (v) => v],
      [517, 211, (v, info, vp) => zoomAt(info, vp, v, 3.7, 40, 150)],
      [180, 640, (v, info, vp) => panBy(info, vp, zoomAt(info, vp, v, 2.2, 90, 500), -37, 61)],
    ];
    for (const plane of PLANES) {
      const info = planeInfo(SHAPE, SPACING, plane);
      for (const [w, h, make] of views) {
        const vp = { width: w, height: h };
        const view = make(fitView(info), info, vp);
        for (let z = 0; z < SHAPE[0]; z++)
          for (let y = 0; y < SHAPE[1]; y++)
            for (let x = 0; x < SHAPE[2]; x++) {
              const zyx: [number, number, number] = [z, y, x];
              const k = sliceIndexFor(plane, zyx);
              const [sx, sy] = voxelToScreen(info, vp, view, SHAPE, plane, zyx);
              expect(screenToVoxel(info, vp, view, SHAPE, plane, k, sx, sy)).toEqual(zyx);
              // ...and the bitmap pixel at that spot holds this voxel's value.
              const [col, row] = crosshair(SHAPE, plane, zyx);
              expect(voxelIndex(SHAPE, plane, k, col, row)).toBe(flat(zyx));
            }
      }
    }
  });
});

describe("overlay alignment through the real renderer", () => {
  function frame(plane: Plane) {
    const rec = recordings.get(screen.getByTestId(`slice-${plane}`) as HTMLCanvasElement)!;
    const base = rec.drawn[0];
    const bitmap = recordings.get(base.src)!.image!;
    const bright = [];
    for (let i = 0; i < bitmap.width * bitmap.height; i++) if (bitmap.data[i * 4] === 255) bright.push(i);
    expect(bright).toHaveLength(1);
    const col = bright[0] % bitmap.width;
    const row = Math.floor(bright[0] / bitmap.width);
    const kx = base.w / bitmap.width;
    const ky = base.h / bitmap.height;
    return { rec, col, row, rect: [base.x + col * kx, base.y + row * ky, base.x + (col + 1) * kx, base.y + (row + 1) * ky] };
  }

  function expectAligned(plane: Plane, onPosition: ReturnType<typeof vi.fn>) {
    const { rec, col, row, rect } = frame(plane);
    const [x0, y0, x1, y1] = rect;
    // The bright voxel sits at the pixel the display convention says it should.
    expect([col, row]).toEqual(crosshair(SHAPE, plane, MARK));
    // The mask outline (fill and line) traces exactly that pixel's edges.
    expect(rec.fills).toHaveLength(1);
    expect(rec.strokes).toHaveLength(1);
    for (const pts of [...rec.fills, ...rec.strokes]) {
      const xs = pts.map((p) => p[0]);
      const ys = pts.map((p) => p[1]);
      expect(Math.min(...xs)).toBeCloseTo(x0, 6);
      expect(Math.max(...xs)).toBeCloseTo(x1, 6);
      expect(Math.min(...ys)).toBeCloseTo(y0, 6);
      expect(Math.max(...ys)).toBeCloseTo(y1, 6);
    }
    // The crosshair passes through the pixel's centre (to the nearest device pixel).
    const vx = rec.lines[0][0];
    const hy = rec.lines[2][1];
    expect(Math.abs(vx - (x0 + x1) / 2)).toBeLessThanOrEqual(0.5);
    expect(Math.abs(hy - (y0 + y1) / 2)).toBeLessThanOrEqual(0.5);
    // A click in the middle of the drawn pixel selects the marked voxel.
    onPosition.mockClear();
    fireEvent.pointerDown(screen.getByTestId(`slice-${plane}`), { clientX: (x0 + x1) / 2, clientY: (y0 + y1) / 2, pointerId: 9 });
    fireEvent.pointerUp(screen.getByTestId(`slice-${plane}`), { pointerId: 9 });
    expect(onPosition).toHaveBeenLastCalledWith(MARK);
  }

  it.each(PLANES)("%s: image, mask, crosshair and click agree after resize, zoom and pan", (plane) => {
    const { image, mask } = scan();
    const onPosition = vi.fn();
    render(
      <SliceView
        image={image}
        plane={plane}
        position={MARK}
        onPosition={onPosition}
        level={500}
        width={1000}
        smooth={false}
        overlays={[{ id: "m", data: mask, type: "label", opacity: 1, style: "both", colors: { 1: [255, 0, 0] } }]}
      />,
    );
    expectAligned(plane, onPosition);
    resizeTo(437, 263);
    expectAligned(plane, onPosition);
    const wrapper = screen.getByLabelText(new RegExp(`${plane} view`, "i"));
    fireEvent.keyDown(wrapper, { key: "+" });
    fireEvent.keyDown(wrapper, { key: "+" });
    expectAligned(plane, onPosition);
    const canvas = screen.getByTestId(`slice-${plane}`);
    fireEvent.pointerDown(canvas, { clientX: 200, clientY: 120, shiftKey: true, pointerId: 3 });
    fireEvent.pointerMove(canvas, { clientX: 173, clientY: 151, shiftKey: true, pointerId: 3 });
    fireEvent.pointerUp(canvas, { pointerId: 3 });
    expectAligned(plane, onPosition);
    resizeTo(190, 610);
    expectAligned(plane, onPosition);
  });
});

describe("overlay geometry checks", () => {
  const { image, mask } = scan();
  it("accepts an overlay on the same grid and explains every kind of mismatch", () => {
    expect(geometryMismatch(image, mask)).toBeNull();
    expect(geometryMismatch(image, { ...mask, frame: "C/T1" })).toBe("it belongs to scan C/T1, not C/T0");
    expect(geometryMismatch(image, { ...mask, shape: [6, 10, 8] })).toMatch(/matrix 8×10×6 differs from the scan's 10×8×6/);
    expect(geometryMismatch(image, { ...mask, spacing: [0.8, 1.5, 3.0] })).toMatch(/voxel size/);
    expect(geometryMismatch(image, { ...mask, origin: [-4, 7, 17] })).toBe("its grid is shifted 5.0 mm from the scan's");
    expect(geometryMismatch(image, { ...mask, direction: [0, -1, 0, 1, 0, 0, 0, 0, 1] })).toMatch(/rotated/);
    expect(geometryMismatch(image, { ...mask, origin: undefined })).toMatch(/unknown/);
    // Within the server's tolerances counts as the same grid.
    expect(geometryMismatch(image, { ...mask, origin: [-4.005, 7, 12], spacing: [0.8004, 1.5, 2.5] })).toBeNull();
  });

  it("gives the same patient position as the server, including for an oblique scan", () => {
    const v: VolumeData = { data: new Int16Array(512), shape: [8, 8, 8], spacing: [2, 3, 4], origin: [10, 20, 30], direction: [0, -1, 0, 1, 0, 0, 0, 0, 1], scale: 1, offset: 0 };
    // Volume.index_to_physical((1, 2, 3)) for the same geometry prints [4, 26, 34].
    expect(voxelToPatient(v, [1, 2, 3])).toEqual([4, 26, 34]);
    expect([patientAxis(0, -12.5), patientAxis(1, 3), patientAxis(2, -0.01)]).toEqual(["R 12.5", "P 3.0", "S 0.0"]);
  });
});

// ----- switching scans and loading overlays through the real hooks -----

function volumeResponse(v: VolumeData) {
  const headers: Record<string, string> = {
    "X-Shape": v.shape.join(","),
    "X-Spacing": v.spacing.join(","),
    "X-Origin": (v.origin ?? [0, 0, 0]).join(","),
    "X-Direction": (v.direction ?? [1, 0, 0, 0, 1, 0, 0, 0, 1]).join(","),
    "X-Dtype": v.data instanceof Int16Array ? "int16" : "uint8",
    "X-Scale": "1",
    "X-Offset": "0",
    "X-Frame": v.frame ?? "",
  };
  return { ok: true, headers: { get: (h: string) => headers[h] ?? null }, arrayBuffer: async () => v.data.slice().buffer };
}

describe("switching scans", () => {
  it("never hands back the previous scan while the next one loads", async () => {
    const { image } = scan();
    const pending: Record<string, (r: unknown) => void> = {};
    vi.stubGlobal("fetch", vi.fn((url: string) => new Promise((resolve) => (pending[url] = resolve))));
    function Probe() {
      const [url, setUrl] = useState("/test/switch/T0");
      const { volume, loading } = useVolume(url);
      return (
        <button onClick={() => setUrl("/test/switch/T1")}>
          {volume ? volume.frame : loading ? "loading" : "none"}
        </button>
      );
    }
    render(<Probe />);
    await act(async () => pending["/test/switch/T0"](volumeResponse({ ...image, frame: "C/T0" })));
    await waitFor(() => expect(screen.getByRole("button")).toHaveTextContent("C/T0"));
    fireEvent.click(screen.getByRole("button"));
    expect(screen.getByRole("button")).toHaveTextContent("loading");
    await act(async () => pending["/test/switch/T1"](volumeResponse({ ...image, frame: "C/T1" })));
    await waitFor(() => expect(screen.getByRole("button")).toHaveTextContent("C/T1"));
  });

  it("does not draw an overlay from another scan or grid, and says why", async () => {
    const { image, mask } = scan();
    const byUrl: Record<string, VolumeData> = {
      "/test/cv/image": image,
      "/test/cv/good": mask,
      "/test/cv/other-scan": { ...mask, frame: "C/T1" },
      "/test/cv/shifted": { ...mask, origin: [1, 7, 12] },
    };
    vi.stubGlobal("fetch", vi.fn(async (url: string) => volumeResponse(byUrl[url])));
    const legend = [{ name: "Lesion", values: [1], color: [255, 0, 0] as [number, number, number] }];
    render(
      <CaseViewer
        caseId="C"
        tpId="T0"
        imageUrl="/test/cv/image"
        layout="axial"
        overlays={[
          { id: "good", name: "AI drawing", url: "/test/cv/good", type: "label", opacity: 1, legend },
          { id: "other", name: "Follow-up drawing", url: "/test/cv/other-scan", type: "label", opacity: 1, legend },
          { id: "shifted", name: "Shifted map", url: "/test/cv/shifted", type: "label", opacity: 1, legend },
        ]}
      />,
    );
    await waitFor(() => expect(screen.getAllByRole("alert")).toHaveLength(2));
    const alerts = screen.getAllByRole("alert").map((a) => a.textContent);
    expect(alerts).toContain("Not drawn: it belongs to scan C/T1, not C/T0.");
    expect(alerts).toContain("Not drawn: its grid is shifted 5.0 mm from the scan's.");
    // Only the matching overlay reaches the canvas: one outline (fill + line) on the slice through it.
    fireEvent.change(screen.getByLabelText("axial slice"), { target: { value: String(MARK[0]) } });
    const rec = recordings.get(screen.getByTestId("slice-axial") as HTMLCanvasElement)!;
    expect(rec.fills).toHaveLength(1);
    expect(screen.getByText(/R 1\.6 · P 13\.0 · S 22\.0 mm|· [RL] [0-9.]+ · [AP] [0-9.]+ · [IS] [0-9.]+ mm/)).toBeInTheDocument();
  });
});
