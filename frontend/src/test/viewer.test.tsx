import { fireEvent, render, screen } from "@testing-library/react";
import { beforeAll, describe, expect, it, vi } from "vitest";
import type { VolumeData } from "../api/client";
import { loopArea, traceContours, type Loop } from "../viewer/contours";
import { SliceView } from "../viewer/SliceView";
import { MAX_ZOOM, fitView, panBy, pixelSize, planeToScreen, screenToPlane, zoomAt } from "../viewer/transform";
import { planeInfo, sliceLabels } from "../viewer/volume";

const grid = (w: number, h: number, inside: (c: number, r: number) => boolean) => {
  const a = new Uint8Array(w * h);
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) a[r * w + c] = inside(c, r) ? 1 : 0;
  return a;
};
const count = (a: Uint8Array, v = 1) => a.reduce((n, x) => n + (x === v ? 1 : 0), 0);
const evenOddArea = (loops: Loop[]) => {
  // Largest loop is the outside; any others in these shapes are holes.
  const areas = loops.map((l) => Math.abs(loopArea(l))).sort((a, b) => b - a);
  return areas[0] - areas.slice(1).reduce((s, a) => s + a, 0);
};

describe("label outlines (display only)", () => {
  it("returns nothing for an empty slice", () => {
    expect(traceContours(new Uint8Array(16), 4, 4, 1)).toEqual([]);
  });

  it("encloses each simple region's voxel count minus exactly half a voxel", () => {
    const shapes: Record<string, [number, number, (c: number, r: number) => boolean]> = {
      single: [5, 5, (c, r) => c === 2 && r === 2],
      rect: [9, 7, (c, r) => c >= 2 && c < 7 && r >= 2 && r < 5],
      disc: [24, 24, (c, r) => (c - 11.5) ** 2 + (r - 11.5) ** 2 <= 64],
      lShape: [8, 8, (c, r) => (c >= 1 && c < 3 && r >= 1 && r < 7) || (r >= 5 && r < 7 && c >= 1 && c < 6)],
      touchingBorder: [6, 4, (c, r) => c < 3 && r < 4],
    };
    for (const [name, [w, h, inside]] of Object.entries(shapes)) {
      const mask = grid(w, h, inside);
      const loops = traceContours(mask, w, h, 1);
      expect(loops, name).toHaveLength(1);
      expect(Math.abs(loopArea(loops[0])), name).toBeCloseTo(count(mask) - 0.5, 6);
    }
  });

  it("keeps holes, and a ring's outline encloses exactly its voxel count", () => {
    const ring = grid(12, 12, (c, r) => c >= 1 && c < 11 && r >= 1 && r < 11 && !(c >= 4 && c < 8 && r >= 4 && r < 8));
    const loops = traceContours(ring, 12, 12, 1);
    expect(loops).toHaveLength(2);
    expect(evenOddArea(loops)).toBeCloseTo(count(ring), 6);
  });

  it("puts every vertex on a voxel edge between an inside and an outside voxel", () => {
    const w = 20;
    const mask = grid(w, w, (c, r) => (c - 9) ** 2 / 36 + (r - 10) ** 2 / 16 <= 1 || (c > 14 && r < 4));
    const at = (c: number, r: number) => (c < 0 || r < 0 || c >= w || r >= w ? 0 : mask[r * w + c]);
    for (const loop of traceContours(mask, w, w, 1))
      for (let p = 0; p < loop.length; p += 2) {
        const [x, y] = [loop[p], loop[p + 1]];
        // Either on a vertical voxel edge (integer x, mid-row y) or a horizontal one.
        const [a, b] = Number.isInteger(x) ? [at(x - 1, y - 0.5), at(x, y - 0.5)] : [at(x - 0.5, y - 1), at(x - 0.5, y)];
        expect(Number.isInteger(x) !== Number.isInteger(y)).toBe(true);
        expect(a + b).toBe(1);
      }
  });

  it("separates label values and keeps diagonal-only neighbours apart", () => {
    const w = 6;
    const labels = new Uint8Array(w * w);
    labels[1 * w + 1] = 2;
    labels[2 * w + 2] = 2; // diagonal to the first
    labels[4 * w + 4] = 3;
    expect(traceContours(labels, w, w, 2)).toHaveLength(2);
    expect(traceContours(labels, w, w, 3)).toHaveLength(1);
    expect(traceContours(labels, w, w, 1)).toHaveLength(0);
  });
});

describe("view transform", () => {
  // Coronal plane of a 72 x 128 x 128 volume with 2.2 x 2.2 x 2.5 mm voxels: anisotropic on screen.
  const info = planeInfo([72, 128, 128], [2.2, 2.2, 2.5], "coronal");
  const vp = { width: 640, height: 420 };

  it("fits the whole slice at its physical aspect ratio", () => {
    const view = fitView(info);
    const [x0, y0] = planeToScreen(info, vp, view, 0, 0);
    const [x1, y1] = planeToScreen(info, vp, view, info.width, info.height);
    expect(x0).toBeGreaterThanOrEqual(-1e-9);
    expect(y0).toBeGreaterThanOrEqual(-1e-9);
    expect(x1).toBeLessThanOrEqual(vp.width + 1e-9);
    expect(y1).toBeLessThanOrEqual(vp.height + 1e-9);
    expect((x1 - x0) / (y1 - y0)).toBeCloseTo(info.widthMm / info.heightMm, 9);
    const [kx, ky] = pixelSize(info, vp, 1);
    expect(kx / ky).toBeCloseTo(2.2 / 2.5, 9);
  });

  it("maps screen and plane coordinates back and forth exactly at any zoom and pan", () => {
    let view = fitView(info);
    for (const [f, dx, dy] of [
      [2.5, 30, -12],
      [0.6, -80, 40],
      [7, 5, 5],
    ] as const) {
      view = panBy(info, vp, zoomAt(info, vp, view, f, 100, 200), dx, dy);
      for (const [u, v] of [
        [0, 0],
        [17.5, 33.25],
        [info.width, info.height],
      ]) {
        const [x, y] = planeToScreen(info, vp, view, u, v);
        const [u2, v2] = screenToPlane(info, vp, view, x, y);
        expect(u2).toBeCloseTo(u, 9);
        expect(v2).toBeCloseTo(v, 9);
      }
    }
  });

  it("zooms about the cursor so the point under it stays put", () => {
    const view = fitView(info);
    const before = screenToPlane(info, vp, view, 410, 130);
    const zoomed = zoomAt(info, vp, view, 3, 410, 130);
    expect(zoomed.zoom).toBe(3);
    const [x, y] = planeToScreen(info, vp, zoomed, before[0], before[1]);
    expect(x).toBeCloseTo(410, 9);
    expect(y).toBeCloseTo(130, 9);
    expect(zoomAt(info, vp, zoomed, 1000, 0, 0).zoom).toBe(MAX_ZOOM);
  });

  it("never pans the slice centre off the image", () => {
    const view = panBy(info, vp, fitView(info), 1e6, -1e6);
    expect(view.center).toEqual([0, info.height]);
  });
});

describe("slice viewer controls", () => {
  beforeAll(() => {
    globalThis.ResizeObserver ??= class {
      observe() {}
      unobserve() {}
      disconnect() {}
    } as unknown as typeof ResizeObserver;
    // jsdom has no 2-D canvas; the viewer must still mount and handle input.
    HTMLCanvasElement.prototype.getContext = vi.fn(() => null) as unknown as HTMLCanvasElement["getContext"];
    // Nor pointer events: a MouseEvent with a pointerId is enough for the viewer.
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

  const image: VolumeData = { data: new Int16Array(10 * 8 * 6), shape: [10, 8, 6], spacing: [1, 1, 2], origin: [-3, -4, -10], scale: 1, offset: 0 };

  it("changes slice with the keyboard and the slider, staying in range", () => {
    const onPosition = vi.fn();
    const { rerender } = render(<SliceView image={image} plane="axial" position={[4, 3, 2]} onPosition={onPosition} level={0} width={100} />);
    const view = screen.getByLabelText(/axial view/i);
    fireEvent.keyDown(view, { key: "ArrowUp" });
    expect(onPosition).toHaveBeenLastCalledWith([5, 3, 2]);
    fireEvent.keyDown(view, { key: "PageDown" });
    expect(onPosition).toHaveBeenLastCalledWith([3, 3, 2]);
    fireEvent.keyDown(view, { key: "End" });
    expect(onPosition).toHaveBeenLastCalledWith([9, 3, 2]);
    fireEvent.change(screen.getByLabelText("axial slice"), { target: { value: "0" } });
    expect(onPosition).toHaveBeenLastCalledWith([0, 3, 2]);

    onPosition.mockClear();
    rerender(<SliceView image={image} plane="axial" position={[9, 3, 2]} onPosition={onPosition} level={0} width={100} />);
    fireEvent.keyDown(screen.getByLabelText(/axial view/i), { key: "ArrowUp" });
    expect(onPosition).not.toHaveBeenCalled();
  });

  it("shows the slice position in millimetres and the zoom", () => {
    render(<SliceView image={image} plane="axial" position={[4, 3, 2]} onPosition={() => {}} level={0} width={100} />);
    expect(screen.getByText("I 2.0 mm")).toBeInTheDocument(); // z = -10 + 4 x 2 mm (LPS), i.e. 2 mm inferior
    expect(screen.getByText(/^100%/)).toBeInTheDocument();
  });

  it("applies every move of a pan drag, so a later click lands on the panned voxel", () => {
    // 300 x 300 viewport, 6 x 8 mm axial slice: 37.5 px per voxel; a 60 px pan is 1.6 voxels.
    const onPosition = vi.fn();
    render(<SliceView image={image} plane="axial" position={[4, 3, 2]} onPosition={onPosition} level={0} width={100} />);
    const canvas = screen.getByTestId("slice-axial");
    fireEvent.pointerDown(canvas, { clientX: 150, clientY: 150, shiftKey: true, pointerId: 1 });
    fireEvent.pointerMove(canvas, { clientX: 180, clientY: 150, shiftKey: true, pointerId: 1 });
    fireEvent.pointerMove(canvas, { clientX: 210, clientY: 150, shiftKey: true, pointerId: 1 });
    fireEvent.pointerUp(canvas, { clientX: 210, clientY: 150, pointerId: 1 });
    expect(onPosition).not.toHaveBeenCalled();
    fireEvent.pointerDown(canvas, { clientX: 150, clientY: 150, pointerId: 2 });
    // Centre was x = 3.0 voxels; after panning 1.6 voxels right the screen centre shows x = 1.4.
    expect(onPosition).toHaveBeenLastCalledWith([4, 4, 1]);
  });

  it("extracts slice labels in display order for every plane", () => {
    const v: VolumeData = { data: new Uint8Array(4 * 3 * 2), shape: [4, 3, 2], spacing: [1, 1, 1], scale: 1, offset: 0 };
    v.data[3 * 6 + 1 * 2 + 1] = 7; // z=3 (top slice), y=1, x=1
    expect(sliceLabels(v, "axial", 3)[1 * 2 + 1]).toBe(7);
    expect(sliceLabels(v, "coronal", 1)[0 * 2 + 1]).toBe(7); // superior at the top row
    expect(sliceLabels(v, "sagittal", 1)[0 * 3 + 1]).toBe(7);
  });
});
