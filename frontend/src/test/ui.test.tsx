import { render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, fetchVolume, isUnavailable } from "../api/client";
import { EvidenceBadge, ErrorState, FindingList, UnavailableState } from "../components/ui";
import { planeInfo, pixelToVoxel, voxelIndex } from "../viewer/volume";

afterEach(() => vi.restoreAllMocks());

describe("error and unavailable states", () => {
  it("shows a backend-unreachable state for network errors", () => {
    render(<ErrorState error={new ApiError(0, "network_error", "The OncoNova API is not reachable.")} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Backend unreachable");
    expect(screen.getByRole("alert")).toHaveTextContent("not reachable");
  });

  it("shows the API error code and request id", () => {
    render(<ErrorState error={new ApiError(422, "unsupported_modality", "Needs CT.", "abc123")} />);
    expect(screen.getByRole("alert")).toHaveTextContent("unsupported_modality");
    expect(screen.getByRole("alert")).toHaveTextContent("abc123");
  });

  it("renders an explicit unavailable state instead of a prediction", () => {
    render(
      <UnavailableState
        data={{ status: "unavailable", kind: "unavailable", reason: "Needs two scans.", missing_inputs: ["second timepoint"], how_to_fix: "Import a follow-up." }}
      />,
    );
    const box = screen.getByRole("status");
    expect(box).toHaveTextContent("Needs two scans.");
    expect(box).toHaveTextContent("second timepoint");
    expect(box).toHaveTextContent("No estimate was produced");
  });

  it("labels evidence kinds distinctly", () => {
    render(
      <FindingList
        items={[
          { kind: "measured", text: "a" },
          { kind: "model_output", text: "b" },
          { kind: "requires_expert_review", text: "c" },
          { kind: "simulated", text: "d" },
        ]}
      />,
    );
    for (const label of ["Measured", "AI prediction", "Expert review", "Simulated"]) expect(screen.getByText(label)).toBeInTheDocument();
  });

  it("falls back to the unavailable label for unknown kinds", () => {
    // @ts-expect-error deliberately invalid kind
    render(<EvidenceBadge kind="bogus" />);
    expect(screen.getByText("Unavailable")).toBeInTheDocument();
  });
});

describe("api client", () => {
  it("turns fetch failures into ApiError(network_error)", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("Failed to fetch"));
    await expect(api.cases()).rejects.toMatchObject({ status: 0, code: "network_error" });
  });

  it("parses the server error envelope", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ error: { code: "case_not_found", message: "No case 'X'." }, request_id: "r1" }), { status: 404 }),
    );
    await expect(api.case("X")).rejects.toMatchObject({ status: 404, code: "case_not_found", requestId: "r1" });
  });

  it("rejects a volume whose payload does not match its shape", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(new Uint8Array(10), { status: 200, headers: { "X-Shape": "2,2,2", "X-Spacing": "1,1,1", "X-Dtype": "uint8" } }),
    );
    await expect(fetchVolume("/x")).rejects.toMatchObject({ code: "bad_volume" });
  });

  it("detects unavailable payloads", () => {
    expect(isUnavailable({ status: "unavailable" })).toBe(true);
    expect(isUnavailable({ status: "ok" })).toBe(false);
  });
});

describe("viewer geometry", () => {
  const shape: [number, number, number] = [4, 5, 6];
  it("uses physical millimetres for aspect ratio", () => {
    const info = planeInfo(shape, [0.5, 1, 3], "coronal");
    expect(info).toMatchObject({ width: 6, height: 4, widthMm: 3, heightMm: 12, count: 5 });
  });
  it("round-trips pixel -> voxel -> flat index on every plane", () => {
    for (const plane of ["axial", "coronal", "sagittal"] as const) {
      const [z, y, x] = pixelToVoxel(shape, plane, 2, 1, 3);
      expect(voxelIndex(shape, plane, 2, 1, 3)).toBe(z * 30 + y * 6 + x);
    }
  });
  it("puts superior at the top of coronal views", () => {
    expect(pixelToVoxel(shape, "coronal", 0, 0, 0)[0]).toBe(3);
  });
});
