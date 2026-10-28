import { describe, expect, it } from "vitest";

import { fetchAlphaFoldModel } from "./alphafold.js";
import { distanceMatrix, extractAccession, isAccession, parseCaAtoms, sequenceOffset } from "./pdb.js";
import { estimateSecondaryStructure, helicalWheel, secondarySegments } from "./secondary.js";

const PDB = [
  "ATOM      1  N   MET A   1      10.000  10.000  10.000  1.00 95.12           N",
  "ATOM      2  CA  MET A   1      11.000  10.000  10.000  1.00 95.12           C",
  "ATOM      3  CA  LYS A   2      14.000  14.000  10.000  1.00 88.40           C",
  "ATOM      4  CB  LYS A   2      15.000  14.000  10.000  1.00 88.40           C",
  "ATOM      5  CA  VAL A   3      11.000  10.000  22.000  1.00 45.00           C",
  "HETATM    6  CA  CA  A 100       0.000   0.000   0.000  1.00 10.00          CA",
].join("\n");

describe("pdb parsing", () => {
  it("reads only C-alpha ATOM records", () => {
    const atoms = parseCaAtoms(PDB);
    expect(atoms.map((atom) => atom.resSeq)).toEqual([1, 2, 3]);
    expect(atoms[1]).toMatchObject({ x: 14, y: 14, z: 10, bFactor: 88.4 });
  });

  it("builds a symmetric distance matrix", () => {
    const atoms = parseCaAtoms(PDB);
    const distances = distanceMatrix(atoms);
    expect(distances[0 * 3 + 1]).toBeCloseTo(5, 5);
    expect(distances[1 * 3 + 0]).toBeCloseTo(5, 5);
    expect(distances[0 * 3 + 2]).toBeCloseTo(12, 5);
    expect(distances[1 * 3 + 1]).toBe(0);
  });
});

describe("accessions", () => {
  it("extracts accessions from common header styles", () => {
    expect(extractAccession("sp|P00698|LYSC_CHICK Lysozyme C")).toBe("P00698");
    expect(extractAccession("tr|A0A024R161|A0A024R161_HUMAN foo")).toBe("A0A024R161");
    expect(extractAccession("P69905")).toBe("P69905");
    expect(extractAccession("AF-P00698-F1-model_v4")).toBe("P00698");
    expect(extractAccession("P04637-2 isoform")).toBe("P04637");
  });

  it("returns an empty string for headers without one", () => {
    expect(extractAccession("my_protein_1")).toBe("");
    expect(extractAccession("")).toBe("");
    expect(isAccession("lysozyme")).toBe(false);
    expect(isAccession(" p00698 ")).toBe(true);
  });

  it("locates an input inside the reference sequence", () => {
    expect(sequenceOffset("KVFG", "MRSLKVFGRC")).toBe(4);
    expect(sequenceOffset("KVFG", "KVFG")).toBe(0);
    expect(sequenceOffset("AAAA", "MRSLKVFGRC")).toBeNull();
  });
});

describe("alphafold client", () => {
  const response = (body, init = {}) => ({
    ok: init.ok ?? true,
    status: init.status ?? 200,
    json: async () => body,
    text: async () => body,
  });

  it("fetches metadata then the structure and maps the offset", async () => {
    const calls = [];
    const fetcher = async (url) => {
      calls.push(url);
      if (url.includes("/api/prediction/")) {
        return response([{ entryId: "AF-P00698-F1", pdbUrl: "https://example.test/model.pdb", uniprotSequence: "MRSLKVFGRC", globalMetricValue: 93.9, latestVersion: 6 }]);
      }
      return response(PDB);
    };

    const model = await fetchAlphaFoldModel("p00698", "KVFG", fetcher);
    expect(calls[0]).toBe("https://alphafold.ebi.ac.uk/api/prediction/P00698");
    expect(model.offset).toBe(4);
    expect(model.atoms).toHaveLength(3);
    expect(model.meanPlddt).toBe(93.9);
  });

  it("reports missing models and bad accessions clearly", async () => {
    await expect(fetchAlphaFoldModel("nope", "KV")).rejects.toThrow("UniProt accession");
    const missing = async () => response(null, { ok: false, status: 404 });
    await expect(fetchAlphaFoldModel("P00698", "KV", missing)).rejects.toThrow("no model");
  });
});

describe("secondary structure estimate", () => {
  it("calls an alanine/glutamate run helix and a valine/isoleucine run strand", () => {
    expect(new Set(estimateSecondaryStructure("AEAEAEAEAEAEAEAE"))).toEqual(new Set(["H"]));
    expect(new Set(estimateSecondaryStructure("VIVIVIVIVIVIVIVI"))).toEqual(new Set(["E"]));
    expect(new Set(estimateSecondaryStructure("GPGPGPGPGPGPGPGP"))).toEqual(new Set(["C"]));
  });

  it("merges states into segments and drops very short elements", () => {
    const segments = secondarySegments(["C", "H", "C", "C", "H", "H", "H", "H", "C"], 3);
    expect(segments).toEqual([
      { start: 0, end: 4, state: "C" },
      { start: 4, end: 8, state: "H" },
      { start: 8, end: 9, state: "C" },
    ]);
  });
});

describe("helical wheel", () => {
  it("places residues 100 degrees apart", () => {
    const wheel = helicalWheel("AAAAAAAA", 2, 4);
    expect(wheel.points.map((point) => point.position)).toEqual([2, 3, 4, 5]);
    expect(wheel.points[1].angle - wheel.points[0].angle).toBeCloseTo((100 * Math.PI) / 180, 8);
  });

  it("gives a large hydrophobic moment for an amphipathic helix and a small one for a uniform one", () => {
    const amphipathic = "LKKLLKLLKKLLKLLKKL";
    const uniform = "AAAAAAAAAAAAAAAAAA";
    expect(helicalWheel(amphipathic, 0, 18).moment).toBeGreaterThan(helicalWheel(uniform, 0, 18).moment * 2);
  });

  it("handles windows that run past the end", () => {
    expect(helicalWheel("MKV", 1, 10).points).toHaveLength(2);
  });
});
