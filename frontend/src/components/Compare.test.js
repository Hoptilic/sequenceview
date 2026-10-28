import { describe, expect, it } from "vitest";

import { alignmentLines } from "./Compare.jsx";

describe("alignmentLines", () => {
  it("marks identities, substitutions and gaps and tracks residue numbers across gaps", () => {
    const [line] = alignmentLines("MKV-LA", "MKVALG", 60);
    expect(line.marks).toBe("||| |.");
    expect(line.startA).toBe(1);
    expect(line.startB).toBe(1);
  });

  it("starts later lines after the residues already consumed, ignoring gap characters", () => {
    const lines = alignmentLines("AB-CD", "ABECD", 3);
    expect(lines).toHaveLength(2);
    expect(lines[1].startA).toBe(3);
    expect(lines[1].startB).toBe(4);
  });
});
