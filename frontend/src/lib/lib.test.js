import { describe, expect, it } from "vitest";

import { parseFastaRecords } from "./fasta.js";
import { csvValue, escapeHtml, toCsv } from "./format.js";
import { findMotifs, motifMask, prositeToRegex } from "./motif.js";
import { buildBatchCsv, buildHtmlReport, buildReportCsv } from "./report.js";
import { chunkSequence, residueGroup } from "./residues.js";
import { decodeSequenceHash, encodeSequenceHash } from "./share.js";

describe("fasta", () => {
  it("parses plain text as one record", () => {
    expect(parseFastaRecords("mkv\nlla")).toEqual([{ id: "sequence", sequence: "MKVLLA" }]);
  });

  it("parses several records and skips empty ones", () => {
    const records = parseFastaRecords(">a one\nMK\nV\n>empty\n>b\r\nGHK\r\n");
    expect(records).toEqual([
      { id: "a one", sequence: "MKV" },
      { id: "b", sequence: "GHK" },
    ]);
  });
});

describe("motifs", () => {
  it("converts PROSITE syntax", () => {
    expect(prositeToRegex("C-x(2,4)-C-x(3)-[LIVMFYWC]-x(8)-H-x(3,5)-H.")).toBe("C.{2,4}C.{3}[LIVMFYWC].{8}H.{3,5}H");
    expect(prositeToRegex("<M-{P}-x")).toBe("^M[^P].");
  });

  it("finds overlapping matches with 0-based ranges", () => {
    const { matches } = findMotifs("AAAA", "AA");
    expect(matches).toEqual([
      { start: 0, end: 2 },
      { start: 1, end: 3 },
      { start: 2, end: 4 },
    ]);
  });

  it("accepts plain regular expressions and reports bad ones", () => {
    expect(findMotifs("MKCAACKT", "C..C").matches).toEqual([{ start: 2, end: 6 }]);
    expect(findMotifs("MKV", "[").error).toBe("Invalid pattern.");
    expect(findMotifs("MKV", "  ")).toEqual({ matches: [], error: "" });
  });

  it("builds a mask", () => {
    expect(Array.from(motifMask(5, [{ start: 1, end: 3 }]))).toEqual([0, 1, 1, 0, 0]);
  });
});

describe("residues", () => {
  it("classifies residues", () => {
    expect(residueGroup("K")).toBe("positive");
    expect(residueGroup("D")).toBe("negative");
    expect(residueGroup("X")).toBe("unknown");
  });

  it("chunks sequences into numbered lines of blocks", () => {
    const lines = chunkSequence("A".repeat(65));
    expect(lines.map((line) => line.start)).toEqual([0, 60]);
    expect(lines[0].blocks).toHaveLength(6);
    expect(lines[1].blocks.map((block) => block.text.length)).toEqual([5]);
    expect(lines[1].blocks[0].start).toBe(60);
  });
});

describe("share links", () => {
  it("round-trips text including multibyte characters", () => {
    const text = ">id ü\nMKVLAAGHK";
    expect(decodeSequenceHash(`#${encodeSequenceHash(text)}`)).toBe(text);
  });

  it("refuses oversized input and ignores bad hashes", () => {
    expect(encodeSequenceHash("A".repeat(7000))).toBeNull();
    expect(decodeSequenceHash("#seq=***")).toBe("");
    expect(decodeSequenceHash("")).toBe("");
  });
});

describe("exports", () => {
  it("quotes csv values", () => {
    expect(csvValue('a,"b"')).toBe('"a,""b"""');
    expect(toCsv([["x", 1], []])).toBe("x,1\n");
  });

  it("escapes html", () => {
    expect(escapeHtml('<a href="x">&</a>')).toBe("&lt;a href=&quot;x&quot;&gt;&amp;&lt;/a&gt;");
  });

  const result = {
    input: { id: "<lyz>" },
    model: { version: "abc123", checkpoint_sha256: "f".repeat(64) },
    prediction: {
      class_name: "hydrolase",
      predicted_class: "hydrolase",
      confidence: 0.9,
      confidence_threshold: 0.7,
      calibrated: true,
      temperature: 1.5,
      probabilities: { hydrolase: 0.9, oxidoreductase: 0.1 },
    },
    analysis: {
      length: 12,
      normalized_sequence: "MKVLAAGHKTAA",
      prediction_sequence: "MKVLAAGHKTAA",
      amino_acid_frequency: { A: 0.5 },
      secondary_structure_fraction: { helix: 0.3, turn: 0.2, sheet: 0.1 },
    },
    warnings: [{ code: "short_sequence", message: "Sequence has 12 residues." }],
  };

  it("builds the single-record csv", () => {
    const csv = buildReportCsv(result);
    expect(csv).toContain("model_version,abc123");
    expect(csv).toContain("warnings,short_sequence");
    expect(csv).toContain("A,0.5");
  });

  it("builds the batch csv with a row per record", () => {
    const csv = buildBatchCsv([
      { index: 1, id: "a", class_name: "hydrolase", probabilities: { hydrolase: 0.8, oxidoreductase: 0.2 } },
      { index: 2, id: "b", error: "bad" },
    ]);
    const lines = csv.split("\n");
    expect(lines).toHaveLength(3);
    expect(lines[2].endsWith(",bad")).toBe(true);
  });

  it("builds a standalone html report with escaped content", () => {
    const html = buildHtmlReport(result, new Date("2026-01-01T00:00:00Z"));
    expect(html).toContain("&lt;lyz&gt;");
    expect(html).not.toContain("<lyz>");
    expect(html).toContain("2026-01-01T00:00:00.000Z");
    expect(html).toContain("abc123");
    expect(html).toContain("MKVLAAGHKT AA");
  });
});
