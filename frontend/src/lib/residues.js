export const AMINO_ACID_ORDER = "ACDEFGHIKLMNPQRSTVWY".split("");

export const RESIDUE_GROUPS = {
  nonpolar: "AVLIMFWPG",
  polar: "STCYNQ",
  positive: "KRH",
  negative: "DE",
};

export const GROUP_LABELS = {
  nonpolar: "Nonpolar",
  polar: "Polar",
  positive: "Basic",
  negative: "Acidic",
};

export function residueGroup(acid) {
  const group = Object.keys(RESIDUE_GROUPS).find((name) => RESIDUE_GROUPS[name].includes(acid));
  return group || "unknown";
}

export function normalizeFrequencies(frequencyMap) {
  return AMINO_ACID_ORDER.map((acid) => ({ acid, value: Number(frequencyMap?.[acid] || 0) }));
}

export function chunkSequence(sequence, blockSize = 10, blocksPerLine = 6) {
  const lineSize = blockSize * blocksPerLine;
  const lines = [];
  for (let start = 0; start < sequence.length; start += lineSize) {
    const line = sequence.slice(start, start + lineSize);
    const blocks = [];
    for (let offset = 0; offset < line.length; offset += blockSize) {
      blocks.push({ start: start + offset, text: line.slice(offset, offset + blockSize) });
    }
    lines.push({ start, blocks });
  }
  return lines;
}
