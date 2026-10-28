export function parseCaAtoms(pdbText) {
  const atoms = [];
  for (const line of pdbText.split("\n")) {
    if (!line.startsWith("ATOM") || line.slice(12, 16).trim() !== "CA") {
      continue;
    }
    atoms.push({
      resSeq: Number.parseInt(line.slice(22, 26), 10),
      x: Number.parseFloat(line.slice(30, 38)),
      y: Number.parseFloat(line.slice(38, 46)),
      z: Number.parseFloat(line.slice(46, 54)),
      bFactor: Number.parseFloat(line.slice(60, 66)),
    });
  }
  return atoms;
}

export function distanceMatrix(atoms) {
  const count = atoms.length;
  const distances = new Float32Array(count * count);
  for (let row = 0; row < count; row += 1) {
    for (let column = row + 1; column < count; column += 1) {
      const distance = Math.hypot(atoms[row].x - atoms[column].x, atoms[row].y - atoms[column].y, atoms[row].z - atoms[column].z);
      distances[row * count + column] = distance;
      distances[column * count + row] = distance;
    }
  }
  return distances;
}

const ACCESSION = /^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$/;

/** Pull a UniProt accession out of a FASTA header such as "sp|P00698|LYSC_CHICK" or "AF-P00698-F1". */
export function extractAccession(header) {
  const text = String(header || "").trim();
  const candidates = [];

  const pipe = text.split("|");
  if (pipe.length >= 3 && ["sp", "tr"].includes(pipe[0].toLowerCase())) {
    candidates.push(pipe[1]);
  }
  const alphafold = /^AF-([A-Z0-9]+)-F\d+/.exec(text);
  if (alphafold) {
    candidates.push(alphafold[1]);
  }
  candidates.push(text.split(/[\s|]/)[0]);

  const found = candidates.map((candidate) => candidate.replace(/-\d+$/, "").toUpperCase()).find((candidate) => ACCESSION.test(candidate));
  return found || "";
}

export function isAccession(value) {
  return ACCESSION.test(String(value || "").trim().toUpperCase());
}

/** Offset of the input within the reference sequence, or null if the input is not a contiguous part of it. */
export function sequenceOffset(input, reference) {
  const index = reference.indexOf(input);
  return index === -1 ? null : index;
}
