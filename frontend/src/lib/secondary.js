// Chou-Fasman conformational propensities (helix, sheet), per residue.
const PROPENSITY = {
  A: [142, 83], R: [98, 93], N: [67, 89], D: [101, 54], C: [70, 119],
  E: [151, 37], Q: [111, 110], G: [57, 75], H: [100, 87], I: [108, 160],
  L: [121, 130], K: [114, 74], M: [145, 105], F: [113, 138], P: [57, 55],
  S: [77, 75], T: [83, 119], W: [108, 137], Y: [69, 147], V: [106, 170],
};
const NEUTRAL = [100, 100];

function windowMean(values, centre, size) {
  const half = Math.floor(size / 2);
  const low = Math.max(0, centre - half);
  const high = Math.min(values.length, centre + half + 1);
  let total = 0;
  for (let index = low; index < high; index += 1) {
    total += values[index];
  }
  return total / (high - low);
}

/**
 * Per-residue helix (H), strand (E) or coil (C) call from windowed Chou-Fasman propensities.
 * This is an estimate from composition alone, not a structure prediction.
 */
export function estimateSecondaryStructure(sequence) {
  const helix = Array.from(sequence, (acid) => (PROPENSITY[acid] || NEUTRAL)[0]);
  const sheet = Array.from(sequence, (acid) => (PROPENSITY[acid] || NEUTRAL)[1]);

  return Array.from(sequence, (_, index) => {
    const helixScore = windowMean(helix, index, 6);
    const sheetScore = windowMean(sheet, index, 5);
    if (helixScore > 103 && helixScore >= sheetScore) {
      return "H";
    }
    if (sheetScore > 105 && sheetScore > helixScore) {
      return "E";
    }
    return "C";
  });
}

export function secondarySegments(states, minLength = 3) {
  const segments = [];
  let start = 0;
  for (let index = 1; index <= states.length; index += 1) {
    if (index === states.length || states[index] !== states[start]) {
      const state = states[start];
      const long = index - start >= minLength;
      segments.push({ start, end: index, state: state !== "C" && !long ? "C" : state });
      start = index;
    }
  }

  const merged = [];
  segments.forEach((segment) => {
    const last = merged[merged.length - 1];
    if (last && last.state === segment.state) {
      last.end = segment.end;
    } else {
      merged.push({ ...segment });
    }
  });
  return merged;
}

// Eisenberg consensus hydrophobicity scale.
export const EISENBERG = {
  A: 0.62, R: -2.53, N: -0.78, D: -0.9, C: 0.29, Q: -0.85, E: -0.74, G: 0.48, H: -0.4, I: 1.38,
  L: 1.06, K: -1.5, M: 0.64, F: 1.19, P: 0.12, S: -0.18, T: -0.05, W: 0.81, Y: 0.26, V: 1.08,
};

/** Positions of a window of residues on a helical wheel (100 degrees per residue) plus hydrophobic moment. */
export function helicalWheel(sequence, start, length, degreesPerResidue = 100) {
  const residues = sequence.slice(start, start + length).split("");
  let x = 0;
  let y = 0;
  let counted = 0;

  const points = residues.map((acid, offset) => {
    const angle = (offset * degreesPerResidue * Math.PI) / 180;
    const hydrophobicity = EISENBERG[acid];
    if (hydrophobicity !== undefined) {
      x += hydrophobicity * Math.cos(angle);
      y += hydrophobicity * Math.sin(angle);
      counted += 1;
    }
    return { acid, position: start + offset, angle, hydrophobicity };
  });

  return {
    points,
    moment: counted ? Math.hypot(x, y) / counted : 0,
    momentAngle: Math.atan2(y, x),
  };
}
