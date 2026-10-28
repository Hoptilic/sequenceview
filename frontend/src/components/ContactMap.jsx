import { useEffect, useMemo, useRef, useState } from "react";

import { distanceMatrix } from "../lib/pdb.js";

const CUTOFF = 8;

function mixColor(base, target, amount) {
  return base.map((channel, index) => Math.round(channel + (target[index] - channel) * amount));
}

export default function ContactMap({ atoms, scores, offset }) {
  const canvasRef = useRef(null);
  const [hover, setHover] = useState("");

  const distances = useMemo(() => distanceMatrix(atoms), [atoms]);
  const count = atoms.length;

  const perAtom = useMemo(
    () => atoms.map((atom) => {
      const index = atom.resSeq - 1 - (offset ?? 0);
      return scores && offset !== null && index >= 0 && index < scores.length ? scores[index] : 0;
    }),
    [atoms, scores, offset],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    canvas.width = count;
    canvas.height = count;
    const context = canvas.getContext("2d");
    const image = context.createImageData(count, count);

    const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const paper = dark ? [31, 31, 31] : [244, 244, 244];
    const ink = dark ? [200, 200, 200] : [40, 40, 40];
    const positive = dark ? [111, 163, 245] : [26, 79, 160];
    const negative = dark ? [240, 144, 74] : [194, 87, 26];

    for (let row = 0; row < count; row += 1) {
      for (let column = 0; column < count; column += 1) {
        const contact = distances[row * count + column] < CUTOFF && row !== column;
        let colour = paper;
        if (row > column) {
          colour = contact ? ink : paper;
        } else if (row < column && contact) {
          const weight = Math.sqrt(Math.abs(perAtom[row]) * Math.abs(perAtom[column]));
          const direction = perAtom[row] + perAtom[column] >= 0 ? positive : negative;
          colour = mixColor(paper, direction, Math.min(1, weight * 1.4));
        } else if (row === column) {
          colour = ink;
        }
        const offsetIndex = (row * count + column) * 4;
        image.data[offsetIndex] = colour[0];
        image.data[offsetIndex + 1] = colour[1];
        image.data[offsetIndex + 2] = colour[2];
        image.data[offsetIndex + 3] = 255;
      }
    }
    context.putImageData(image, 0, 0);
  }, [distances, perAtom, count]);

  function handleMove(event) {
    const rectangle = event.currentTarget.getBoundingClientRect();
    const column = Math.floor(((event.clientX - rectangle.left) / rectangle.width) * count);
    const row = Math.floor(((event.clientY - rectangle.top) / rectangle.height) * count);
    if (row < 0 || column < 0 || row >= count || column >= count) {
      setHover("");
      return;
    }
    setHover(`Residues ${atoms[row].resSeq} and ${atoms[column].resSeq}: ${distances[row * count + column].toFixed(1)} Å between C-alpha atoms`);
  }

  return (
    <div className="contact">
      <canvas ref={canvasRef} className="contact-canvas" onMouseMove={handleMove} onMouseLeave={() => setHover("")} role="img" aria-label="Contact map" />
      <p className="note">
        Lower triangle: C-alpha pairs within {CUTOFF} {"Å"}. Upper triangle: the same contacts shaded by the geometric mean
        of the two residues' attribution magnitudes (blue when they jointly support the predicted class, orange when they
        oppose it). Dark blocks mean the model's sensitivity concentrates in residues that are close in space.
      </p>
      <p className="meta contact-hover">{hover || "Hover the map to read a residue pair."}</p>
    </div>
  );
}
