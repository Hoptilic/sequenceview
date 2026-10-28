import { useEffect, useMemo, useRef, useState } from "react";

import ContactMap from "./ContactMap.jsx";
import { fetchAlphaFoldModel } from "../lib/alphafold.js";
import { extractAccession } from "../lib/pdb.js";
import { residueGroup } from "../lib/residues.js";

const KYTE_DOOLITTLE = {
  A: 1.8, R: -4.5, N: -3.5, D: -3.5, C: 2.5, Q: -3.5, E: -3.5, G: -0.4, H: -3.2, I: 4.5,
  L: 3.8, K: -3.9, M: 1.9, F: 2.8, P: -1.6, S: -0.8, T: -0.7, W: -0.9, Y: -1.3, V: 4.2,
};
const CLASS_COLOURS = { nonpolar: "#8a8a8a", polar: "#1f7a5a", positive: "#1a4fa0", negative: "#b3261e", unknown: "#bbbbbb" };

function blend(from, to, amount) {
  const mix = (index) => Math.round(from[index] + (to[index] - from[index]) * amount);
  return (mix(0) << 16) | (mix(1) << 8) | mix(2);
}

function plddtColour(value) {
  if (value > 90) return 0x0053d6;
  if (value > 70) return 0x65cbf3;
  if (value > 50) return 0xffdb13;
  return 0xff7d45;
}

function cssToHex(colour) {
  const match = /rgba?\((\d+),\s*(\d+),\s*(\d+)/.exec(colour);
  return match ? `#${[1, 2, 3].map((index) => Number(match[index]).toString(16).padStart(2, "0")).join("")}` : "#ffffff";
}

const MODES = [
  { id: "attribution", label: "Attribution" },
  { id: "plddt", label: "AlphaFold confidence (pLDDT)" },
  { id: "hydropathy", label: "Hydropathy" },
  { id: "class", label: "Residue class" },
  { id: "plain", label: "Plain" },
];

export default function Structure({ header, sequence, scores, focus, highlights }) {
  const [accession, setAccession] = useState(() => extractAccession(header));
  const [state, setState] = useState({ status: "idle", error: "", model: null });
  const [mode, setMode] = useState("plddt");
  const containerRef = useRef(null);
  const viewerRef = useRef(null);
  const libraryRef = useRef(null);

  const { model } = state;
  const mapped = model && model.offset !== null;

  async function load() {
    setState({ status: "loading", error: "", model: null });
    try {
      const loaded = await fetchAlphaFoldModel(accession, sequence);
      setState({ status: "ready", error: "", model: loaded });
      setMode(loaded.offset !== null && scores ? "attribution" : "plddt");
    } catch (error) {
      setState({ status: "error", error: error.message, model: null });
    }
  }

  useEffect(() => {
    if (!model || !containerRef.current) {
      return undefined;
    }

    let cancelled = false;
    (async () => {
      try {
        const module = await import("3dmol");
        const library = module.createViewer ? module : module.default;
        if (cancelled || !containerRef.current) {
          return;
        }
        libraryRef.current = library;
        const background = cssToHex(getComputedStyle(document.body).backgroundColor);
        const viewer = library.createViewer(containerRef.current, { backgroundColor: background });
        viewer.addModel(model.pdb, "pdb");
        viewerRef.current = viewer;
        applyStyle();
        viewer.zoomTo();
        viewer.render();
      } catch {
        setState((previous) => ({ ...previous, error: "This browser could not start the 3D viewer (WebGL is required)." }));
      }
    })();

    return () => {
      cancelled = true;
      viewerRef.current?.clear?.();
      viewerRef.current = null;
    };
    // The viewer is rebuilt only when a new model is loaded; styling updates are handled below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model]);

  const colourFor = useMemo(() => {
    const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const grey = dark ? [90, 90, 90] : [214, 214, 214];
    const positive = dark ? [111, 163, 245] : [26, 79, 160];
    const negative = dark ? [240, 144, 74] : [194, 87, 26];
    const hydrophilic = [26, 79, 160];
    const hydrophobic = [194, 87, 26];

    return (atom) => {
      const index = atom.resi - 1 - (model?.offset ?? 0);
      const acid = sequence[index];
      if (mode === "plddt") {
        return plddtColour(atom.b);
      }
      if (!mapped || index < 0 || index >= sequence.length) {
        return mode === "plain" ? 0xbbbbbb : blend(grey, grey, 0);
      }
      if (mode === "attribution") {
        const score = scores?.[index];
        return score === undefined ? blend(grey, grey, 0) : blend(grey, score >= 0 ? positive : negative, Math.min(1, Math.abs(score)));
      }
      if (mode === "hydropathy") {
        const value = (KYTE_DOOLITTLE[acid] ?? 0) / 4.5;
        return blend([235, 235, 235], value >= 0 ? hydrophobic : hydrophilic, Math.abs(value));
      }
      if (mode === "class") {
        return CLASS_COLOURS[residueGroup(acid)];
      }
      return 0xbbbbbb;
    };
  }, [mode, mapped, model, scores, sequence]);

  function applyStyle() {
    const viewer = viewerRef.current;
    if (!viewer || !model) {
      return;
    }

    viewer.setStyle({}, { cartoon: { colorfunc: colourFor } });
    const offset = model.offset ?? 0;
    const residues = new Set();
    highlights.forEach(({ start, end }) => {
      for (let index = start; index < end; index += 1) {
        residues.add(index + 1 + offset);
      }
    });
    if (focus !== null && focus !== undefined) {
      residues.add(focus + 1 + offset);
    }
    if (mapped && residues.size) {
      viewer.addStyle({ resi: [...residues] }, { stick: { radius: 0.28, colorfunc: () => 0xe0457b } });
    }
    viewer.render();
  }

  useEffect(applyStyle, [colourFor, focus, highlights, model, mapped]);

  return (
    <div>
      <div className="structure-load">
        <label htmlFor="accessionInput">UniProt accession</label>
        <input
          id="accessionInput"
          type="text"
          value={accession}
          onChange={(event) => setAccession(event.target.value.trim().toUpperCase())}
          placeholder="P00698"
          spellCheck={false}
        />
        <button type="button" className="button" onClick={load} disabled={state.status === "loading" || !accession}>
          {state.status === "loading" ? "Loading..." : "Load AlphaFold model"}
        </button>
      </div>
      <p className="note">
        Only the accession is sent, to the public AlphaFold Database. The structure is a prediction made for the reference
        UniProt sequence, not an experimental structure of your input.
      </p>
      {state.error ? <p className="error">{state.error}</p> : null}

      {model ? (
        <div>
          <p className="meta">
            {model.entryId}
            {model.description ? ` | ${model.description}` : ""}
            {model.organism ? ` | ${model.organism}` : ""}
            {model.meanPlddt !== null ? ` | mean pLDDT ${model.meanPlddt}` : ""}
          </p>
          {!mapped ? (
            <p className="notice">
              The input is not a contiguous part of the AlphaFold reference sequence, so residues cannot be matched. Only the
              pLDDT colouring is available.
            </p>
          ) : null}

          <div className="structure-modes">
            {MODES.map((item) => (
              <button
                key={item.id}
                type="button"
                className={`link${mode === item.id ? " active" : ""}`}
                disabled={(item.id === "attribution" && !(scores && mapped)) || ((item.id === "hydropathy" || item.id === "class") && !mapped)}
                onClick={() => setMode(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>

          <div ref={containerRef} className="structure-view" />
          {mode === "plddt" ? (
            <div className="legend">
              <span style={{ color: "#0053d6" }}>above 90</span>
              <span style={{ color: "#3aa8d8" }}>70 to 90</span>
              <span style={{ color: "#b89b00" }}>50 to 70</span>
              <span style={{ color: "#ff7d45" }}>below 50</span>
            </div>
          ) : null}
          {mapped && (focus !== null || highlights.length) ? <p className="note">Selected residues are drawn as pink sticks.</p> : null}

          <h3 className="subheading">Contact map</h3>
          <ContactMap atoms={model.atoms} scores={scores} offset={model.offset} />
        </div>
      ) : null}
    </div>
  );
}
