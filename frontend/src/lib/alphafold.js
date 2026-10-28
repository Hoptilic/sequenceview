import { isAccession, parseCaAtoms, sequenceOffset } from "./pdb.js";

const API = "https://alphafold.ebi.ac.uk/api/prediction/";

/** Fetch the AlphaFold DB model for a UniProt accession. Only the accession leaves the browser. */
export async function fetchAlphaFoldModel(accession, inputSequence, fetcher = fetch) {
  const id = String(accession || "").trim().toUpperCase();
  if (!isAccession(id)) {
    throw new Error("Enter a UniProt accession such as P00698.");
  }

  const metadataResponse = await fetcher(`${API}${id}`);
  if (metadataResponse.status === 404) {
    throw new Error(`AlphaFold DB has no model for ${id}.`);
  }
  if (!metadataResponse.ok) {
    throw new Error(`AlphaFold DB request failed (${metadataResponse.status}).`);
  }

  const entries = await metadataResponse.json();
  const entry = Array.isArray(entries) ? entries[0] : null;
  if (!entry?.pdbUrl) {
    throw new Error(`AlphaFold DB returned no structure file for ${id}.`);
  }

  const pdbResponse = await fetcher(entry.pdbUrl);
  if (!pdbResponse.ok) {
    throw new Error(`Could not download the structure file (${pdbResponse.status}).`);
  }
  const pdb = await pdbResponse.text();

  const reference = entry.uniprotSequence || entry.sequence || "";
  return {
    accession: id,
    entryId: entry.entryId || entry.modelEntityId || id,
    description: entry.uniprotDescription || "",
    organism: entry.organismScientificName || "",
    meanPlddt: entry.globalMetricValue ?? null,
    version: entry.latestVersion ?? null,
    pdb,
    atoms: parseCaAtoms(pdb),
    referenceSequence: reference,
    offset: sequenceOffset(String(inputSequence || ""), reference),
  };
}
