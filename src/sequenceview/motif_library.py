from __future__ import annotations

import re
from typing import Any

HYDROLASE = "hydrolase"
OXIDOREDUCTASE = "oxidoreductase"

# Patterns copied from the PROSITE entries named in the id field (https://prosite.expasy.org).
# The two entries without a PROSITE id are common structural heuristics, not curated signatures.
MOTIFS: list[dict[str, str]] = [
    {"id": "PS00120", "name": "Lipase active-site serine", "pattern": "[LIV]-{KG}-[LIVFY]-[LIVMST]-G-[HYWV]-S-{YAG}-G-[GSTAC]", "class": HYDROLASE},
    {"id": "PS00122", "name": "Type-B carboxylesterase serine", "pattern": "F-[GR]-G-x(4)-[LIVM]-x-[LIV]-x-G-x-S-[STAG]-G", "class": HYDROLASE},
    {"id": "PS00134", "name": "Trypsin-family histidine", "pattern": "[LIVM]-[ST]-A-[STAG]-H-C", "class": HYDROLASE},
    {"id": "PS00135", "name": "Trypsin-family serine", "pattern": "[DNSTAGC]-[GSTAPIMVQH]-x(2)-G-[DE]-S-G-[GS]-[SAPHV]-[LIVMFYWH]-[LIVMFYSTANQH]", "class": HYDROLASE},
    {"id": "PS00137", "name": "Subtilase histidine", "pattern": "H-G-[STM]-x-[VIC]-[STAGC]-[GS]-x-[LIVMA]-[STAGCLV]-[SAGM]", "class": HYDROLASE},
    {"id": "PS00138", "name": "Subtilase serine", "pattern": "G-T-S-x-[SA]-x-P-x-{L}-[STAVC]-[AG]", "class": HYDROLASE},
    {"id": "PS00139", "name": "Thiol protease cysteine", "pattern": "Q-{V}-x-{DE}-[GE]-{F}-C-[YW]-{DN}-x-[STAGC]-[STAGCV]", "class": HYDROLASE},
    {"id": "PS00141", "name": "Aspartyl protease active site", "pattern": "[LIVMFGAC]-[LIVMTADN]-[LIVFSA]-D-[ST]-G-[STAV]-[STAPDENQ]-{GQ}-[LIVMFSTNC]-{EGK}-[LIVMFGTA]", "class": HYDROLASE},
    {"id": "PS00142", "name": "Zinc protease HExxH", "pattern": "[GSTALIVN]-{PCHR}-{KND}-H-E-[LIVMFYW]-{DEHRKP}-H-{EKPC}-[LIVMFYWGSPQ]", "class": HYDROLASE},
    {"id": "PS00123", "name": "Alkaline phosphatase active site", "pattern": "[IV]-x-D-S-[GAS]-[GASC]-[GAST]-[GA]-T", "class": HYDROLASE},
    {"id": "PS00572", "name": "Glycosyl hydrolase family 1 active site", "pattern": "[LIVMFSTC]-[LIVFYS]-[LIV]-[LIVMST]-E-N-G-[LIVMFAR]-[CSAGN]", "class": HYDROLASE},
    {"id": "PS00133", "name": "Zinc carboxypeptidase zinc-binding", "pattern": "H-[STAG]-{ADNV}-{VGFI}-{YAR}-[LIVME]-{SDEP}-x-[LIVMFYW]-P-[FYW]", "class": HYDROLASE},
    {"id": "PS00143", "name": "Insulinase zinc-binding", "pattern": "G-x(8,9)-G-x-[STA]-H-[LIVMFY]-[LIVMC]-[DERN]-[HRKL]-[LMFAT]-x-[LFSTH]-x-[GSTAN]-[GST]", "class": HYDROLASE},
    {"id": "", "name": "GxSxG nucleophile elbow (heuristic)", "pattern": "G-x-S-x-G", "class": HYDROLASE},
    {"id": "PS00059", "name": "Zinc alcohol dehydrogenase", "pattern": "G-H-E-x-{EL}-G-{AP}-x(4)-[GA]-x(2)-[IVSAC]", "class": OXIDOREDUCTASE},
    {"id": "PS00060", "name": "Iron-containing alcohol dehydrogenase", "pattern": "[GSW]-x-[LIVTSACD]-[GH]-x(2)-[GSAE]-[GSHYQ]-x-[LIVTP]-[GAST]-[GAS]-x(3)-[LIVMT]-x-[HNS]-[GA]-x-[GTAC]", "class": OXIDOREDUCTASE},
    {"id": "PS00061", "name": "Short-chain dehydrogenase/reductase", "pattern": "[LIVSPADNK]-x(9)-{P}-x(2)-Y-[PSTAGNCV]-[STAGNQCIVM]-[STAGC]-K-{PC}-[SAGFYR]-[LIVMSTAGD]-x-{K}-[LIVMFYW]-{D}-x-{YR}-[LIVMFYWGAPTHQ]-[GSACQRHM]", "class": OXIDOREDUCTASE},
    {"id": "PS00070", "name": "Aldehyde dehydrogenase cysteine", "pattern": "[FYLVA]-x-{GVEP}-{DILV}-G-[QE]-{LPYG}-C-[LIVMGSTANC]-[AGCN]-{HE}-[GSTADNEKR]", "class": OXIDOREDUCTASE},
    {"id": "PS00194", "name": "Thioredoxin active site", "pattern": "[LIVMF]-[LIVMSTA]-x-[LIVMFYC]-[FYWSTHE]-x(2)-[FYWGTN]-C-[GATPLVE]-[PHYWSTA]-C-{I}-x-{A}-x(3)-[LIVMFYWT]", "class": OXIDOREDUCTASE},
    {"id": "PS00435", "name": "Peroxidase proximal heme ligand", "pattern": "[DET]-[LIVMTA]-{NSYL}-{RPFC}-[LIVM]-[LIVMSTAG]-[SAG]-[LIVMSTAG]-H-[STA]-[LIVMFY]", "class": OXIDOREDUCTASE},
    {"id": "PS00436", "name": "Peroxidase active site", "pattern": "[SGATV]-{D}-x(2)-[LIVMA]-R-[LIVMA]-x-[FW]-H-{V}-[SAC]", "class": OXIDOREDUCTASE},
    {"id": "PS00437", "name": "Catalase proximal heme ligand", "pattern": "R-[LIVMFSTAN]-F-[GASTNP]-Y-x-D-[AST]-[QEH]", "class": OXIDOREDUCTASE},
    {"id": "", "name": "GxGxxG dinucleotide-binding fold (heuristic)", "pattern": "G-x-G-x-x-G", "class": OXIDOREDUCTASE},
]


def prosite_to_regex(pattern: str) -> str:
    text = pattern.strip().rstrip(".")
    text = re.sub(r"\{([A-Z]+)\}", r"[^\1]", text)
    text = re.sub(r"\((\d+)(?:,(\d+))?\)", lambda match: f"{{{match.group(1)},{match.group(2)}}}" if match.group(2) else f"{{{match.group(1)}}}", text)
    text = text.replace("x", ".").replace("-", "").replace("<", "^").replace(">", "$")
    return text


_COMPILED = [(motif, re.compile(f"(?=({prosite_to_regex(motif['pattern'])}))")) for motif in MOTIFS]


def scan_motifs(sequence: str) -> list[dict[str, Any]]:
    """Return every library motif found in the sequence, with 0-based half-open ranges."""
    hits = []
    for motif, expression in _COMPILED:
        matches = [
            {"start": match.start(1), "end": match.end(1), "match": match.group(1)}
            for match in expression.finditer(sequence)
        ]
        if matches:
            hits.append({
                "id": motif["id"],
                "name": motif["name"],
                "pattern": motif["pattern"],
                "class": motif["class"],
                "source": f"PROSITE {motif['id']}" if motif["id"] else "heuristic",
                "matches": matches[:20],
            })
    return hits
