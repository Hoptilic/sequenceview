const PROSITE_ONLY = /[-<>{}]|\(\d/;

export function prositeToRegex(pattern) {
  let source = pattern.trim().replace(/\.$/, "");
  source = source
    .replace(/\{([A-Za-z]+)\}/g, "[^$1]")
    .replace(/\((\d+)(?:,(\d+))?\)/g, (_, low, high) => (high ? `{${low},${high}}` : `{${low}}`))
    .replace(/x/g, ".")
    .replace(/-/g, "")
    .replace(/</g, "^")
    .replace(/>/g, "$");
  return source;
}

export function findMotifs(sequence, pattern) {
  const text = pattern.trim();
  if (!text) {
    return { matches: [], error: "" };
  }

  const source = PROSITE_ONLY.test(text) ? prositeToRegex(text) : text;
  let expression;
  try {
    expression = new RegExp(source, "gi");
  } catch {
    return { matches: [], error: "Invalid pattern." };
  }

  const matches = [];
  let match = expression.exec(sequence);
  while (match && matches.length < 500) {
    if (match[0].length) {
      matches.push({ start: match.index, end: match.index + match[0].length });
    }
    expression.lastIndex = match.index + 1;
    match = expression.exec(sequence);
  }
  return { matches, error: "" };
}

export function motifMask(length, matches) {
  const mask = new Uint8Array(length);
  matches.forEach(({ start, end }) => mask.fill(1, start, Math.min(end, length)));
  return mask;
}
