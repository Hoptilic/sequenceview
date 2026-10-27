const MAX_SHARE_LENGTH = 6000;

export function encodeSequenceHash(text) {
  if (!text || text.length > MAX_SHARE_LENGTH) {
    return null;
  }
  const bytes = new TextEncoder().encode(text);
  let binary = "";
  bytes.forEach((byte) => {
    binary += String.fromCharCode(byte);
  });
  return `seq=${btoa(binary).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "")}`;
}

export function decodeSequenceHash(hash) {
  const match = /(?:^#|&)seq=([A-Za-z0-9_-]+)/.exec(hash || "");
  if (!match) {
    return "";
  }
  try {
    const base64 = match[1].replaceAll("-", "+").replaceAll("_", "/");
    const binary = atob(base64 + "=".repeat((4 - (base64.length % 4)) % 4));
    return new TextDecoder().decode(Uint8Array.from(binary, (character) => character.charCodeAt(0)));
  } catch {
    return "";
  }
}
