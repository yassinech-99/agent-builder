const PATTERNS: { label: string; regex: RegExp }[] = [
  { label: "Email address", regex: /[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+/ },
  { label: "Phone number", regex: /\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b/ },
  { label: "SSN", regex: /\b\d{3}-\d{2}-\d{4}\b/ },
  { label: "NVIDIA NIM API Key", regex: /nvapi-[A-Za-z0-9_-]{20,}/ },
  { label: "Generic API Key", regex: /(?:api[_-]?key|apikey|secret[_-]?key)\s*[:=]\s*\S+/i },
];

export function detectRiskyInput(text: string): string[] {
  return PATTERNS.filter((p) => p.regex.test(text)).map((p) => p.label);
}

export function redactSensitiveText(text: string): string {
  let result = text;
  for (const p of PATTERNS) {
    result = result.replace(p.regex, "[REDACTED]");
  }
  return result;
}
