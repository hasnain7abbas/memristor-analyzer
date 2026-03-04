// Column auto-mapping utilities
export function autoDetectColumnType(header: string): string | null {
  const patterns: [RegExp, string][] = [
    [/conductance|cond|g_?us|^g$/i, 'conductance_uS'],
    [/current|i_?ua/i, 'current_uA'],
    [/pulse|num|#|^n$/i, 'pulse_number'],
    [/type|phase/i, 'type'],
    [/cycle/i, 'cycle'],
    [/time|^t$/i, 'time_s'],
    [/voltage|^v$/i, 'voltage_V'],
  ];

  for (const [pattern, type] of patterns) {
    if (pattern.test(header)) return type;
  }
  return null;
}
