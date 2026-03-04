interface DataTableProps {
  headers: string[];
  rows: Record<string, number | string>[];
  maxRows?: number;
}

export function DataTable({ headers, rows, maxRows = 5 }: DataTableProps) {
  const displayRows = rows.slice(0, maxRows);

  return (
    <div className="overflow-x-auto rounded-lg border border-border">
      <table className="w-full text-sm">
        <thead>
          <tr className="bg-surface-alt">
            {headers.map((h) => (
              <th
                key={h}
                className="px-3 py-2 text-left text-text-muted font-medium border-b border-border whitespace-nowrap"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {displayRows.map((row, i) => (
            <tr key={i} className="border-b border-border/50 hover:bg-surface-alt/50">
              {headers.map((h) => (
                <td key={h} className="px-3 py-1.5 font-mono text-xs text-text whitespace-nowrap">
                  {row[h] != null ? String(row[h]) : '—'}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > maxRows && (
        <div className="px-3 py-1.5 text-xs text-text-dim bg-surface-alt/30">
          Showing {maxRows} of {rows.length} rows
        </div>
      )}
    </div>
  );
}
