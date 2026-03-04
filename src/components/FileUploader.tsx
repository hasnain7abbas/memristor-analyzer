import { useState, useCallback } from 'react';
import { Upload, FileSpreadsheet } from 'lucide-react';
import { open } from '@tauri-apps/plugin-dialog';
import { invoke } from '@tauri-apps/api/core';
import type { Dataset } from '../types';

interface FileUploaderProps {
  onFileLoaded: (dataset: Dataset, filePath: string) => void;
  accept?: string[];
}

export function FileUploader({ onFileLoaded, accept }: FileUploaderProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [sheets, setSheets] = useState<string[]>([]);
  const [selectedSheet, setSelectedSheet] = useState<string | null>(null);
  const [filePath, setFilePath] = useState<string | null>(null);

  const loadFile = useCallback(
    async (path: string, sheetName?: string) => {
      setLoading(true);
      setError(null);
      try {
        // Check for Excel to list sheets
        const ext = path.split('.').pop()?.toLowerCase();
        if (['xlsx', 'xls', 'xlsm', 'xlsb', 'ods'].includes(ext || '')) {
          const sheetList = await invoke<string[]>('list_sheets', { path });
          if (sheetList.length > 1 && !sheetName) {
            setSheets(sheetList);
            setFilePath(path);
            setSelectedSheet(sheetList[0]);
            setLoading(false);
            return;
          }
        }

        const result = await invoke<{ headers: string[]; rows: Record<string, unknown>[] }>(
          'read_file',
          { path, sheetName: sheetName || null },
        );

        const filename = path.split(/[\\/]/).pop() || path;
        const columns = result.headers.map((h) => ({
          name: h,
          values: result.rows
            .map((r) => {
              const v = r[h];
              return typeof v === 'number' ? v : parseFloat(String(v));
            })
            .filter((v) => !isNaN(v)),
        }));

        const dataset: Dataset = {
          filename,
          sheetName: sheetName || undefined,
          headers: result.headers,
          rows: result.rows as Record<string, number | string>[],
          columns,
        };

        onFileLoaded(dataset, path);
        setSheets([]);
        setFilePath(null);
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    },
    [onFileLoaded],
  );

  const handlePick = async () => {
    const selected = await open({
      multiple: false,
      filters: [
        {
          name: 'Data files',
          extensions: accept || ['xlsx', 'xls', 'csv', 'tsv', 'txt'],
        },
      ],
    });
    if (selected) {
      await loadFile(selected as string);
    }
  };

  const handleSheetSelect = () => {
    if (filePath && selectedSheet) {
      loadFile(filePath, selectedSheet);
    }
  };

  return (
    <div className="space-y-3">
      <div
        onClick={handlePick}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
        }}
        className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all ${
          dragOver
            ? 'border-accent bg-accent/5'
            : 'border-border hover:border-text-dim hover:bg-surface-alt/50'
        }`}
      >
        {loading ? (
          <div className="flex items-center justify-center gap-2 text-text-muted">
            <div className="w-5 h-5 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
            Loading...
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2">
            <Upload size={24} className="text-text-dim" />
            <p className="text-sm text-text-muted">
              Click to browse or drag & drop
            </p>
            <p className="text-xs text-text-dim">.xlsx, .xls, .csv, .tsv</p>
          </div>
        )}
      </div>

      {sheets.length > 1 && (
        <div className="flex items-center gap-3 p-3 bg-surface-alt rounded-lg">
          <FileSpreadsheet size={16} className="text-amber" />
          <select
            value={selectedSheet || ''}
            onChange={(e) => setSelectedSheet(e.target.value)}
            className="flex-1 bg-surface border border-border rounded px-3 py-1.5 text-sm text-text"
          >
            {sheets.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
          <button
            onClick={handleSheetSelect}
            className="px-3 py-1.5 bg-accent text-white text-sm rounded-lg hover:bg-accent/80"
          >
            Load Sheet
          </button>
        </div>
      )}

      {error && (
        <p className="text-sm text-red bg-red/10 px-3 py-2 rounded-lg">{error}</p>
      )}
    </div>
  );
}
