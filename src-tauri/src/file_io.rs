use calamine::{open_workbook_auto, Data, DataType, Reader};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DatasetJson {
    pub headers: Vec<String>,
    pub rows: Vec<HashMap<String, serde_json::Value>>,
}

#[tauri::command]
pub fn read_file(path: String, sheet_name: Option<String>) -> Result<DatasetJson, String> {
    let ext = path
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "xlsx" | "xls" | "xlsm" | "xlsb" | "ods" => read_excel(&path, sheet_name),
        "csv" => read_csv(&path, b','),
        "tsv" => read_csv(&path, b'\t'),
        "txt" => read_csv(&path, b'\t'),
        _ => Err(format!("Unsupported file format: .{}", ext)),
    }
}

#[tauri::command]
pub fn list_sheets(path: String) -> Result<Vec<String>, String> {
    let workbook = open_workbook_auto(&path).map_err(|e| format!("Cannot open file: {}", e))?;
    Ok(workbook.sheet_names().to_vec())
}

fn read_excel(path: &str, sheet_name: Option<String>) -> Result<DatasetJson, String> {
    let mut workbook =
        open_workbook_auto(path).map_err(|e| format!("Cannot open Excel file: {}", e))?;

    let sheet = match &sheet_name {
        Some(name) => name.clone(),
        None => {
            let names = workbook.sheet_names().to_vec();
            if names.is_empty() {
                return Err("No sheets found in workbook".to_string());
            }
            names[0].clone()
        }
    };

    let range = workbook
        .worksheet_range(&sheet)
        .map_err(|e| format!("Cannot read sheet '{}': {}", sheet, e))?;

    let mut rows_iter = range.rows();

    // Find header row (first non-empty row)
    let header_row = loop {
        match rows_iter.next() {
            Some(row) => {
                let non_empty: Vec<_> = row
                    .iter()
                    .filter(|c| !c.is_empty())
                    .collect();
                if !non_empty.is_empty() {
                    break row;
                }
            }
            None => return Err("No data found in sheet".to_string()),
        }
    };

    let headers: Vec<String> = header_row
        .iter()
        .enumerate()
        .map(|(i, cell)| {
            let s = cell.to_string().trim().to_string();
            if s.is_empty() {
                format!("Column_{}", i + 1)
            } else {
                s
            }
        })
        .collect();

    let mut rows = Vec::new();
    for row in rows_iter {
        let mut map = HashMap::new();
        let mut has_data = false;
        for (i, cell) in row.iter().enumerate() {
            if i >= headers.len() {
                break;
            }
            let value = match cell {
                Data::Float(f) => {
                    has_data = true;
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(*f).unwrap_or(serde_json::Number::from(0)),
                    )
                }
                Data::Int(n) => {
                    has_data = true;
                    serde_json::Value::Number(serde_json::Number::from(*n))
                }
                Data::String(s) => {
                    let trimmed = s.trim().to_string();
                    if trimmed.is_empty() {
                        serde_json::Value::Null
                    } else {
                        has_data = true;
                        // Try parsing as number
                        if let Ok(f) = trimmed.parse::<f64>() {
                            serde_json::Value::Number(
                                serde_json::Number::from_f64(f)
                                    .unwrap_or(serde_json::Number::from(0)),
                            )
                        } else {
                            serde_json::Value::String(trimmed)
                        }
                    }
                }
                Data::Bool(b) => {
                    has_data = true;
                    serde_json::Value::Bool(*b)
                }
                Data::Empty => serde_json::Value::Null,
                _ => {
                    let s = cell.to_string().trim().to_string();
                    if s.is_empty() {
                        serde_json::Value::Null
                    } else {
                        has_data = true;
                        if let Ok(f) = s.parse::<f64>() {
                            serde_json::Value::Number(
                                serde_json::Number::from_f64(f)
                                    .unwrap_or(serde_json::Number::from(0)),
                            )
                        } else {
                            serde_json::Value::String(s)
                        }
                    }
                }
            };
            map.insert(headers[i].clone(), value);
        }
        if has_data {
            rows.push(map);
        }
    }

    if rows.is_empty() {
        return Err(format!(
            "No data rows found in sheet '{}'. Found {} column headers: [{}]. \
             The sheet may be empty — check that your instrument recorded data correctly.",
            sheet,
            headers.len(),
            headers.join(", ")
        ));
    }

    Ok(DatasetJson { headers, rows })
}

fn read_csv(path: &str, delimiter: u8) -> Result<DatasetJson, String> {
    // Read the entire file to handle metadata rows before headers
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read file: {}", e))?;

    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Err("File is empty".to_string());
    }

    // Strategy: find the header row by looking for the first line where
    // most fields look like column headers (not metadata key-value pairs).
    // Keithley instruments often output metadata lines like "Test Type\tIV Sweep"
    // before the actual column headers.
    let header_line_idx = find_header_line(&lines, delimiter);

    // Build a CSV reader from the remaining content (header + data)
    let remaining: String = lines[header_line_idx..]
        .iter()
        .map(|l| *l)
        .collect::<Vec<&str>>()
        .join("\n");

    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .flexible(true)
        .has_headers(true)
        .from_reader(remaining.as_bytes());

    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| format!("Cannot read headers: {}", e))?
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let s = h.trim().to_string();
            if s.is_empty() {
                format!("Column_{}", i + 1)
            } else {
                s
            }
        })
        .collect();

    let mut rows = Vec::new();
    for result in rdr.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue, // Skip malformed rows
        };
        let mut map = HashMap::new();
        let mut has_numeric = false;
        for (i, field) in record.iter().enumerate() {
            if i >= headers.len() {
                break;
            }
            let trimmed = field.trim();
            let value = if trimmed.is_empty() {
                serde_json::Value::Null
            } else if let Ok(f) = trimmed.parse::<f64>() {
                has_numeric = true;
                serde_json::Value::Number(
                    serde_json::Number::from_f64(f).unwrap_or(serde_json::Number::from(0)),
                )
            } else {
                serde_json::Value::String(trimmed.to_string())
            };
            map.insert(headers[i].clone(), value);
        }
        // Only include rows that have at least one numeric value
        // This filters out stray metadata or comment rows below headers
        if has_numeric {
            rows.push(map);
        }
    }

    if rows.is_empty() {
        return Err(format!(
            "No numeric data found in file. Found {} column headers: [{}]. \
             Check that the file contains actual measurement data.",
            headers.len(),
            headers.join(", ")
        ));
    }

    Ok(DatasetJson { headers, rows })
}

/// Find the header line index by scoring each line.
/// The header line should have multiple fields and few/no pure numbers.
/// Metadata lines typically have 1-2 fields (key-value pairs) or are all text.
fn find_header_line(lines: &[&str], delimiter: u8) -> usize {
    let delim_char = delimiter as char;
    let mut best_idx = 0;
    let mut best_score = 0i32;

    for (i, line) in lines.iter().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(delim_char).collect();
        let num_fields = fields.len();

        // Skip lines with only 1-2 fields (likely metadata key:value)
        if num_fields < 2 {
            continue;
        }

        // Score: prefer lines with more fields where most fields are text (headers)
        let num_text = fields
            .iter()
            .filter(|f| {
                let t = f.trim();
                !t.is_empty() && t.parse::<f64>().is_err()
            })
            .count();
        let num_numeric = fields
            .iter()
            .filter(|f| f.trim().parse::<f64>().is_ok())
            .count();

        // A good header row: many text fields, few/no numeric fields, many columns
        let score = (num_text as i32 * 3) + (num_fields as i32) - (num_numeric as i32 * 5);

        if score > best_score {
            best_score = score;
            best_idx = i;
        }

        // If we've found a line that looks like headers, check if the next line has data
        if num_text >= 2 && num_numeric == 0 && i + 1 < lines.len() {
            let next_fields: Vec<&str> = lines[i + 1].split(delim_char).collect();
            let next_numeric = next_fields
                .iter()
                .filter(|f| f.trim().parse::<f64>().is_ok())
                .count();
            // If next line has numeric data, this is very likely the header
            if next_numeric >= 1 {
                return i;
            }
        }
    }

    best_idx
}
