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

    Ok(DatasetJson { headers, rows })
}

fn read_csv(path: &str, delimiter: u8) -> Result<DatasetJson, String> {
    let mut rdr = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .flexible(true)
        .has_headers(true)
        .from_path(path)
        .map_err(|e| format!("Cannot open CSV file: {}", e))?;

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
        let record = result.map_err(|e| format!("Error reading row: {}", e))?;
        let mut map = HashMap::new();
        for (i, field) in record.iter().enumerate() {
            if i >= headers.len() {
                break;
            }
            let trimmed = field.trim();
            let value = if trimmed.is_empty() {
                serde_json::Value::Null
            } else if let Ok(f) = trimmed.parse::<f64>() {
                serde_json::Value::Number(
                    serde_json::Number::from_f64(f).unwrap_or(serde_json::Number::from(0)),
                )
            } else {
                serde_json::Value::String(trimmed.to_string())
            };
            map.insert(headers[i].clone(), value);
        }
        rows.push(map);
    }

    Ok(DatasetJson { headers, rows })
}
