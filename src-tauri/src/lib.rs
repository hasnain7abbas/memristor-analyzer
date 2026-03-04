mod file_io;
mod smoothing;
mod parameters;
mod ann;
mod export;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            file_io::read_file,
            file_io::list_sheets,
            smoothing::smooth_data,
            parameters::extract_parameters,
            ann::train_ann,
            export::generate_python_script,
            export::export_chart_data,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
