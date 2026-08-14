# Launches the lada GUI from source with the gvsbuild GTK4 stack (build_gtk/).
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$gtk = "$root\build_gtk\gtk\x64\release"
$env:PATH = "$gtk\bin;$env:PATH"
$env:GI_TYPELIB_PATH = "$gtk\lib\girepository-1.0"
$env:GST_PLUGIN_PATH = "$gtk\lib\gstreamer-1.0"
$env:LOG_LEVEL = "DEBUG"
Set-Location $root
& "$root\.venv\Scripts\python.exe" -m lada.gui.main *>&1 | Tee-Object -FilePath "$root\gui_debug.log"
