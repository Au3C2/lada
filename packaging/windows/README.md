## Build exe

```powershell
powershell -ExecutionPolicy Bypass ./packaging/windows/package_executable.ps1 -extra nvidia
```

Script options:
* `-extra <EXTRA>`: Install python extra. EXTRA currently can be `intel` or `nvidia`
* `-skipWinget`: Skip installing/upgrading system dependencies via winget
* `-skipGvsbuild`: Skip installing/upgrading system dependencies via gvsbuild
* `-skipArchive`: Skip creating 7z archives
* `-skipTranslations`: Skip compiling translations. If set no translations will be included in the release
* `-cleanGvsbuild`: Does a clean build of gvsbuild
* `-cliOnly`: Builds only `lada-cli.exe`

> [!TIP]
> If you updated `gvsbuild`, `uv` or `python` do a clean build (`--clean-gvsbuild`)

> [!TIP]
> If you get a build error about *Clock skew* check you Date & Time settings. If it doesn't work rewrite timestamps:
> ```Powershell
> $now = (Get-Date)
> Get-ChildItem -Path ./build_gtk_release/build -Recurse | ForEach-Object { $_.LastWriteTime = $now }
> ```

> [!TIP]
> After doing major packaging changes that involve system dependencies test the exe on another pristine Windows VM.
> 
> This makes sure that PyInstaller picked up all required dependencies which are available on the build machine but maybe aren't on the users Windows installation.

## Publish exe

* Attach the `lada-<version>.7z.00N` chunks and their `.sha256` files to the GitHub Draft Release — the CI release workflow does this automatically
* GitHub Releases is the only download source; there is no pixeldrain upload anymore

## Update gvsbuild

Checklist when updating gvsbuild / Windows GUI system dependencies. Some steps might not be relevant depending on the change.

* Update `package_executable.ps1` (at least version)
* Update/add/remove patches for gvsbuild
* Rebuild and commit the new archive to `packaging/windows/deps/` (or host it and set the `LADA_WINDOWS_GTK_ARCHIVE_URL` repository variable)
