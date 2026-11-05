param(
    [string]$Compiler = "xelatex",
    [string]$ReportPython = "python"
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    if ([System.IO.Path]::GetFileNameWithoutExtension($Compiler) -eq "tectonic") {
        & $Compiler --keep-logs --keep-intermediates main.tex
        if ($LASTEXITCODE -ne 0) { throw "TeX compilation failed." }
    } else {
        for ($pass = 0; $pass -lt 2; $pass++) {
            & $Compiler -interaction=nonstopmode -halt-on-error main.tex
            if ($LASTEXITCODE -ne 0) { throw "TeX compilation failed." }
        }
    }
    & $ReportPython -X utf8 finalize_pdf.py
    if ($LASTEXITCODE -ne 0) { throw "PDF text mapping or validation failed." }
} finally {
    Pop-Location
}
