<#
.SYNOPSIS
    Genera el ZIP de entrega del TFM con el nombre exigido por la guia.

.DESCRIPTION
    La guia (pag. 7, "Notas Generales") exige un nombre de fichero con el patron
    Nombre_Apellido1_Apellido2_TituloTrabajo.zip (ejemplo de la propia guia:
    "Maria_Garcia_Perez_Estudio_pajaros.zip"). Para este TFM, el nombre es:

        David_Sandoval_Aleman_Prediccion_emisiones_buques.zip

    El script:
      1. Regenera mrv.sqlite (notebook 09) si no existe o esta desactualizado
         respecto al consolidado, para que el Anexo F y el fichero sean de la
         misma pasada.
      2. Copia el repositorio a una carpeta temporal, excluyendo lo que nunca
         debe ir al ZIP: .git, entornos virtuales, cachés y ficheros temporales
         de Office.
      3. Comprime esa copia con el nombre correcto en el Escritorio.

.PARAMETER Destino
    Carpeta donde se deja el ZIP final. Por defecto, el Escritorio.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\scripts\empaquetar_zip.ps1
#>

param(
    [string]$Destino = "$env:USERPROFILE\Desktop"
)

$ErrorActionPreference = 'Stop'

$NombreZip = "David_Sandoval_Aleman_Prediccion_emisiones_buques.zip"
$RepoRoot  = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ZipPath   = Join-Path $Destino $NombreZip

Write-Host "Repositorio: $RepoRoot"
Write-Host "ZIP de salida: $ZipPath"
Write-Host ""

# --- 1. Regenerar mrv.sqlite si hace falta -------------------------------
$Sqlite      = Join-Path $RepoRoot "data\processed\mrv.sqlite"
$Consolidado = Join-Path $RepoRoot "data\processed\mrv_consolidado.parquet"

$hayQueRegenerar = $true
if ((Test-Path $Sqlite) -and (Test-Path $Consolidado)) {
    $mtimeSqlite = (Get-Item $Sqlite).LastWriteTimeUtc
    $mtimeParquet = (Get-Item $Consolidado).LastWriteTimeUtc
    if ($mtimeSqlite -gt $mtimeParquet) {
        $hayQueRegenerar = $false
    }
}

if ($hayQueRegenerar) {
    Write-Host "mrv.sqlite no existe o es anterior al consolidado: reejecutando notebook 09..."
    Push-Location (Join-Path $RepoRoot "notebooks")
    try {
        jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=120 09_bbdd_sql.ipynb
        if ($LASTEXITCODE -ne 0) { throw "La reejecucion del notebook 09 devolvio un error." }
    }
    finally {
        Pop-Location
    }
    Write-Host "mrv.sqlite regenerado."
}
else {
    Write-Host "mrv.sqlite ya esta al dia respecto al consolidado: no hace falta reejecutar el notebook 09."
}
Write-Host ""

# --- 2. Copiar el repo a una carpeta temporal, excluyendo lo que no va al ZIP
$ExcluirDirs = @(
    '.git',
    '.pytest_cache',
    '__pycache__',
    '.ipynb_checkpoints',
    '.venv',
    'venv'
)
$ExcluirFicheros = @(
    '~$*'
)

$Temp = Join-Path $env:TEMP ("tfm_zip_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $Temp | Out-Null

Write-Host "Copiando el repositorio a una carpeta temporal (excluyendo lo que no va al ZIP)..."

# robocopy con /XD (excluir directorios) y /XF (excluir ficheros); /E copia
# subcarpetas incluidas las vacias. Los codigos de salida 0-7 de robocopy son
# exito, asi que se comprueba explicitamente en vez de fiarse de $LASTEXITCODE.
robocopy $RepoRoot $Temp /E /XD $ExcluirDirs /XF $ExcluirFicheros /NFL /NDL /NJH /NJS | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy fallo al copiar el repositorio (codigo $LASTEXITCODE)." }

# --- 3. Comprimir ---------------------------------------------------------
if (Test-Path $ZipPath) {
    Write-Host "Ya existe un ZIP anterior en el destino: se sobrescribe."
    Remove-Item $ZipPath -Force
}

Write-Host "Comprimiendo..."
Compress-Archive -Path (Join-Path $Temp '*') -DestinationPath $ZipPath -CompressionLevel Optimal

Remove-Item $Temp -Recurse -Force

$tamanoMB = [math]::Round((Get-Item $ZipPath).Length / 1MB, 1)
Write-Host ""
Write-Host "Listo: $ZipPath ($tamanoMB MB)"
Write-Host "Nombre segun la guia (pag. 7): Nombre_Apellido1_Apellido2_TituloTrabajo.zip"
