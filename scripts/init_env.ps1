# Creates .env from .env.example with freshly generated database passwords.
# Never overwrites an existing .env. Run from the project root (start.bat does this).
$ErrorActionPreference = 'Stop'

if (Test-Path '.env') { Write-Host '.env already exists - leaving it untouched.'; exit 0 }
if (-not (Test-Path '.env.example')) { Write-Host 'ERROR: .env.example not found. Run this from the project root.'; exit 1 }

# A MySQL data volume keeps the passwords it was FIRST created with. If one already
# exists, generating new passwords would lock the backend out of the database.
$existing = $null
try { $existing = docker volume ls -q 2>$null | Select-String 'mysql_data' } catch { }
if ($existing) {
    Write-Host 'WARNING: an existing MySQL data volume was found, but there is no .env file.'
    Write-Host 'A new random password would not match the password stored in that volume.'
    Write-Host 'Choose ONE of these:'
    Write-Host '  1) Copy your previous .env into this folder, then run start.bat again.'
    Write-Host '  2) Reset the database (DELETES stored prediction history):'
    Write-Host '       docker compose down -v'
    Write-Host '     then run start.bat again.'
    exit 2
}

function New-Secret { -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 24 | ForEach-Object { [char]$_ }) }
$rootPw = New-Secret
$userPw = New-Secret
# Replace the longer placeholder first so 'change_me' does not clobber 'change_me_root'.
(Get-Content '.env.example') -replace 'change_me_root', $rootPw -replace 'change_me', $userPw |
    Set-Content -Path '.env' -Encoding ascii
Write-Host 'Created .env with generated database passwords (not shown, not committed).'
