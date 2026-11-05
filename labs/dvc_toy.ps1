# Task 1.4 (toy part) — DVC on a tiny CSV. Run line by line in PowerShell and
# screenshot each step. Requires: git, dvc (pip install dvc), and a DagsHub repo
# named dvc-toy (or reuse your <ROLL>_A2_Ops repo).
#
#   $env:DAGSHUB_USER  = "your-dagshub-username"
#   $env:DAGSHUB_TOKEN = "your-dagshub-token"   # Settings > Tokens on DagsHub; never commit it
#   $env:TOY_REPO      = "dvc-toy"

$ErrorActionPreference = "Stop"
$toy = "$HOME\Desktop\dvc_toy"
New-Item -ItemType Directory -Force $toy | Out-Null
Set-Location $toy

# 1. empty folder + small CSV, initialise Git and DVC
"id,value`n1,10`n2,20`n3,30" | Out-File -Encoding utf8 data.csv
git init
dvc init                       # creates .dvc/ (config, cache) and .dvcignore
git commit -m "Initialise DVC"

# 2. track the CSV: DVC moves the content into .dvc/cache and writes data.csv.dvc (pointer)
dvc add data.csv               # also adds data.csv to .gitignore
Get-Content data.csv.dvc       # shows md5 hash + size + path: this is all Git stores
git add data.csv.dvc .gitignore
git commit -m "Track data.csv v1 with DVC"

# 3. change the data, commit version 2
"4,40" | Out-File -Append -Encoding utf8 data.csv
dvc add data.csv
git add data.csv.dvc
git commit -m "data.csv v2 (added row 4)"
git log --oneline

# 4. return to the older version: Git restores the old pointer, DVC restores the matching content
git checkout HEAD~1 -- data.csv.dvc
dvc checkout data.csv
Get-Content data.csv           # 3 rows again
git checkout HEAD -- data.csv.dvc   # back to v2
dvc checkout data.csv

# 5. remote storage on DagsHub (credentials stored in .dvc/config.local, which is git-ignored)
git remote add origin "https://dagshub.com/$env:DAGSHUB_USER/$env:TOY_REPO.git"
dvc remote add -d origin "https://dagshub.com/$env:DAGSHUB_USER/$env:TOY_REPO.dvc"
dvc remote modify origin --local auth basic
dvc remote modify origin --local user $env:DAGSHUB_USER
dvc remote modify origin --local password $env:DAGSHUB_TOKEN
git add .dvc/config
git commit -m "Configure DagsHub DVC remote"
git push -u origin HEAD          # pushes code + pointer files (enter DagsHub username/token if asked)
dvc push                         # pushes the actual data to DagsHub storage

# 6. delete local copy and cache, then pull it back from the remote
Remove-Item data.csv
Remove-Item -Recurse -Force .dvc\cache
dvc pull
Get-Content data.csv
