# Task 1.4 (apply part) — version the real dataset in THIS repository with DVC,
# using DagsHub storage as the remote. Run from the project root after
# `git init` + connecting the repo to GitHub/DagsHub (see docs/RUNBOOK.md).
#   $env:DAGSHUB_USER / $env:DAGSHUB_TOKEN / $env:DAGSHUB_REPO must be set (DAGSHUB_REPO = <ROLL>_A2_Ops)

$ErrorActionPreference = "Stop"
if (-not (Test-Path .dvc)) { dvc init; git commit -m "Initialise DVC" }

dvc add data/raw                     # writes data/raw.dvc, git-ignores data/raw
git add data/raw.dvc data/.gitignore
git commit -m "Track Kaggle dataset with DVC"

dvc remote add -d -f origin "https://dagshub.com/$env:DAGSHUB_USER/$env:DAGSHUB_REPO.dvc"
dvc remote modify origin --local auth basic
dvc remote modify origin --local user $env:DAGSHUB_USER
dvc remote modify origin --local password $env:DAGSHUB_TOKEN
git add .dvc/config
git commit -m "DagsHub DVC remote"
dvc push
git push

Get-Content data/raw.dvc
