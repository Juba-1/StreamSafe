#!/bin/bash
# Rebuild and publish the site to Vercel (account juba-1, team juba-1s-projects, project "streamsafe").
# Live address: https://streamsafe-nu.vercel.app   Needs `vercel login` once on the machine that runs it.
set -eu
cd "$(dirname "$0")"
python3 build.py
D=$(mktemp -d)/streamsafe
cp -r public "$D"
cd "$D" && vercel deploy --prod --yes --scope juba-1s-projects
