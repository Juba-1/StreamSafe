#!/bin/bash
# Rebuild and publish the site to Vercel (account juba-1, team juba-1s-projects, project "streamsafe").
# Live address: https://streamsafe-sa.vercel.app (streamsafe-nu.vercel.app redirects there).

# Needs `vercel login` once on the machine that runs it.
set -eu
cd "$(dirname "$0")"
python3 build.py
D=$(mktemp -d)/streamsafe
cp -r public "$D"
# The installer (about 145 MB) is delivered by Dodo Payments as a Digital File entitlement: Vercel Hobby refuses files over 100 MB.
cd "$D" && vercel deploy --prod --yes --scope juba-1s-projects
