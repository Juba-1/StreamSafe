# The risk heads that ship inside the installer

Put the owner's real head files here (private repository only):

    fast_h6.txt  male_h6.txt  thr_h6.txt  [scene_h.txt]  heads.json      heads.json: {"version": 1}

`tools/make_heads.py` copies them into the installer. Without them (a CI test build) it writes synthetic
PLACEHOLDER heads: the installer then builds and installs, but the app refuses to start a stream with them.
A build from a `v*` tag fails when this folder has no real heads. Raise `version` whenever the heads change: an
update (signed manifest) only replaces heads with a higher version.
