# Recordings and generated results

Your prior recordings, training runs, mission images and CSV logs are preserved.
The old dashboard's jobs, profiles, model metadata, history and trash were moved
intact to `archived_webapp_20261030/`. They are archived data, not an active server.

`robust-april-tag.py` can record a new AVI with `--record` or timing CSV with
`--log-csv`; both refuse to overwrite existing files. `training/` commands continue
to save training outputs here. The simplified `stop-and-stare.py` prints decisions
directly in its terminal and does not start a background job or create display files.
Generated contents remain ignored by Git.
