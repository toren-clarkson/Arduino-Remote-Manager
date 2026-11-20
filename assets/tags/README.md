# Printable AprilTags

`apriltag_36h11_id0.png` is the existing ID 0 marker used by the default mission.
Print it flat, keep the white border and avoid glare. Select the same ID with
`--tag-id` in the script. Generate another marker from the project root:

```powershell
.\.venv\Scripts\python.exe -m tools.generate_tag 3
```

See [tools](../../tools/README.md). Marker image thresholds are not physical distances.
