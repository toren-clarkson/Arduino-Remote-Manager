# Printable AprilTags

```powershell
.\.venv\Scripts\python.exe -m tools.generate_tag 3
```

This generates `assets/tags/apriltag_36h11_id3.png` using the tag36h11 dictionary, with a white quiet border. Change the number for another ID; the program validates the ID range and refuses to overwrite an existing file. `--output path\to\new.png` chooses another output path.

ID 0 already exists in `assets/tags/`; print it directly or choose a new filename to regenerate it. Print flat, with the border intact and good contrast. Set the same ID in the mission's `--tag-id` or configuration. A tag is not a QR code. Apparent size thresholds are image ratios, not physical distances.
