# Printable markers

The [local webapp](../../webapp/README.md) AprilTags page generates and prints
markers, tests real photos, and injects synthetic tags into test images.
Select the same ID and detector settings in a saved mission profile.

`apriltag_36h11_id0.png` is the existing ID 0 marker. Keep the white border when printing, mount it flat, and avoid glare. Generate another ID with `python -m tools.generate_tag ID` from the root. See [tools/README.md](../../tools/README.md).
