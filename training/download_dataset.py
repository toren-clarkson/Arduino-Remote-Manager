"""Optional Roboflow YOLO-format export download using ROBOFLOW_API_KEY from the environment."""

import argparse
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


def safe_members(archive, output):
    root = output.resolve()
    for member in archive.infolist():
        destination = (root/member.filename).resolve()
        if not destination.is_relative_to(root) or (member.external_attr >> 16) & 0o170000 == 0o120000:
            raise ValueError("Unsafe path in dataset archive")
        yield member


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        parser.error("Set ROBOFLOW_API_KEY locally; do not put the key in source code")
    if args.output.exists():
        parser.error("Choose an output directory that does not already exist")
    if args.version < 1:
        parser.error("version must be positive")
    path = "/".join(urllib.parse.quote(str(part), safe="") for part in
                    (args.workspace, args.project, args.version, "yolov8"))
    url = "https://api.roboflow.com/"+path+"?"+urllib.parse.urlencode({"api_key": key})
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            export = json.load(response)
        link = export.get("export", {}).get("link")
        if not link:
            raise SystemExit("No dataset export was returned. Generate the version/export in Roboflow first.")
        with tempfile.TemporaryDirectory() as folder:
            archive_path = Path(folder)/"dataset.zip"
            with urllib.request.urlopen(link, timeout=120) as response, archive_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            with zipfile.ZipFile(archive_path) as archive:
                members = list(safe_members(archive, args.output))
                args.output.mkdir(parents=True)
                archive.extractall(args.output, members=members)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Roboflow request failed with HTTP {exc.code}; check key, IDs, and export readiness.") from None
    except urllib.error.URLError:
        raise SystemExit("Could not reach the Roboflow export service.") from None
    print("Dataset saved:", args.output.resolve())


if __name__ == "__main__":
    main()
