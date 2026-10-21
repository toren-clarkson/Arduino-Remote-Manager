# Robot missions

Use [stop_and_stare/README.md](stop_and_stare/README.md) for the main YOLO-object -> AprilTag program. Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m missions.stop_and_stare.run --target-class bottle
```

Default mode previews decisions without an Arduino connection. `--drive` enables timed motor steps. All perception happens while commanded motor PWM is zero; movement, braking and settling ignore the video feed. The mission folder documents every module, configuration option, training workflow, threshold and known timing limit.
