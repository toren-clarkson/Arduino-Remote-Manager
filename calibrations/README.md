# Robot profiles and console preferences

`adaptive/` stores the new standalone adaptive mission's robot/surface profiles.
Use `python run.py`, select Adaptive homing, and choose Learn robot/surface profile.
Each profile has motor pins, limits, first observed moving PWM, separate learned
ball/tag response matrices, estimator state and target signatures. Create separate
profiles for different bodies, surfaces and camera mounting. These JSON files are
created locally and ignored by Git.

The OpenCV HSV variant also saves the clicked color in its adaptive profile.
Drive restores that HSV; switching from YOLO or changing the color/tolerances
requires matching object calibration. Use `python adaptive-homing.py --color --setup`.

`console/` stores per-program choices made with `--setup`. Other missions can use
console preferences without using a learned body model. Explicit CLI arguments
override the legacy setup defaults. Direct runs without setup use script/CLI defaults.

Old webapp body profiles, if present, remain historical data. They are incompatible
with the new adaptive schema and are not loaded. See the
[adaptive calibration guide](../docs/ADAPTIVE_HOMING.md) for operation and limitations.
