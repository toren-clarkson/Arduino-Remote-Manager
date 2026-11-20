# Hardware-free verification

From the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Checks cover PWM ramps/reversal/stopping, intermittent search, short-term AprilTag
memory, real OpenCV/AprilTag detections, recorded-video replay, YOLO result freshness
and training helper input validation. The standalone stop-and-stare tests simulate
an entire ball -> tag mission and verify that neither detection nor target-memory
updates uses moving-camera frames or runs while commanded PWM is nonzero.

No test connects a physical Arduino. `test_wifi_led.py` at the project root is a
manual hardware check, not part of this suite. The removed dashboard and framework
no longer have tests or runtime dependencies. Physical balance, friction PWM and
stopping distance still need calibration on the robot.

`test_network.py` checks the AP firmware/Python address contract, all eight clients,
PhoneSense PC-address selection with Ethernet/VPN interfaces, DHCP changes and
invalid addresses. Its routing test verifies that no probe packets or TCP commands
are sent. See the [firmware guide](../Telemetrix4UnoR4WIFI/README.md) for compilation.

`test_adaptive_homing.py` covers signed wheel identification, unequal/reversed
motors, RLS adaptation to gain changes, profile persistence and invalid inputs,
PWM integration, brief prediction and long loss, fresh arrival confirmation,
simulated closed-loop convergence, real OpenCV tags, ByteTrack ID lock, and an
entire continuous ball -> tag mission using a simulated board. `test_console.py`
checks menu input and preference persistence without hardware.

`test_adaptive_color.py` checks clicked HSV selection/locking, red hue wraparound,
blue distractor rejection, stored HSV/profile compatibility, absence of YOLO
loading, and real OpenCV color -> AprilTag replay without a phone or robot.
