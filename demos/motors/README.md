# Hardware checks and motor template

## Pin 13 Wi-Fi LED check

```powershell
.\.venv\Scripts\python.exe -m demos.motors.blink_led
```

`blink_led.py` was `test_wifi_led.py`. Set `ARDUINO_IP` near the top to the address shown by Serial Monitor. It connects using Telemetrix Wi-Fi AIO, configures pin 13 as an output, sends ten half-second ON/OFF pairs, leaves the LED off, and closes the socket. It sends no motor commands. Watch the physical L LED; terminal output only confirms a write was requested. Close any other Python board client before the test.

## Dual-motor camera template

```powershell
.\.venv\Scripts\python.exe -m demos.motors.dual_motor
```

`dual_motor.py` was `dual_motor_template.py`. Set the IP and pins in its edit block. It opens a PhoneSense preview and configures the motors. The example `await move_motors(0.5, 0.5)` is intentionally commented out. To build an experiment, insert commands inside the marked loop block. Call repeatedly to advance the shared PWM ramp. Speed demands range from -1 (reverse) to +1 (forward), with zero requesting a ramp toward a stop.

Implemented: one event loop for board/camera work, configurable motor-start PWM floor, speed ramps, reversal through zero, stale-camera stop, and an immediate stop during exit cleanup. This is a template, not an autonomous search or tag mission. Its original pin directions and IP may differ from the current mission's settings.
