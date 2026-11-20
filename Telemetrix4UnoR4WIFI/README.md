# UNO R4 WiFi robot access point

Upload **this folder's** `Telemetrix4UnoR4WIFI.ino` to the **Arduino UNO R4 WiFi**.
An already-uploaded station-mode sketch does not change until you upload this
updated version. The older copy in Documents/Arduino is a separate file.

| Setting | Default |
|---|---|
| Robot Wi-Fi SSID | `UNO-R4-Robot` |
| Robot Wi-Fi password | `Robot295!` |
| Arduino / AP address | `192.168.4.1` |
| AP client subnet | `192.168.4.0/24` |
| Telemetrix TCP port | `31336` |

The sketch uses `WiFiS3`, explicitly calls `WiFi.config(ROBOT_AP_IP)`, then
`WiFi.beginAP(AP_SSID, AP_PASSWORD)`. It no longer joins an external network.
`192.168.4.1` is also the default in Arduino's official
[UNO R4 access-point example](https://github.com/arduino/ArduinoCore-renesas/blob/main/libraries/WiFiS3/examples/AP_SimpleWebServer/AP_SimpleWebServer.ino).
Here it is configured explicitly, so Python does not need to discover a DHCP lease.
The AP supplies DHCP addresses to the PC and phone; their own addresses can change.

## Connect and run

1. In Arduino IDE, open the sketch from this Main project. Select **Arduino UNO R4
   WiFi**, its USB port, then upload. Keep the existing Telemetrix4UnoR4, Servo,
   NewPing and DHTStable dependencies installed. The UNO R4 core supplies WiFiS3.
2. Power the robot. After startup, join **UNO-R4-Robot** with **Robot295!** on both
   the computer and phone. Accept **stay connected / use without Internet**.
   The AP is a local robot network; it does not provide an Internet connection.
3. Leave IPv4/DHCP automatic on both devices. Do not assign the PC `192.168.4.1`:
   that address belongs to the Arduino. The PC might receive `192.168.4.2`, `.3`,
   or another client address; no particular PC address is assumed.
4. Run a Python mission normally. All seven clients import the fixed Telemetrix
   endpoint from `robotics_lab/network.py`. No Arduino-IP argument is needed:

   ```powershell
   .\.venv\Scripts\python.exe test_wifi_led.py
   .\.venv\Scripts\python.exe stop-and-stare.py
   ```

   The LED script issues only pin-13 commands. The second command is camera-only
   preview; add `--drive` for physical robot motion after checking motor settings.
5. Open the mission's printed **PhoneSense PC camera URL** on the phone, allow the
   local HTTPS certificate/camera prompts, then **Connect** and **Start camera**.
   For example, if the PC received `.2`, the URL is
   `https://192.168.4.2:8080/phone`. Use the printed value, not this example.

```text
Phone 192.168.4.y -- robot AP -- PC 192.168.4.x:8080 (PhoneSense)
                                |
                                +-- Arduino 192.168.4.1:31336 (Telemetrix)
```

The phone streams to the PC through the AP. Python processes the frames and sends
motor commands to the Arduino. `192.168.4.1:31336` speaks Telemetrix, not HTTP and
not PhoneSense. Opening the Arduino address in a phone browser does not open the
camera page. Run only one Telemetrix control script at a time.

## What changed in firmware

- SSID/password now configure the robot's own AP. External-router credentials
  were removed from this copy of the sketch.
- The Serial wait is bounded at 1.5 seconds, so battery startup does not require
  opening Serial Monitor. Startup details are available at 115200 baud.
- Command numbers, Arduino ID, Telemetrix payloads and TCP port are unchanged.
- A normal TCP disconnect clears configured outputs, releases session sensor
  allocations and resets session state without rebooting the AP. A partial packet
  from a disconnected client can exit the parser and reach this cleanup.
- Explicit Telemetrix hard-reset commands still reboot the board. Python scripts
  use `hard_reset_on_shutdown=False` to preserve the AP on normal exit.

Disconnect cleanup runs after the TCP connection is observed closed; this is not
a firmware heartbeat watchdog for a connected-but-stalled host. That remains TODO.

## Automatic PC-address selection

The network helper chooses an assigned PC IPv4 address in `192.168.4.0/24`, excluding
the Arduino, network and broadcast addresses. If several match, it prefers the
source address of the OS route to the Arduino. This local UDP routing query sends
no packets or Telemetrix commands. It does not probe the board with a TCP connection.

PhoneSense still binds its normal listener; the helper selects the correct advertised
PC address rather than an unrelated Ethernet/VPN/default-Internet interface. If no
AP-client address is available, it prints join/restart instructions and diagnostic
URLs. Existing `PHONE_IP` / `--phone-ip` overrides remain optional; leave them unset.

## Troubleshooting

- **AP does not appear:** upload the updated sketch, verify board power, wait for
  startup, and read Serial Monitor at 115200. An AP failure message may require
  checking/updating the UNO's connectivity-module firmware using Arduino IDE.
- **PC or phone leaves Wi-Fi:** keep the no-Internet connection and disable automatic
  switching away from this network if the device keeps dropping it.
- **No PC address in the AP subnet:** connect the PC to the robot AP before launching
  Python. Stop/restart the script after changing Wi-Fi so URLs/certificates reflect
  the assigned interface. Resolve a conflicting `192.168.4.0/24` VPN/other network.
- **Phone page works on PC but not phone:** confirm both devices are on the robot AP.
  On Windows, mark this trusted robot network Private and allow this project's
  Python/PhoneSense inbound HTTPS port 8080 on that profile.
- **No Internet for model downloads:** download weights/datasets beforehand, or use
  a separate Internet interface on the PC. Local `.pt` inference works offline.
- **Changing the network settings:** SSID/password live at the top of the `.ino`.
  If changing the AP address/subnet or SSID, update `robotics_lab/network.py` to match.

## Verification

Compiled locally using Arduino UNO R4 Boards **1.6.0**, target
`arduino:renesas_uno:unor4wifi`, and the installed Telemetrix libraries. NewPing's
architecture metadata warns about renesas_uno, but this sketch builds successfully.
Python network tests cover matching firmware/client constants, all seven clients,
PC-interface selection, changing DHCP leases, invalid addresses and no-data route
queries. Compilation and tests do not upload firmware or verify physical Wi-Fi/video.
