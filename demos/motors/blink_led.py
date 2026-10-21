"""Blink the Uno R4's built-in L LED over Telemetrix Wi-Fi; no motor commands."""

import asyncio

from telemetrix_uno_r4.wifi.telemetrix_uno_r4_wifi_aio.telemetrix_uno_r4_wifi_aio import TelemetrixUnoR4WiFiAio


# Edit this to match the IP shown by the Arduino Serial Monitor / LED matrix.
ARDUINO_IP = "10.13.96.68"
ARDUINO_PORT = 31336
LED_PIN = 13
BLINK_COUNT = 10
BLINK_SECONDS = 0.5


async def main() -> None:
    print(f"Connecting to Arduino at {ARDUINO_IP}:{ARDUINO_PORT}...", flush=True)
    board = TelemetrixUnoR4WiFiAio(
        transport_address=ARDUINO_IP,
        ip_port=ARDUINO_PORT,
        autostart=False,
        loop=asyncio.get_running_loop(),
        shutdown_on_exception=False,
        close_loop_on_shutdown=False,
        hard_reset_on_shutdown=False,
    )
    led_configured = False
    connected = False
    try:
        await asyncio.wait_for(board.start_aio(), timeout=15)
        connected = True
        await board.disable_all_reporting()
        await board.set_pin_mode_digital_output(LED_PIN)
        led_configured = True
        for blink in range(1, BLINK_COUNT + 1):
            await board.digital_write(LED_PIN, 1)
            print(f"{blink}/{BLINK_COUNT}: L LED ON (command sent)", flush=True)
            await asyncio.sleep(BLINK_SECONDS)
            await board.digital_write(LED_PIN, 0)
            print(f"{blink}/{BLINK_COUNT}: L LED OFF (command sent)", flush=True)
            await asyncio.sleep(BLINK_SECONDS)
        print("Finished. Confirm that the physical L LED blinked ten times.")
    finally:
        try:
            if led_configured:
                await board.digital_write(LED_PIN, 0)
            if connected:
                await board.shutdown()
        finally:
            # Installed Telemetrix shutdown() does not close its Wi-Fi writer.
            if board.sock is not None and board.sock.writer is not None:
                board.sock.writer.close()
                await board.sock.writer.wait_closed()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stopped.")
    except (TimeoutError, OSError, RuntimeError) as exc:
        raise SystemExit(
            f"Wi-Fi LED test failed: {exc or type(exc).__name__}. "
            "Check ARDUINO_IP and the Telemetrix firmware. "
            "Close other Python scripts connected to the board before retrying."
        ) from exc
