"""Fixed robot AP endpoint and selection of the PC's PhoneSense address.

The Arduino is always .1. The PC and phone obtain their own DHCP addresses;
the phone connects to PhoneSense on the PC, not to the Arduino's TCP server.
No socket or network service is opened merely by importing this module.
"""

import ipaddress
import socket
from urllib.parse import urlsplit

AP_SSID = 'UNO-R4-Robot'
ARDUINO_IP = '192.168.4.1'
TELEMETRIX_PORT = 31336
AP_NETWORK = ipaddress.IPv4Network('192.168.4.0/24')


def pc_address_on_ap(addresses, route_ip=None):
    """Choose a local AP-client address; prefer the OS route toward the board."""
    candidates = []
    for value in addresses:
        try:
            address = ipaddress.IPv4Address(value)
        except (ValueError,TypeError):
            continue
        if (address in AP_NETWORK and str(address) != ARDUINO_IP
                and address not in {AP_NETWORK.network_address,AP_NETWORK.broadcast_address}):
            if str(address) not in candidates:
                candidates.append(str(address))
    if route_ip in candidates:
        return route_ip
    return candidates[0] if candidates else None


def route_address():
    """Ask the local routing table; UDP connect alone sends no packet to the robot."""
    try:
        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as probe:
            probe.connect((ARDUINO_IP,TELEMETRIX_PORT))
            return probe.getsockname()[0]
    except OSError:
        return None


def configure_phone_camera(camera, preferred_ip=None):
    """Select/print the PhoneSense PC URL on the robot AP, once at startup."""
    urls = getattr(camera,'phone_urls',None)
    if urls is None:
        url = getattr(camera,'phone_url',None)
        urls = [url] if url else []
    urls = list(urls)
    addresses = getattr(camera,'lan_ips',None)
    if addresses is None:
        addresses = [urlsplit(url).hostname for url in urls]
    selected = preferred_ip or pc_address_on_ap(addresses,route_address())
    if selected:
        camera.select_ip(selected)
        urls = [camera.phone_url]
    elif getattr(camera,'lan_ips',None) is not None:
        print(f'No PC address on {AP_NETWORK}. Join {AP_SSID} on the PC and phone, '
              'then restart the script. Other available URLs are listed for diagnosis.',flush=True)
    print(f'Robot Wi-Fi: {AP_SSID} | Telemetrix: {ARDUINO_IP}:{TELEMETRIX_PORT}',flush=True)
    print('PhoneSense PC camera URL:' if selected else 'PhoneSense camera URLs:',flush=True)
    for url in urls:
        print(' ',url,flush=True)
    print('On the phone: stay on the robot Wi-Fi despite no Internet, open the PC URL, '
          'Connect, then Start camera. Keep this terminal running.',flush=True)
    return selected
