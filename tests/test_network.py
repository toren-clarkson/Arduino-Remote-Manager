"""Robot AP addressing and PhoneSense interface selection; no hardware contacted."""

import ast
import contextlib
import io
import re
import socket
import unittest
from pathlib import Path
from unittest.mock import MagicMock,patch

from robotics_lab import network


class Camera:
    def __init__(self,addresses):
        self.lan_ips=addresses
        self.lan_ip=addresses[0]

    @property
    def phone_urls(self):
        return [f'https://{ip}:8080/phone' for ip in self.lan_ips]

    @property
    def phone_url(self):
        return f'https://{self.lan_ip}:8080/phone'

    def select_ip(self,ip):
        if ip not in self.lan_ips:raise ValueError('Address is not assigned to this PC')
        self.lan_ip=ip


class NetworkTests(unittest.TestCase):
    def test_chooses_ap_even_when_internet_or_vpn_is_default(self):
        self.assertEqual(network.pc_address_on_ap(['10.1.2.3','192.168.4.12','172.18.0.1'],'10.1.2.3'),'192.168.4.12')

    def test_prefers_route_among_multiple_ap_addresses(self):
        self.assertEqual(network.pc_address_on_ap(['192.168.4.2','192.168.4.7'],'192.168.4.7'),'192.168.4.7')

    def test_never_uses_board_network_broadcast_or_other_subnets_as_pc(self):
        values=['192.168.4.1','192.168.4.0','192.168.4.255','192.168.137.2','127.0.0.1','::1',None,'bad']
        self.assertIsNone(network.pc_address_on_ap(values,'192.168.4.1'))

    def test_prints_actual_pc_url_and_handles_dhcp_address_changes(self):
        for assigned in ('192.168.4.2','192.168.4.9'):
            camera=Camera(['10.0.0.10',assigned])
            with patch.object(network,'route_address',return_value=assigned),contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(network.configure_phone_camera(camera),assigned)
            self.assertIn(f'https://{assigned}:8080/phone',output.getvalue())
            self.assertNotIn('https://192.168.4.1',output.getvalue())
            self.assertNotIn('https://10.0.0.10',output.getvalue())

    def test_missing_ap_does_not_advertise_board_as_phone_server(self):
        camera=Camera(['10.0.0.2'])
        with patch.object(network,'route_address',return_value='10.0.0.2'),contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertIsNone(network.configure_phone_camera(camera))
        self.assertIn('Join UNO-R4-Robot',output.getvalue())
        self.assertNotIn('https://192.168.4.1',output.getvalue())

    def test_explicit_pc_override_is_validated_by_phonesense(self):
        camera=Camera(['10.0.0.2','192.168.4.3'])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(network.configure_phone_camera(camera,'192.168.4.3'),'192.168.4.3')
            with self.assertRaises(ValueError):network.configure_phone_camera(camera,'192.168.4.99')

    def test_route_lookup_sends_no_data_or_tcp_probe(self):
        probe=MagicMock();probe.getsockname.return_value=('192.168.4.2',12345)
        with patch.object(network.socket,'socket') as create:
            create.return_value.__enter__.return_value=probe
            self.assertEqual(network.route_address(),'192.168.4.2')
        create.assert_called_once_with(socket.AF_INET,socket.SOCK_DGRAM)
        probe.connect.assert_called_once_with((network.ARDUINO_IP,network.TELEMETRIX_PORT))
        probe.send.assert_not_called();probe.sendto.assert_not_called()

    def test_route_failure_falls_back_to_known_local_addresses(self):
        with patch.object(network.socket,'socket',side_effect=OSError('no route')):
            self.assertIsNone(network.route_address())

    def test_firmware_and_every_client_share_endpoint_and_preserve_ap_on_shutdown(self):
        root=Path(__file__).resolve().parents[1]
        source=(root/'Telemetrix4UnoR4WIFI/Telemetrix4UnoR4WIFI.ino').read_text()
        octets=re.search(r'ROBOT_AP_IP\(([^)]+)\)',source)[1]
        self.assertEqual('.'.join(part.strip() for part in octets.split(',')),network.ARDUINO_IP)
        self.assertEqual(int(re.search(r'const uint16_t PORT = (\d+)',source)[1]),network.TELEMETRIX_PORT)
        self.assertEqual(re.search(r'AP_SSID = "([^"]+)"',source)[1],network.AP_SSID)
        self.assertIn('WiFi.beginAP(AP_SSID, AP_PASSWORD)',source)
        self.assertNotIn('WiFi.begin(',source)
        self.assertNotIn('board_hard_reset();',source[source.index('void loop()'):])
        files=['default-opencv-tracking.py','default-yolo-tracking.py','simple-april-tag.py',
               'robust-april-tag.py','dual_motor_template.py','test_wifi_led.py','stop-and-stare.py','adaptive-homing.py']
        for filename in files:
            tree=ast.parse((root/filename).read_text())
            imports=[node for node in ast.walk(tree) if isinstance(node,ast.ImportFrom) and node.module=='robotics_lab.network']
            self.assertTrue(any('ARDUINO_IP' in {n.name for n in node.names} for node in imports),filename)
            clients=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)
                     and node.func.id=='TelemetrixUnoR4WiFiAio']
            self.assertEqual(len(clients),1,filename)
            kwargs={kw.arg:kw.value for kw in clients[0].keywords}
            self.assertIn('ip_port',kwargs,filename)
            self.assertIs(kwargs['hard_reset_on_shutdown'].value,False,filename)


if __name__=='__main__':unittest.main()
