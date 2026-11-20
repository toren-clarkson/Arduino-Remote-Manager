"""Console selection and settings persistence without connecting hardware."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robotics_lab import console


class ConsoleTests(unittest.TestCase):
    def test_choice_reprompts_invalid_and_accepts_enter_default(self):
        with patch('builtins.input',side_effect=['0','bad','']),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(console.choose('mode',['preview','drive']),0)

    def test_rejects_pin_conflicts_and_non_pwm_enable(self):
        self.assertEqual(console.pins('7,4,5'),(7,4,5))
        for raw in ('1,1,5','7,4,2','1,2','4,5,99'):
            with self.assertRaises(ValueError):console.pins(raw)

    def test_setup_only_changes_requested_process_and_saves_next_run_defaults(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(console,'PROJECT_ROOT',Path(directory)), \
             patch.object(sys,'argv',['test_wifi_led.py','--setup']), \
             patch('builtins.input',side_effect=['3','.2']),contextlib.redirect_stdout(io.StringIO()):
            namespace={'__file__':'test_wifi_led.py','BLINK_COUNT':10,'BLINK_SECONDS':.5}
            console.configure_script(namespace)
            self.assertEqual(namespace['BLINK_COUNT'],3)
            self.assertEqual(namespace['BLINK_SECONDS'],.2)
            self.assertNotIn('--setup',sys.argv)
            self.assertTrue((Path(directory)/'calibrations/console/test_wifi_led.json').is_file())


if __name__=='__main__':unittest.main()
