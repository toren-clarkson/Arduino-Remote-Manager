"""Workbench API/storage validation. No Arduino or live camera connection."""

import io
import json
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from missions.stop_and_stare.config import Config
from missions.stop_and_stare.controller import Planner, Sighting
from missions.stop_and_stare.run import Runner, StopRequested
from webapp.app import create_app
from webapp.jobs import config_values
from webapp.storage import extract_dataset, training_yaml


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for folder in ('models', 'datasets', 'assets/tags', 'runs/verification', 'missions', 'webapp'):
            (self.root/folder).mkdir(parents=True, exist_ok=True)
        (self.root/'README.md').write_text('# Test project', encoding='utf-8')
        (self.root/'models/example.pt').write_bytes(b'not-real-weights')
        self.app = create_app(self.root, testing=True)
        self.client = self.app.test_client()
        self.token = self.client.get('/api/session').json['csrf']
        self.headers = {'X-CSRF-Token': self.token}
        self.jobs = self.app.extensions['jobs']
        self.store = self.app.extensions['store']

    def tearDown(self):
        self.jobs.close()
        for process in list(self.jobs.processes.values()):
            process.wait(timeout=15)
        # Monitor threads persist job.json after process exit.
        deadline = time.monotonic()+3
        while self.jobs.processes and time.monotonic() < deadline:
            time.sleep(.01)
        self.temp.cleanup()

    def post(self, path, data):
        return self.client.post(path, json=data, headers=self.headers)

    def test_all_page_shells_and_catalog(self):
        for path in ('/', '/datasets', '/models', '/training', '/tags', '/missions', '/jobs', '/files', '/settings'):
            self.assertEqual(self.client.get(path).status_code, 200, path)
        data = self.client.get('/api/state').json
        self.assertEqual(data['models'][0]['path'], 'models/example.pt')
        self.assertGreaterEqual(len(data['catalog']['trackers']), 3)
        self.assertIn('task', {f['key'] for f in data['catalog']['jobs']['train']['fields']})

    def test_csrf_origin_and_host(self):
        self.assertEqual(self.client.post('/api/folders', json={'name':'x'}).status_code, 403)
        self.assertEqual(self.client.post('/api/folders', json={'name':'x'}, headers={**self.headers,'Origin':'http://evil.example'}).status_code, 403)
        self.assertEqual(self.client.get('/api/state', headers={'Host':'evil.example'}).status_code, 403)

    def test_path_and_secret_protection(self):
        for path in ('../README.md', '.env', '.control/jobs', 'C:/Windows', 'datasets/../../README.md', 'datasets\\..\\.env', '/README.md', 'assets/key.pem'):
            self.assertEqual(self.client.get('/api/file', query_string={'path':path}).status_code, 400, path)

    def test_revision_conflict_and_backup(self):
        original = self.client.get('/api/text?path=README.md').json
        result = self.client.put('/api/text', json={**original, 'text':'# Updated'}, headers=self.headers)
        self.assertEqual(result.status_code, 200)
        conflict = self.client.put('/api/text', json={**original, 'text':'# Stale'}, headers=self.headers)
        self.assertEqual(conflict.status_code, 400)
        self.assertEqual((self.root/'README.md').read_text(), '# Updated')
        self.assertEqual(len(list((self.store.state/'history').iterdir())), 1)

    def test_file_crud_and_restore(self):
        self.assertEqual(self.post('/api/folders', {'parent':'datasets', 'name':'collection'}).status_code, 200)
        self.assertEqual(self.post('/api/new-file', {'parent':'datasets/collection', 'name':'notes.md'}).status_code, 200)
        self.assertEqual(self.post('/api/rename', {'path':'datasets/collection/notes.md','name':'renamed.md'}).status_code, 200)
        ticket = self.post('/api/trash', {'path':'datasets/collection/renamed.md'}).json['id']
        self.assertFalse((self.root/'datasets/collection/renamed.md').exists())
        self.assertEqual(self.post('/api/restore', {'id':ticket}).status_code, 200)
        self.assertTrue((self.root/'datasets/collection/renamed.md').exists())
        self.assertEqual(self.post('/api/trash', {'path':'models'}).status_code, 400)

    def test_upload_no_overwrite_and_traversal(self):
        result = self.client.post('/api/upload', data={'folder':'datasets/photos', 'files':(io.BytesIO(b'image'), 'test.jpg')}, headers=self.headers)
        self.assertEqual(result.status_code, 200)
        result = self.client.post('/api/upload', data={'folder':'datasets/photos', 'files':(io.BytesIO(b'replace'), 'test.jpg')}, headers=self.headers)
        self.assertEqual(result.status_code, 400)
        result = self.client.post('/api/upload', data={'folder':'datasets/photos', 'files':(io.BytesIO(b'x'), '../escape.py')}, headers=self.headers)
        self.assertEqual(result.status_code, 400)
        self.assertEqual((self.root/'datasets/photos/test.jpg').read_bytes(), b'image')

    def test_html_is_downloaded_not_executed(self):
        (self.root/'datasets/test.html').write_text('<script>bad()</script>')
        response = self.client.get('/api/file?path=datasets/test.html&inline=1')
        self.assertIn('attachment', response.headers['Content-Disposition'])
        response.close()

    def test_model_metadata_rename(self):
        self.assertEqual(self.client.put('/api/models', json={'path':'models/example.pt','label':'Cup detector','notes':'v2','provider':'ultralytics'}, headers=self.headers).status_code, 200)
        self.post('/api/rename', {'path':'models/example.pt','name':'cup.pt'})
        model = self.client.get('/api/state').json['models'][0]
        self.assertEqual(model['label'], 'Cup detector')
        self.assertEqual(model['path'], 'models/cup.pt')

    def test_profiles_validate_and_roundtrip(self):
        self.assertEqual(self.post('/api/profiles', {'name':'Tag run','config':{'mission_mode':'tag_only','tracker':'none'}}).status_code, 200)
        profiles = self.client.get('/api/state').json['profiles']
        self.assertEqual(profiles[0]['config']['mission_mode'], 'tag_only')
        for config in ({'settle_seconds':-1}, {'tag_id':587}, {'tracker':'../custom.yaml'}, {'left_pins':[7,4,4]}, {'observe_frames':2.5}, {'arduino_ip':'not-an-ip'}, {'extra':1}):
            self.assertEqual(self.post('/api/profiles', {'name':'bad','config':config}).status_code, 400, config)

    def test_job_validation_rejects_arbitrary_commands(self):
        self.assertEqual(self.post('/api/jobs', {'kind':'shell','params':{'command':'whoami'}}).status_code, 400)
        self.assertEqual(self.post('/api/jobs', {'kind':'tag_generate','params':{'tag_id':0,'output':'tag.png','command':'whoami'}}).status_code, 400)
        self.assertEqual(self.post('/api/jobs', {'kind':'mission','params':{'drive':True,'video':'runs/verification/no.avi'}}).status_code, 400)

    def test_new_job_schema_with_distinct_inputs_and_formats(self):
        from webapp.catalog import JOBS, field
        custom = dict(label='Custom offline mission', group='missions', handler='example_handler',
                      resources=['custom_resource'], outputs=['csv','custom_json'],
                      fields=[field('duration','Duration','number',.5,min=.1,max=10),
                              field('route','Route','select','triangle',options=['triangle','square'])])
        with patch.dict(JOBS, {'custom_mission':custom}):
            catalog=self.client.get('/api/state').json['catalog']['jobs']['custom_mission']
            self.assertEqual(catalog['outputs'],['csv','custom_json'])
            values=self.jobs.prepare('custom_mission',{'duration':.25,'route':'square'},self.store.state)
            self.assertEqual(values,{'duration':.25,'route':'square'})
            with self.assertRaises(ValueError):
                self.jobs.prepare('custom_mission',{'duration':.25,'route':'unknown'},self.store.state)

    def test_hardware_plugin_must_declare_cooperative_stop(self):
        from webapp.catalog import JOBS
        with patch.dict(JOBS, {'bad_hardware':dict(label='Unsafe plugin',group='missions',handler='x',resources=['camera_robot'],outputs=[],fields=[])}):
            self.assertEqual(self.post('/api/jobs',{'kind':'bad_hardware','params':{}}).status_code,400)

    def test_job_resources_and_edit_lock(self):
        self.jobs.records['busy'] = dict(id='busy',kind='mission',params={},status='running',label='Robot preview',created='2026')
        self.assertEqual(self.post('/api/jobs', {'kind':'inspect_model','params':{'model':'models/example.pt'}}).status_code, 400)
        self.assertEqual(self.post('/api/rename', {'path':'README.md','name':'renamed.md'}).status_code, 400)
        self.assertEqual(self.post('/api/trash', {'path':'models/example.pt'}).status_code, 400)
        self.jobs.records.clear()

    def test_mission_stop_does_not_terminate_process(self):
        from unittest.mock import Mock
        ticket = 'a'*32
        directory = self.store.state/'jobs'/ticket
        directory.mkdir()
        self.jobs.records[ticket] = dict(id=ticket,kind='mission',params={},status='running',label='Mission',created='2026')
        process = Mock()
        self.jobs.processes[ticket] = process
        self.assertEqual(self.post(f'/api/jobs/{ticket}/stop', {}).status_code, 200)
        self.assertTrue((directory/'stop').exists())
        process.terminate.assert_not_called()
        self.jobs.records.clear()
        self.jobs.processes.clear()

    def test_stop_marker_checked_before_camera_or_motion(self):
        runner = object.__new__(Runner)
        runner.stop_file = self.root/'stop'
        runner.stop_file.touch()
        with self.assertRaises(StopRequested):
            runner.poll('MOVE')

    def test_tag_only_and_object_only_planners(self):
        tag = Planner(Config(mission_mode='tag_only'))
        self.assertEqual(tag.leg, 'tag')
        self.assertEqual(tag.plan(Sighting('tag','tag:0',0,.4),1).name, 'done')
        obj = Planner(Config(mission_mode='object_only'))
        self.assertEqual(obj.plan(Sighting('object','bottle',0,.4),1).name, 'done')

    def test_local_dataset_yaml_and_download_rejection(self):
        folder=self.root/'datasets/example'
        for split in ('train','valid'):
            (folder/split/'images').mkdir(parents=True)
        yaml=folder/'data.yaml'
        yaml.write_text('train: train/images\nval: valid/images\nnames: [cup]\n')
        normalized=training_yaml(self.store,'datasets/example/data.yaml',self.root/'validated.yaml')
        self.assertTrue(Path(normalized['train']).is_absolute())
        yaml.write_text('train: train/images\nval: valid/images\nnames: [cup]\ndownload: echo bad\n')
        with self.assertRaises(ValueError):
            training_yaml(self.store,'datasets/example/data.yaml',self.root/'validated.yaml')

    def test_dataset_zip_traversal_and_format(self):
        for filename in ('../escape.txt','/absolute.txt','safe/../../escape.txt','safe/train.py','safe/CON.txt'):
            archive=self.root/'bad.zip'
            with zipfile.ZipFile(archive,'w') as handle:
                handle.writestr(filename,'bad')
            with self.assertRaises(ValueError, msg=filename):
                extract_dataset(archive,self.root/'datasets/import')
            self.assertFalse((self.root/'datasets/import').exists())
        with zipfile.ZipFile(self.root/'good.zip','w') as handle:
            handle.writestr('data.yaml','names: [cup]')
            handle.writestr('train/labels/1.txt','0 .5 .5 .2 .2')
        extract_dataset(self.root/'good.zip',self.root/'datasets/import')
        self.assertTrue((self.root/'datasets/import/data.yaml').exists())

    def test_roboflow_metadata_and_export_paths_are_normalized_without_editing_source(self):
        import yaml
        folder=self.root/'datasets/red-ball'
        for split in ('train','valid','test'):
            (folder/split/'images').mkdir(parents=True)
        source=folder/'data.yaml'
        original=('train: ../train/images\nval: ../valid/images\ntest: ../test/images\n'
                  'nc: 1\nnames: [Red-Ball]\nroboflow:\n  workspace: example\n'
                  '  project: red-ball\n  version: 1\n  url: https://app.roboflow.com/example/red-ball/1\n')
        source.write_text(original,encoding='utf-8')
        output=self.store.state/'normalized.yaml'
        data=training_yaml(self.store,'datasets/red-ball/data.yaml',output)
        self.assertNotIn('roboflow',data)
        self.assertEqual(data['names'],['Red-Ball'])
        for key,split in [('train','train'),('val','valid'),('test','test')]:
            self.assertEqual(Path(data[key]),folder/split/'images')
        self.assertEqual(source.read_text(encoding='utf-8'),original)
        self.assertEqual(yaml.safe_load(output.read_text(encoding='utf-8')),data)

    def test_existing_sibling_paths_and_split_lists_are_preserved(self):
        folder=self.root/'datasets/configs'
        folder.mkdir()
        for split in ('train','valid'):
            (self.root/'datasets'/split/'images').mkdir(parents=True)
        (folder/'data.yaml').write_text('train: [../train/images]\nval: ../valid/images\nnames: [cup]\n')
        data=training_yaml(self.store,'datasets/configs/data.yaml',self.store.state/'normalized.yaml')
        self.assertEqual(data['train'],[str(self.root/'datasets/train/images')])
        self.assertEqual(data['val'],str(self.root/'datasets/valid/images'))

    def test_export_path_fallback_cannot_escape_or_override_explicit_root(self):
        folder=self.root/'datasets/example'
        (folder/'train/images').mkdir(parents=True)
        (folder/'valid/images').mkdir(parents=True)
        source=folder/'data.yaml'
        for paths in ('train: ../../train/images\nval: valid/images\n',
                      'path: .\ntrain: ../train/images\nval: valid/images\n'):
            source.write_text(paths+'names: [cup]\n')
            with self.assertRaises(ValueError):
                training_yaml(self.store,'datasets/example/data.yaml',self.store.state/'normalized.yaml')

    def test_unsupported_yaml_setting_error_identifies_the_setting(self):
        source=self.root/'datasets/data.yaml'
        source.write_text('names: [cup]\ndownload: echo bad\n')
        with self.assertRaisesRegex(ValueError,'setting\\(s\\): download'):
            training_yaml(self.store,'datasets/data.yaml',self.store.state/'normalized.yaml')

    def test_real_marker_job_and_artifacts(self):
        response=self.post('/api/jobs',{'kind':'tag_generate','params':{'tag_id':17,'output':'marker17.png'}})
        self.assertEqual(response.status_code,201,response.json)
        ticket=response.json['id']
        deadline=time.monotonic()+30
        while self.jobs.records[ticket]['status'] in {'running','starting'} and time.monotonic()<deadline:
            time.sleep(.05)
        detail=self.client.get(f'/api/jobs/{ticket}').json
        self.assertEqual(detail['status'],'succeeded',detail.get('log'))
        self.assertTrue((self.root/'assets/tags/marker17.png').is_file())
        self.assertIn('result.json',detail['artifacts'])
        self.assertEqual(self.client.get(f'/api/jobs/{ticket}/artifact?name=../../README.md').status_code,400)


if __name__=='__main__':
    unittest.main()
