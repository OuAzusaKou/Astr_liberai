import unittest
import argparse
import ast
import contextlib
import io
from types import SimpleNamespace

from astra.runtime import fail_closed_eval_source, strict_eval_cli_source
from astra.config import SOURCE


class RuntimeTests(unittest.TestCase):
    def test_exception_does_not_retry_or_reset_another_layout(self):
        source = '''
def run(env):
    for layout in range(3):
        try:
            env.reset(layout)
            env.run_eval()
        except TimeoutError:
            continue
        except Exception:
            continue
'''
        namespace = {}
        exec(compile(fail_closed_eval_source(source), '<local-runtime>', 'exec'), namespace)
        for error in (TimeoutError('supervisor'), RuntimeError('agent_aborted')):
            resets = []
            def fail():
                raise error
            env = SimpleNamespace(reset=resets.append, run_eval=fail)
            with self.assertRaises(type(error)):
                namespace['run'](env)
            self.assertEqual(resets, [0])

    def test_unknown_entrypoint_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            fail_closed_eval_source('print("changed upstream")')

    def test_real_entrypoint_device_flags_survive_isaac_preliminary_parse(self):
        source = (SOURCE / 'src/eval_client/main.py').read_text()
        for patched in (False, True):
            tree = ast.parse(strict_eval_cli_source(source) if patched else source)
            statements = []
            for node in tree.body:
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                    func = node.value.func
                    if isinstance(func, ast.Attribute) and func.attr == 'add_app_launcher_args':
                        break
                    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == 'parser':
                        statements.append(node)
                elif isinstance(node, ast.Assign):
                    if any((isinstance(t, ast.Name) and t.id == 'parser') or
                           (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == 'parser')
                           for t in node.targets):
                        statements.append(node)
            for gpu in range(8):
                namespace = {'argparse': argparse}
                exec(compile(ast.Module(body=statements, type_ignores=[]), '<actual-eval-parser>', 'exec'), namespace)
                parser = namespace['parser']
                argv = ['--task_name', 'plug_in_charger', '--env_cfg_type', 'arx_x5',
                        '--device_id', str(gpu), '--device', f'cuda:{gpu}',
                        '--policy_name', 'AstraImageWAM', '--port', '40000',
                        '--additional_info', 'regression', '--seed', '0']
                if not patched:
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        parser.parse_known_args(argv)
                else:
                    preliminary, unknown = parser.parse_known_args(argv)
                    self.assertEqual(preliminary.device_id, gpu)
                    self.assertEqual(unknown, ['--device', f'cuda:{gpu}'])
                    parser.add_argument('--device')
                    final = parser.parse_args(argv)
                    self.assertEqual((final.device_id, final.device), (gpu, f'cuda:{gpu}'))
