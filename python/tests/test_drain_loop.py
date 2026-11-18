import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lib.loop import LoopSettings


class DrainLoopTests(unittest.TestCase):
    def test_partial_resume_rejects_changed_settings_before_connecting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "games_0.checkpoint.json").write_text("{}")
            (root / "drain-config.json").write_text("{}")
            fixed = SimpleNamespace(
                game=SimpleNamespace(name="ttt"), start_pos="default", simulations_per_gen=50,
                search_batch_size=16, eval_random_symmetries=True,
            )
            settings = SimpleNamespace(
                fixed_settings=fixed, selfplay_path=str(root), muzero=False,
                selfplay_settings=SimpleNamespace(as_dict=lambda: {"search": 1}),
            )
            gen = SimpleNamespace(simulations_path=str(root / "games_0"))
            with patch("lib.loop.SelfplayClient") as client:
                with self.assertRaisesRegex(ValueError, "Selfplay settings differ"):
                    LoopSettings.run_loop_inner(settings, gen, None, None, None, None)
            client.assert_not_called()

    def test_partial_generation_zero_reuses_initial_network(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "selfplay").mkdir()
            initial_calls = []
            network = SimpleNamespace(to=lambda _: None)
            settings = SimpleNamespace(
                fixed_settings=SimpleNamespace(game=SimpleNamespace(name="ttt")),
                max_buffer_size=100, test_fraction=0.1, initial_data_files=[],
                selfplay_path=str(root / "selfplay"), training_path=str(root / "training"),
                initial_network=lambda: initial_calls.append(1) or network,
            )
            buffer = SimpleNamespace(files=[], simulation_count=0, position_count=0)
            def save(_network, path):
                Path(path).write_text("initial")
            with patch("lib.loop.LoopBuffer", return_value=buffer), \
                    patch("lib.loop.Logger"), \
                    patch("lib.loop.torch.jit.script", side_effect=lambda n: n), \
                    patch("lib.loop.torch.jit.save", side_effect=save), \
                    patch("lib.loop.torch.jit.load", return_value=network) as load:
                LoopSettings.load_start_state(settings)
                (root / "selfplay/games_0.checkpoint.json").touch()
                LoopSettings.load_start_state(settings)
            self.assertEqual(len(initial_calls), 1)
            load.assert_called_once_with(str(root / "training/gen_0/initial_network.pt"))

    def test_complete_generation_is_trained_before_drain_acknowledgement(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            selfplay = root / "selfplay"
            selfplay.mkdir()
            flag = root / "drain_requested"
            done = root / "drain_done"
            events = []
            drain_sent = threading.Event()

            class Client:
                def __init__(self, _port):
                    self.count = 0

                def send_startup_settings(self, _): pass
                def send_new_settings(self, _): pass
                def send_new_network(self, _): events.append("network")
                def send_dummy_network(self): events.append("dummy")
                def send_wait_for_new_network(self): events.append("wait")
                def send_drain(self):
                    events.append("drain")
                    drain_sent.set()
                def send_stop(self): events.append("stop")
                def wait_for_update(self):
                    self.count += 1
                    if self.count == 1:
                        return {"FinishedFile": {"index": 0}}
                    if self.count == 2:
                        self.assert_drain()
                        return {"Drained": {"index": 1, "games": 3}}
                    return "Stopped"
                def assert_drain(self):
                    if not drain_sent.wait(3):
                        raise AssertionError("drain command not sent")

            class Network:
                def parameters(self): return []

            class Logger:
                def start_batch(self): pass
                def log(self, *_): pass
                def save(self, _): events.append("saved")

            class Buffer:
                position_count = 10
                simulation_count = 2
                def append(self, *_): events.append("append")

            class Sampler:
                group = SimpleNamespace(positions=[1])
                def next_batch_either(self): return object()
                def close(self): pass

            class Training:
                def train_step(self, *_):
                    flag.touch()
                    if not drain_sent.wait(3):
                        raise AssertionError("drain request not sent during optimization")
                    events.append("trained")

            class Fixed:
                game = SimpleNamespace(name="ttt")
                start_pos = "default"
                simulations_per_gen = 50
                search_batch_size = 16
                eval_random_symmetries = True
                max_batch_wait_ms = 100
                def to_startup(self, **_): return {}

            class Settings:
                def as_dict(self): return {"search": 1}

            def gen(index):
                path = root / "training" / f"gen_{index}"
                return SimpleNamespace(
                    gi=index, train_path=str(path), settings_path=str(path / "settings.json"),
                    simulations_path=str(selfplay / f"games_{index}"),
                    network_path_pt=str(path / "network.pt"), network_path_onnx=str(path / "network.onnx"),
                    finished_path=str(path / "finished.txt"), cleanup_old_pt_files=lambda *a, **k: None,
                )

            def export(_network, name):
                path = root / f"{name}.onnx"
                path.write_bytes(b"onnx")
                return str(path)

            settings = SimpleNamespace(
                fixed_settings=Fixed(), selfplay_settings=Settings(), selfplay_path=str(selfplay),
                muzero=False, only_generate=False, target_gen=0, optimizer=lambda _: object(),
                min_buffer_size=1, wait_for_new_network=True, dummy_network=None, port=1,
                log_path=str(root / "log.npz"), train_settings=Training(),
                save_tmp_onnx_network=export, evaluate_network=lambda *_: None,
                sampler=lambda *a, **k: Sampler(), calc_batch_count_per_gen=lambda *a: 1.0,
            )
            network = Network()
            with patch.dict(os.environ, {"DRAIN_FLAG_PATH": str(flag), "DRAIN_DONE_PATH": str(done)}), \
                    patch("lib.loop.SelfplayClient", Client), \
                    patch("lib.loop.Generation.from_gi", side_effect=lambda _, index: gen(index)), \
                    patch("lib.loop.DataFile.open", return_value=object()), \
                    patch("lib.loop.print_param_count"), \
                    patch("lib.loop.torch.jit.save"), \
                    patch("lib.loop.torch.jit.load", return_value=network):
                LoopSettings.run_loop_inner(settings, gen(0), Buffer(), Logger(), None, network)

            self.assertTrue(done.exists())
            self.assertTrue((root / "training/gen_0/finished.txt").exists())
            self.assertTrue((selfplay / "drain-config.json").exists())
            self.assertLess(events.index("wait"), events.index("drain"))
            self.assertLess(events.index("drain"), events.index("trained"))
            self.assertLess(events.index("trained"), events.index("saved"))
            self.assertLess(events.index("saved"), events.index("stop"))


if __name__ == "__main__":
    unittest.main()
