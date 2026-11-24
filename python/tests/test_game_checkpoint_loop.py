import json
import os
import socket
import tempfile
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from lib.loop import LoopSettings


def loop_settings(root: Path, **overrides):
    settings = SimpleNamespace(
        fixed_settings=SimpleNamespace(
            game=SimpleNamespace(name="oriflambe-repetition"), start_pos="default", simulations_per_gen=50,
            to_startup=lambda **_: SimpleNamespace(as_dict=lambda: {}),
        ),
        selfplay_settings=SimpleNamespace(as_dict=lambda: {"max_game_length": 450}),
        selfplay_path=str(root / "selfplay"), muzero=False, only_generate=False, target_gen=0, port=0,
        **overrides,
    )
    settings.game_checkpoint_path = LoopSettings.game_checkpoint_path.fget(settings)
    settings.game_checkpoint_config_path = LoopSettings.game_checkpoint_config_path.fget(settings)
    for name in ["game_checkpoint_config", "check_game_checkpoint_resumable", "request_game_checkpoint"]:
        setattr(settings, name, types.MethodType(getattr(LoopSettings, name), settings))
    return settings


class GameCheckpointLoopTests(unittest.TestCase):
    def test_checkpoint_during_optimization_saves_generation_and_sends_nothing_after(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "selfplay").mkdir()
            flag, done = root / "checkpoint_requested", root / "checkpoint_done"
            server, client_end = socket.socketpair()
            commands = []
            checkpoint_received = threading.Event()

            def fake_server():
                with server, server.makefile("r") as reader:
                    def read():
                        commands.append(json.loads(reader.readline()))
                        return commands[-1]
                    while read() != "UseDummyNetwork":
                        pass
                    server.sendall(b'{"FinishedFile": {"index": 0}}\n')
                    while read() != "Checkpoint":
                        pass
                    checkpoint_received.set()
                    server.sendall(b'{"Checkpointed": {"index": 1, "games": 3, "active": 5}}\n')

            threading.Thread(target=fake_server, daemon=True).start()

            def train_step(*_):
                # request the checkpoint while optimization is in progress
                flag.touch()
                self.assertTrue(checkpoint_received.wait(5), "checkpoint was not sent during optimization")

            def gen(index):
                path = root / "training" / f"gen_{index}"
                return SimpleNamespace(
                    gi=index, train_path=str(path), settings_path=str(path / "settings.json"),
                    simulations_path=str(root / "selfplay" / f"games_{index}"),
                    network_path_pt=str(path / "network.pt"), network_path_onnx=str(path / "network.onnx"),
                    finished_path=str(path / "finished.txt"), cleanup_old_pt_files=lambda *a, **k: None,
                )

            def export(_network, name):
                path = root / f"{name}.onnx"
                path.write_bytes(b"onnx")
                return str(path)

            settings = loop_settings(
                root, optimizer=lambda _: None, min_buffer_size=1, wait_for_new_network=True, dummy_network=None,
                log_path=str(root / "log.npz"), train_settings=SimpleNamespace(train_step=train_step),
                save_tmp_onnx_network=export, evaluate_network=lambda *_: None,
                sampler=lambda *a, **k: SimpleNamespace(
                    group=SimpleNamespace(positions=[1]), next_batch_either=lambda: None, close=lambda: None),
                calc_batch_count_per_gen=lambda *_: 1.0,
            )
            buffer = SimpleNamespace(position_count=0, simulation_count=1)
            buffer.append = lambda *_: setattr(buffer, "position_count", 10)
            logger = SimpleNamespace(start_batch=lambda: None, log=lambda *_: None, save=lambda _: None)
            network = SimpleNamespace(parameters=lambda: [])

            env = {"CHECKPOINT_FLAG_PATH": str(flag), "CHECKPOINT_DONE_PATH": str(done)}
            with patch.dict(os.environ, env), \
                    patch("lib.selfplay_client.connect_to_selfplay_server", return_value=client_end), \
                    patch("lib.loop.Generation.from_gi", side_effect=lambda _, index: gen(index)), \
                    patch("lib.loop.DataFile.open"), patch("lib.loop.print_param_count"), \
                    patch("lib.loop.json.dump"), patch("lib.loop.torch.jit.save"), \
                    patch("lib.loop.torch.jit.load", return_value=network):
                LoopSettings.run_loop_inner(settings, gen(0), buffer, logger, None, network)

            self.assertEqual(commands[3:], ["WaitForNewNetwork", "Checkpoint"])
            self.assertTrue((root / "training" / "gen_0" / "finished.txt").exists())
            self.assertTrue(done.exists())
            self.assertTrue(Path(settings.game_checkpoint_config_path).exists())

    def test_changed_settings_are_rejected_before_connecting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "selfplay").mkdir()
            settings = loop_settings(root)
            Path(settings.game_checkpoint_path).write_text("{}")
            config = settings.game_checkpoint_config()
            config["selfplay_settings"] = {"max_game_length": 300}
            Path(settings.game_checkpoint_config_path).write_text(json.dumps(config))

            with patch("lib.loop.SelfplayClient") as client:
                with self.assertRaisesRegex(ValueError, "Selfplay settings differ"):
                    LoopSettings.run_loop_inner(settings, None, None, None, None, None)
            client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
