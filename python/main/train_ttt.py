import glob
import sys

from torch.optim import AdamW

from lib.data.file import DataFile
from lib.games import Game
from lib.loop import FixedSelfplaySettings, LoopSettings
from lib.model.post_act import PredictionHeads, ResTower, ScalarHead, DensePolicyHead
from lib.selfplay_client import SelfplaySettings, UctWeights
from lib.train import TrainSettings, ScalarTarget


def main():
    # Configuration scaled from go-9 to ttt
    # Scale factors based on game complexity ratio (go-9 ~180 moves vs ttt ~5 moves)
    game = Game.find("ttt")

    # TTT-specific: 3x3 board, ~5 moves per game
    fixed_settings = FixedSelfplaySettings(
        game=game,
        muzero=False,
        start_pos="default",

        # Scaled from 200: ~40x fewer moves per game means fewer simulations needed
        simulations_per_gen=50,

        cpu_threads_per_device=4,
        gpu_threads_per_device=1,
        # Scaled from 2048: smaller input requires less batch parallelism
        gpu_batch_size=256,
        gpu_batch_size_root=0,
        search_batch_size=16,

        saved_state_channels=0,
        eval_random_symmetries=True,
    )

    # Selfplay settings
    selfplay_settings = SelfplaySettings(
        temperature=1.0,
        zero_temp_move_count=30,
        q_mode="wdl+0.0",
        # Scaled from 400: ttt max game length is 9 moves
        max_game_length=20,
        dirichlet_alpha=0.03,
        dirichlet_eps=0.25,
        search_policy_temperature_root=1.4,
        search_policy_temperature_child=1.0,
        search_fpu_root="fixed+0.1",
        search_fpu_child="relative+0",
        search_virtual_loss_weight=1.0,
        full_search_prob=1.0,
        # Scaled from 800: proportional to game complexity
        full_iterations=100,
        part_iterations=20,
        weights=UctWeights.default(),
        cache_size=800,
        top_moves=100,
    )

    # Training settings
    train_settings = TrainSettings(
        game=game,
        value_weight=0.1,
        wdl_weight=1.0,
        policy_weight=1.0,
        sim_weight=0.0,
        moves_left_delta=20,
        moves_left_weight=0.0001,
        clip_norm=5.0,
        scalar_target=ScalarTarget.Final,
        train_in_eval_mode=False,
        mask_policy=True,
    )

    # Scaled network architecture
    # Scale factor ~1/4 based on board complexity (9 vs ~81 cells)
    def build_network(depth: int, channels: int):
        return PredictionHeads(
            common=ResTower(depth, game.full_input_channels, channels),
            scalar_head=ScalarHead(game.board_size, channels, 4, 32),
            policy_head=DensePolicyHead(game, channels, None, None),
        )

    # Scaled from depth=16, channels=128: complexity ratio
    def initial_network():
        return build_network(4, 32)

    initial_files_pattern = ""

    # Loop settings
    settings = LoopSettings(
        gui=sys.platform == "win32",
        # Separate output directory for ttt experiments
        root_path="data/loop/ttt/first/",
        port=63105,
        wait_for_new_network=True,

        dummy_network=None,
        initial_network=initial_network,
        initial_data_files=[DataFile.open(game, path) for path in glob.glob(initial_files_pattern)],

        only_generate=False,

        # Scaled from 500_000 / 1_000_000: ~1% of original
        # ttt games are ~5 moves, so buffer scales accordingly
        min_buffer_size=5_000,
        max_buffer_size=10_000,

        train_batch_size=128,
        samples_per_position=0.3,
        test_fraction=0.05,

        optimizer=lambda params: AdamW(params, weight_decay=1e-3),

        fixed_settings=fixed_settings,
        selfplay_settings=selfplay_settings,
        train_settings=train_settings,

        sample_muzero_steps=None,
        sample_include_final=False,
        sample_random_symmetries=True,
    )

    settings.calc_batch_count_per_gen(game.estimate_moves_per_game, do_print=True)
    settings.run_loop()


if __name__ == '__main__':
    main()
