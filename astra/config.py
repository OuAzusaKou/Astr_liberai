from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/share-2/code/yulonglu/projects/liber_imagewam/outputs/runtime/robodojo/RoboDojo')
PROJECT = Path('/share-2/code/liupeiqi/imagewam-robodojo-9b-l3mix-launch-42f860c0')
RUN = Path('/share-2/home/vla/liupeiqi/outputs/imagewam-robodojo-9b-l3l2-20260902/train/imagewam-robodojo-9b-l3l2mix-352x480-fp32-compile-s42-72k-20260903T122000Z-42f860c0')
CHECKPOINT = RUN / 'checkpoints/weights/step_072000.pt'
DATA = Path('/share-2/home/vla/yulonglu/data/RoboDojo/RoboDojo_lerobot_v21_video')
POLICY_ENV = Path('/share-2/home/vla/miniforge3/envs/fastwam-dpnew-qwen')
SIM_ENV = SOURCE.parent / 'envs/RoboDojo'
RUNTIME_DEPS = Path('/share-2/code/yulonglu/projects/liber_imagewam/outputs/runtime_deps/imagewam4b-py310')
CORE9 = ['hang_mugs', 'hang_mugs_random', 'sort_nesting_dolls_by_size',
         'sort_nesting_dolls_by_size_random', 'plug_in_charger', 'pour_balls_into_vase',
         'classify_objects', 'play_tic_tac_toe', 'cover_blocks',
         'match_and_pick_from_conveyor', 'general_pickup']
# Exact semantic mappings to the dataset's tasks.jsonl, verified during preparation.
DEMO_TASK_INDEX = {'hang_mugs': 10, 'hang_mugs_random': 10,
                   'organize_table': 17, 'fill_pen_holder': 8,
                   'sort_nesting_dolls_by_size': 28, 'sort_nesting_dolls_by_size_random': 28,
                   'plug_in_charger': 22, 'pour_balls_into_vase': 23,
                   'classify_objects': 2, 'play_tic_tac_toe': 21,
                   'cover_blocks': 3, 'match_and_pick_from_conveyor': 16}
