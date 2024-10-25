from minigrid.envs.babyai.core.roomgrid_level import RoomGridLevel
from minigrid.envs.babyai.core.verifier import GoToInstr, ObjDesc
from minigrid.core.constants import COLOR_ENV_NAMES, COLOR_NAMES

from gymnasium.envs.registration import register

FIXINSTGOTO_ENVS = []
for color in COLOR_ENV_NAMES:
    print(f"color: {color}")
    for obj in ["ball", "box", "key"]:
        FIXINSTGOTO_ENVS.append(
            f"BabyAI-FixInstGoTo{color.capitalize()}{obj.capitalize()}-v0"
        )


class GoToSpecificObject(RoomGridLevel):
    """
    Go to a specific object, single room, with distractors.
    This level has distractors but doesn't make use of language.

    ## Mission Space

    "go to the {color} {obj}"

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent goes to the red ball.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `BabyAI-FixInstGoTo{color}{obj}-v0`

    """

    def __init__(self, color: str, obj: str, room_size=8, num_dists=7, **kwargs):
        assert color in COLOR_NAMES, f"{color} is not a valid color name"
        assert obj in ["ball", "box", "key"], f"{obj} is not a valid object type"
        self.num_dists = num_dists
        self.color = color
        self.obj = obj

        super().__init__(num_rows=1, num_cols=1, room_size=room_size, **kwargs)

    def gen_mission(self):
        self.place_agent()

        obj, _ = self.add_object(0, 0, self.obj, self.color)

        if self.num_dists > 0:
            self.add_distractors(num_distractors=self.num_dists, all_unique=False)

        # Make sure no unblocking is required
        self.check_objs_reachable()

        self.instrs = GoToInstr(ObjDesc(obj.type, obj.color))


def register_envs(num_dists=7):
    """
    Registers envs with all combinations of objects and colors to gym
    """
    ids = []
    for color in COLOR_ENV_NAMES:
        for obj in ["ball", "box", "key"]:
            id = f"BabyAI-FixInstGoTo{color.capitalize()}{obj.capitalize()}-v0"
            ids.append(id)
            register(
                id=id,
                entry_point="diffusion_nl.environments.babyai.goto_specific:GoToSpecificObject",
                kwargs={
                    "room_size": 8,
                    "num_dists": num_dists,
                    "color": color,
                    "obj": obj,
                },
            )
    return ids
