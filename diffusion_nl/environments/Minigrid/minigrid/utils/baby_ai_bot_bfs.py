from __future__ import annotations

import copy
from argparse import Action
from unittest.mock import NonCallableMagicMock

import numpy as np
from collections import deque
from minigrid.core.constants import DIR_TO_VEC, VEC_TO_DIR
from minigrid.core.actions import ActionSpace, Actions
from minigrid.core.world_object import WorldObj
from minigrid.envs.babyai.core.verifier import (
    AfterInstr,
    AndInstr,
    BeforeInstr,
    GoToInstr,
    ObjDesc,
    OpenInstr,
    PickupInstr,
    PutNextInstr,
)
from minigrid.utils.baby_ai_bot import dir2diagonal, rightvec2diagonal, DisappearedBoxError, manhattan_distance


class Subgoal:
    """The base class for all possible Bot subgoals.

    Args:
        bot (BabyAIBot): The bot whose subgoal this is.
        datum (object): The first parameter of the subgoal, e.g. a location or an object description.
        reason (str): Why this subgoal was created. Subgoals created for different reasons require
    """

    def __init__(self, bot: BabyAIBot, datum=None, reason=None):
        self.bot = bot
        self.datum = datum
        self.reason = reason

        self.update_agent_attributes()

        self.actions = self.bot.mission.unwrapped.actions
        self.legal_actions = self.bot.mission.unwrapped.legal_actions

    def __repr__(self):
        """Mainly for debugging purposes"""
        representation = "("
        representation += type(self).__name__
        if self.datum is not None:
            representation += f": {self.datum}"
        if self.reason is not None:
            representation += f", reason: {self.reason}"
        representation += ")"
        return representation

    def update_agent_attributes(self):
        """Should be called at each step before the replanning methods."""
        pass

    @property
    def pos(self):
        return self.bot.mission.unwrapped.agent_pos

    @property
    def dir_vec(self):
        return self.bot.mission.unwrapped.dir_vec

    @property
    def right_vec(self):
        return self.bot.mission.unwrapped.right_vec

    @property
    def right_vec_diagonal(self):
        return rightvec2diagonal(
            self.right_vec
        )[0]

    @property
    def left_vec_diagonal(self):
        return rightvec2diagonal(
            self.right_vec
        )[1]

    @property
    def fwd_pos(self):
        return  self.pos + self.dir_vec

    @property
    def fwd_pos_left(self):
        return self.pos + dir2diagonal(self.dir_vec)[0]

    @property
    def fwd_pos_right(self):
        return self.pos + dir2diagonal(self.dir_vec)[1]

    @property
    def fwd_cell_right(self):
        self.bot.mission.unwrapped.grid.get(*self.fwd_pos_right)

    @property
    def fwd_cell_left(self):
        self.bot.mission.unwrapped.grid.get(*self.fwd_pos_left)

    @property
    def fwd_cell(self):
        return self.bot.mission.unwrapped.grid.get(*self.fwd_pos)

    @property
    def carrying(self):
        return self.bot.mission.unwrapped.carrying


    def replan_before_action(self):
        """Change the plan if needed and return a suggested action.

        This method is called at every iteration for the top-most subgoal
        from the stack. It is supposed to return a suggested action if
        it is clear how to proceed towards achieving the current subgoal.
        If the subgoal is already achieved, or if it is not clear how it
        can be achieved, or if is clear that a better plan exists,
        this method can replan by pushing new subgoals
        from the stack or popping the top one.

        Returns:
            action (object): A suggested action if known, `None` the stack has been
                altered and further replanning is required.
        """
        raise NotImplementedError()

    def replan_after_action(self, action_taken):
        """Change the plan when the taken action is known.

        The action actually taken by the agent can be different from the one
        suggested by `replan_before_action` is the bot can be used in
        advising mode. This method is supposed to adjust the plan in the view
        of the actual action taken.

        """
        pass

    def is_exploratory(self):
        """Whether the subgoal is exploratory or not.

        Exploratory subgoals can be removed from the stack by the bot, e.g.
        when no more exploration is required.

        """
        return False

    def _plan_undo_action(self, action_taken):
        """Plan how to undo the taken action."""
        if action_taken == self.actions.forward:
            # check if the 'forward' action was successful
            if not np.array_equal(self.bot.prev_agent_pos, self.pos):
                self.bot.stack.append(GoNextToSubgoal(self.bot, self.pos))
        elif action_taken == self.actions.left:
            old_fwd_pos = self.pos + self.right_vec
            self.bot.stack.append(GoNextToSubgoal(self.bot, old_fwd_pos))
        elif action_taken == self.actions.right:
            old_fwd_pos = self.pos - self.right_vec
            self.bot.stack.append(GoNextToSubgoal(self.bot, old_fwd_pos))
        elif (
            action_taken == self.actions.drop
            and self.bot.prev_carrying != self.carrying
        ):
            # get that thing back, if dropping was successful
            assert self.fwd_cell.type in ("key", "box", "ball")
            self.bot.stack.append(PickupSubgoal(self.bot))
        elif (
            action_taken == self.actions.pickup
            and self.bot.prev_carrying != self.carrying
        ):
            # drop that thing where you found it
            fwd_cell = self.bot.mission.unwrapped.grid.get(*self.fwd_pos)
            self.bot.stack.append(DropSubgoal(self.bot))
        elif action_taken == self.actions.toggle:
            # if you opened or closed a door, bring it back in the original state
            fwd_cell = self.bot.mission.unwrapped.grid.get(*self.fwd_pos)
            if (
                fwd_cell
                and fwd_cell.type == "door"
                and self.bot.fwd_door_was_open != fwd_cell.is_open
            ):
                self.bot.stack.append(
                    CloseSubgoal(self.bot)
                    if fwd_cell.is_open
                    else OpenSubgoal(self.bot)
                )


class CloseSubgoal(Subgoal):

    def replan_before_action(self):
        assert self.fwd_cell is not None, "Forward cell is empty"
        assert self.fwd_cell.type == "door", "Forward cell has to be a door"
        assert self.fwd_cell.is_open, "Forward door must be open"
        return self.actions.toggle

    def replan_after_action(self, action_taken):
        if action_taken is None or action_taken == self.actions.toggle:
            self.bot.stack.pop()
        elif action_taken in [
            self.actions.forward,
            self.actions.left,
            self.actions.right,
        ]:
            self._plan_undo_action(action_taken)


class OpenSubgoal(Subgoal):
    """Subgoal for opening doors.

    Args:
        reason (str): `None`, `"Unlock"`, or `"UnlockAndKeepKey"`. If the reason is
            `"Unlock"`, the agent will plan dropping the key somewhere after it opens the
            door (see `replan_after_action`). When the agent faces the door, and the reason
            is `None`, this subgoals replaces itself with a similar one, but with with the
            reason `"Unlock"`. `reason="UnlockAndKeepKey` means that the agent should not
            schedule the dropping of the key when it faces a locked door, and should instead
            keep the key.
    """

    def replan_before_action(self):
        assert self.fwd_cell is not None, "Forward cell is empty"
        assert self.fwd_cell.type == "door", "Forward cell has to be a door"
        door_cell = self.fwd_cell
        door_pos = self.fwd_pos

        # If the door is locked, go find the key and then return
        # TODO: do we really need to be in front of the locked door
        # to realize that we need the key for it ?
        got_the_key = (
                self.carrying
                and self.carrying.type == "key"
                and self.carrying.color == door_cell.color
        )
        if door_cell.is_locked and not got_the_key:
            # Find the key
            key_desc = ObjDesc("key", door_cell.color)
            key_desc.find_matching_objs(self.bot.mission)

            # If we're already carrying something
            if self.carrying:
                self.bot.stack.pop()

                # Find a location to drop what we're already carrying
                drop_pos_cur = self.bot._find_drop_pos()

                # Take back the object being carried
                self.bot.stack.append(PickupSubgoal(self.bot))
                self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos_cur))

                # Go back to the door and open it
                self.bot.stack.append(OpenSubgoal(self.bot))
                self.bot.stack.append(GoNextToSubgoal(self.bot, tuple(door_pos)))

                # Go to the key and pick it up
                self.bot.stack.append(PickupSubgoal(self.bot))
                self.bot.stack.append(GoNextToSubgoal(self.bot, key_desc))

                # Drop the object being carried
                self.bot.stack.append(DropSubgoal(self.bot))
                self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos_cur))
            else:
                # This branch is will be used very rarely, given that
                # GoNextToSubGoal(..., reason='Open') should plan
                # going to the key before we get to stand right in front of a door.
                # But the agent can be spawned right in front of a open door,
                # for which we case we do need this code.

                self.bot.stack.pop()

                # Go back to the door and open it
                self.bot.stack.append(OpenSubgoal(self.bot))
                self.bot.stack.append(GoNextToSubgoal(self.bot, tuple(door_pos)))

                # Go to the key and pick it up
                self.bot.stack.append(PickupSubgoal(self.bot))
                self.bot.stack.append(GoNextToSubgoal(self.bot, key_desc))
            return

        if door_cell.is_locked and self.reason is None:
            self.bot.stack.pop()
            self.bot.stack.append(OpenSubgoal(self.bot, reason="Unlock"))
            return

        return self.actions.toggle

    def replan_after_action(self, action_taken):
        if action_taken is None or action_taken == self.actions.toggle:
            if not self.fwd_cell.is_open:
                # This should only trigger when door was open and we closed it.
                # Should fix a bug that occurs when goal is to open a door and opened door is closed (in such case subgoal was incorrectly popped)
                pass
            else:
                self.bot.stack.pop()
                if self.reason == "Unlock":
                    # The reason why this has to be planned after the action is taken
                    # is because if the position for dropping is chosen in advance,
                    # then by the time the key is dropped there, it might already
                    # be occupied.
                    drop_key_pos = self.bot._find_drop_pos()
                    self.bot.stack.append(DropSubgoal(self.bot))
                    self.bot.stack.append(GoNextToSubgoal(self.bot, drop_key_pos))
        else:
            self._plan_undo_action(action_taken)


class DropSubgoal(Subgoal):
    def replan_before_action(self):
        assert self.bot.mission.unwrapped.carrying
        assert not self.fwd_cell
        return self.actions.drop

    def replan_after_action(self, action_taken):
        if action_taken is None or action_taken == self.actions.drop:
            self.bot.stack.pop()
        elif action_taken in [
            self.actions.forward,
            self.actions.left,
            self.actions.right,
        ]:
            self._plan_undo_action(action_taken)


class PickupSubgoal(Subgoal):
    def replan_before_action(self):
        assert not self.bot.mission.unwrapped.carrying
        return self.actions.pickup

    def replan_after_action(self, action_taken):
        if action_taken is None or action_taken == self.actions.pickup:
            self.bot.stack.pop()
        elif action_taken in [self.actions.left, self.actions.right]:
            self._plan_undo_action(action_taken)


class GoNextToSubgoal(Subgoal):
    """The subgoal for going next to objects or positions.

    Args:
        datum (int, int): tuple or `ObjDesc` or object reference
            The position or the description of the object or
            the object to which we are going.
        reason (str): One of the following:
            - `None`: go the position (object) and face it
            - `"PutNext"`: go face an empty position next to the object specified by `datum`
            - `"Explore"`: going to a position, just like when the reason is `None`. The only
                difference is that with this reason the subgoal will be considered exploratory

    """
    def __init__(self, bot: BabyAIBot, datum=None, reason=None):
        super(GoNextToSubgoal, self).__init__(bot=bot, datum=datum, reason=reason)
        # These are used for path caching. As long as agent follows pre-planned there is no need to recalculate the path
        self.path_cache = None
        self.actions_cache = None
        self.target_pos_cache = None


    def replan_before_action(self):
        target_obj = None
        if isinstance(self.datum, ObjDesc):
            target_obj, target_pos = self.bot._find_obj_pos(
                self.datum, self.reason == "PutNext"
            )
            assert target_pos, "There should always be a path to the object"

        elif isinstance(self.datum, WorldObj):
            target_obj = self.datum
            target_pos = target_obj.cur_pos
        else:
            target_pos = tuple(self.datum)

        # we are walking towards the door that we would like to open,
        # it is locked, and we don't have the key. What do we do? If we are carrying
        # something, it makes to just continue, as we still need to bring this object
        # close to the door. If we are not carrying anything though, then it makes
        # sense to change the plan and go straight for the required key.
        if (
                self.reason == "Open"
                and target_obj
                and target_obj.type == "door"
                and target_obj.is_locked
        ):
            key_desc = ObjDesc("key", target_obj.color)
            key_desc.find_matching_objs(self.bot.mission)
            if not self.carrying:
                # No we need to commit to going to this particular door
                self.bot.stack.pop()
                self.bot.stack.append(
                    GoNextToSubgoal(self.bot, target_obj, reason="Open")
                )
                self.bot.stack.append(PickupSubgoal(self.bot))
                self.bot.stack.append(GoNextToSubgoal(self.bot, key_desc))
                return
            else:
                pass

        # We are facing the target cell
        # -> subgoal completed
        if self.reason == "PutNext":
            if manhattan_distance(target_pos, self.fwd_pos) == 1:
                if self.fwd_cell is None:
                    self.bot.stack.pop()
                    return
                if self.fwd_cell.type == "door" and self.fwd_cell.is_open:
                    # We can't drop an object in the cell where the door is.
                    # Instead, we add a subgoal on the stack that will force
                    # the bot to move the target object.
                    self.bot.stack.append(
                        GoNextToSubgoal(self.bot, self.fwd_pos + 2 * self.dir_vec)
                    )
                    return
        else:
            if np.array_equal(target_pos, self.fwd_pos):
                self.bot.stack.pop()
                return

        if self.path_cache is not None and self.actions_cache is not None and target_pos == self.target_pos_cache:
            predicted_pos = self.path_cache[0]
            real_pos = (*self.pos, *self.dir_vec)
            if real_pos != predicted_pos:
                self.path_cache = None
                self.actions_cache = None
                self.target_pos_cache = None
                path, action = self.get_new_path(target_pos=target_pos, adjacent=self.reason == "PutNext")
            else:
                self.path_cache = self.path_cache[1:]
                self.actions_cache = self.actions_cache[1:]
                action = self.actions_cache[0]
                path = self.path_cache
        else:
            path, action = self.get_new_path(target_pos=target_pos, adjacent=self.reason == "PutNext")
        next_cell = self.bot.mission.unwrapped.grid.get(*(path[0][:2]))
        if next_cell and next_cell.type == "door":
            if self.is_forward_move_action(action=action, dir_vec=self.dir_vec):
                # assert not next_cell.is_locked
                if not next_cell.is_open:
                    self.bot.stack.append(OpenSubgoal(self.bot))
                    return
                else:
                    return action

        if next_cell and self.carrying and self.is_forward_move_action(action=action, dir_vec=self.dir_vec):
            drop_pos_cur = self.bot._find_drop_pos()
            drop_pos_block = self.bot._find_drop_pos(drop_pos_cur)

            # Take back the object being carried
            self.bot.stack.append(PickupSubgoal(self.bot))
            self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos_cur))

            # Pick up the blocking object and drop it
            self.bot.stack.append(DropSubgoal(self.bot))
            self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos_block))
            self.bot.stack.append(PickupSubgoal(self.bot))
            self.bot.stack.append(GoNextToSubgoal(self.bot, next_cell))

            # Drop the object being carried
            self.bot.stack.append(DropSubgoal(self.bot))
            self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos_cur))

            return
        elif not self.carrying and next_cell and self.is_forward_move_action(action=action, dir_vec=self.dir_vec):
            drop_pos = self.bot._find_drop_pos()
            self.bot.stack.append(DropSubgoal(self.bot))
            self.bot.stack.append(GoNextToSubgoal(self.bot, drop_pos))
            self.bot.stack.append(PickupSubgoal(self.bot))
            return
        else:
            return action

    @staticmethod
    def is_forward_move_action(action: Actions, dir_vec):
        if action == Actions.forward:
            return True
        elif action == Actions.right_move and np.all(dir_vec == DIR_TO_VEC[0]):
            return True
        elif action == Actions.down_move and np.all(dir_vec == DIR_TO_VEC[1]):
            return True
        elif action == Actions.left_move and np.all(dir_vec == DIR_TO_VEC[2]):
            return True
        elif action == Actions.up_move and np.all(dir_vec == DIR_TO_VEC[3]):
            return True
        else:
            return False



    def get_new_path(self, target_pos, adjacent=False):
        # We are still far from the target
        # -> try to find a non-blocker path
        if adjacent:
            def accept_fn(pos, dir, cell):
                fwd_cell = self.bot.mission.unwrapped.grid.get(pos[0] + dir[0], pos[1] + dir[1])
                man_dist = abs(pos[0] + dir[0] - target_pos[0]) + abs(pos[1] + dir[1] - target_pos[1])
                return fwd_cell is None and man_dist == 1
        else:
            accept_fn = lambda pos, dir, cell: (pos[0] + dir[0] == target_pos[0]) and (pos[1] + dir[1] == target_pos[1])


        path, _, _, actions = self.bot._shortest_path(
            accept_fn=accept_fn
        )

        # No non-blocker path found and
        # reexploration within the room is not allowed or there is nothing to explore
        # -> Look for blocker paths
        if not path:
            path, _, _, actions = self.bot._shortest_path(
                accept_fn=accept_fn,
                try_with_blockers=True
            )

        # No path found
        # -> explore the world
        if not path:
            self.bot.stack.append(ExploreSubgoal(self.bot))
            return

        # So there is a path (blocker, or non-blockers)
        # -> try following it
        action = actions[0]
        self.path_cache = path
        self.actions_cache = actions
        self.target_pos_cache = target_pos
        return path, action

    def replan_after_action(self, action_taken):
        if action_taken in [
            self.actions.pickup,
            self.actions.drop,
            self.actions.toggle,
        ]:
            self._plan_undo_action(action_taken)

    def is_exploratory(self):
        return self.reason == "Explore"


class ExploreSubgoal(Subgoal):
    def replan_before_action(self):
        # Find the closest unseen position
        _, unseen_pos, with_blockers, actions = self.bot._shortest_path(
            lambda pos, dir, cell: not self.bot.vis_mask[pos], try_with_blockers=True
        )

        if unseen_pos:
            self.bot.stack.append(
                GoNextToSubgoal(self.bot, unseen_pos, reason="Explore")
            )
            return None

        # Find the closest unlocked unopened door
        def unopened_unlocked_door(pos, dir, cell):
            return (
                    cell and cell.type == "door" and not cell.is_locked and not cell.is_open
            )

        # Find the closest unopened door
        def unopened_door(pos, dir, cell):
            return cell and cell.type == "door" and not cell.is_open

        # Try to find an unlocked door first.
        # We do this because otherwise, opening a locked door as
        # a subgoal may try to open the same door for exploration,
        # resulting in an infinite loop.
        _, door_pos, _, actions = self.bot._shortest_path(
            unopened_unlocked_door, try_with_blockers=True
        )
        if not door_pos:
            # Try to find a locker door if an unlocked one is not available.
            _, door_pos, _, actions = self.bot._shortest_path(
                unopened_door, try_with_blockers=True
            )

        # Open the door
        if door_pos:
            door_pos = door_pos[:2]
            door_obj = self.bot.mission.unwrapped.grid.get(*door_pos)
            # If we are going to a locked door, there are two cases:
            # - we already have the key, then we should not drop it
            # - we don't have the key, in which case eventually we should drop it
            got_the_key = (
                    self.carrying
                    and self.carrying.type == "key"
                    and self.carrying.color == door_obj.color
            )
            open_reason = "KeepKey" if door_obj.is_locked and got_the_key else None
            self.bot.stack.pop()
            self.bot.stack.append(OpenSubgoal(self.bot, reason=open_reason))
            self.bot.stack.append(GoNextToSubgoal(self.bot, door_obj, reason="Open"))
            return

        assert False, "nothing left to explore"

    def is_exploratory(self):
        return True


class BabyAIBot:
    """A bot that can solve all BabyAI levels*.

    The bot maintains a plan, represented as a stack of the so-called
    subgoals. The initial set of subgoals is generated from the instruction.
    The subgoals are then executed one after another, unless a change of
    plan is required (e.g. the location of the target object is not known
    or there other objects in the way). In this case, the bot changes the plan.

    The bot can also be used to advice a suboptimal agent, e.g. play the
    role of an oracle in algorithms like DAGGER. It changes the plan based on
    the actual action that the agent took.

    The main method of the bot (and the only one you are supposed to use) is `replan`.

    * The bot can solve all BabyAI levels from the original paper. It can also solve
    most of the bonus levels from the original BabyAI repository, but fails to solve the
    following:
      - "BabyAI-PutNextS5N2Carrying-v0",
      - "BabyAI-PutNextS6N3Carrying-v0",
      - "BabyAI-PutNextS7N4Carrying-v0",
      - "BabyAI-KeyInBox-v0".

    Args:
        mission: a freshly created BabyAI environment

    """

    def __init__(self, mission, action_space_type: int):
        # Mission to be solved
        self.mission = mission
        self.action_space_type = action_space_type

        # Grid containing what has been mapped out
        # self.grid = Grid(mission.unwrapped.width, mission.unwrapped.height)

        # Visibility mask. True for explored/seen, false for unexplored.
        # To Do: Replace with environment attribute that indicates whether fully observable or not
        if True:
            self.vis_mask = np.ones(
                shape=(mission.unwrapped.width, mission.unwrapped.height), dtype=bool
            )

        # Stack of tasks/subtasks to complete (tuples)
        self.stack = []

        # Process/parse the instructions

        self._process_instr(mission.unwrapped.instrs)

        # How many BFS searches this bot has performed
        self.bfs_counter = 0

        # How many steps were made in total in all BFS searches
        # performed by this bot
        self.bfs_step_counter = 0

    def replan(self, action_taken=None):
        """Replan and suggest an action.

        Call this method once per every iteration of the environment.

        Args:
            action_taken: The last action that the agent took. Can be `None`, in which
            case the bot assumes that the action it suggested was taken (or that it is
            the first iteration).

        Returns:
            suggested_action: The action that the bot suggests. Can be `done` if the
            bot thinks that the mission has been accomplished.

        """
        # self._process_obs()

        # Check that no box has been opened
        self._check_erroneous_box_opening(action_taken)

        # TODO: instead of updating all subgoals, just add a couple
        # properties to the `Subgoal` class.
        for subgoal in self.stack:
            subgoal.update_agent_attributes()

        if self.stack:
            self.stack[-1].replan_after_action(action_taken)

        # Clear the stack from the non-essential subgoals
        while self.stack and self.stack[-1].is_exploratory():
            self.stack.pop()

        suggested_action = None
        while self.stack:
            subgoal = self.stack[-1]
            suggested_action = subgoal.replan_before_action()

            # If is not clear what can be done for the current subgoal
            # (because it is completed, because there is blocker,
            # or because exploration is required), keep replanning
            if suggested_action is not None:
                break

        if not self.stack:
            suggested_action = self.mission.unwrapped.actions.done

        self._remember_current_state()

        return suggested_action

    def _find_obj_pos(self, obj_desc, adjacent=False):
        """Find the position of the closest visible object matching a given description."""

        assert len(obj_desc.obj_set) > 0

        best_distance_to_obj = 999
        best_pos = None
        best_obj = None

        for i in range(len(obj_desc.obj_set)):
            if obj_desc.obj_set[i].type == "wall":
                continue
            try:
                if obj_desc.obj_set[i] == self.mission.unwrapped.carrying:
                    continue
                obj_pos = obj_desc.obj_poss[i]

                if self.vis_mask[obj_pos]:
                    if adjacent:
                        def accept_fn(pos, dir, cell):
                            fwd_cell = self.mission.unwrapped.grid.get(pos[0] + dir[0], pos[1] + dir[1])
                            man_dist = abs(pos[0] + dir[0] - obj_pos[0]) + abs(pos[1] + dir[1] - obj_pos[1])
                            return fwd_cell is None and man_dist == 1
                    else:
                        accept_fn = lambda pos, dir, cell: (pos[0] + dir[0] == obj_pos[0]) and (pos[1] + dir[1] == obj_pos[1])

                    shortest_path_to_obj, _, with_blockers, actions = self._shortest_path(
                        accept_fn=accept_fn,
                        try_with_blockers=True
                    )
                    assert shortest_path_to_obj is not None
                    distance_to_obj = len(shortest_path_to_obj)

                    if with_blockers:
                        # The distance should take into account the steps necessary
                        # to unblock the way. It will always be at least 1 (door). Not computed exactly atm.
                        distance_to_obj = len(shortest_path_to_obj) + 1

                    if distance_to_obj < best_distance_to_obj:
                        best_distance_to_obj = distance_to_obj
                        best_pos = obj_pos
                        best_obj = obj_desc.obj_set[i]
            except IndexError:
                # Suppose we are tracking red keys, and we just used a red key to open a door,
                # then for the last i, accessing obj_desc.obj_poss[i] will raise an IndexError
                # -> Solution: Not care about that red key we used to open the door
                pass

        return best_obj, best_pos

    def _process_obs(self):
        """Parse the contents of an observation/image and update our state."""
        pass

    def _remember_current_state(self):
        self.prev_agent_pos = self.mission.unwrapped.agent_pos
        self.prev_carrying = self.mission.unwrapped.carrying
        fwd_cell = self.mission.unwrapped.grid.get(
            *self.mission.unwrapped.agent_pos + self.mission.unwrapped.dir_vec
        )
        if fwd_cell and fwd_cell.type == "door":
            self.fwd_door_was_open = fwd_cell.is_open
        self.prev_fwd_cell = fwd_cell

    def _closest_wall_or_door_given_dir(self, position, direction):
        distance = 1
        while True:
            position_to_try = position + distance * direction
            # If the current position is outside the field of view,
            # stop everything and return the previous one
            if not self.mission.unwrapped.in_view(*position_to_try):
                return distance - 1
            cell = self.mission.unwrapped.grid.get(*position_to_try)
            if cell and (cell.type.endswith("door") or cell.type == "wall"):
                return distance
            distance += 1

    def _breadth_first_search(self, initial_states, accept_fn, ignore_blockers, previous_pos=None):
        """Performs breadth first search.

        This is pretty much your textbook BFS. The state space is agent's locations,
        but the current direction is also added to the queue to slightly prioritize
        going straight over turning.

        """
        self.bfs_counter += 1

        queue = []
        if previous_pos is None:
            previous_pos = dict()
        grid = self.mission.unwrapped.grid
        for state in initial_states:
            queue.append(state)
            if state not in previous_pos:
                previous_pos[state] = (None, None)

        while len(queue) > 0:
            state = queue[0]
            queue = queue[1:]
            i, j, di, dj = state
            self.bfs_step_counter += 1

            cell = grid.get(i, j)

            # If we reached a position satisfying the acceptance condition
            if accept_fn((i, j), (di, dj), cell):
                path = []
                actions = []
                pos = (i, j, di, dj)

                while pos:
                    path.append(pos)
                    pos, a = previous_pos[pos]
                    if not (pos is None and a is None):
                        actions.append(a)

                # Order from start to finish (including start state)
                path = path[::-1]
                actions = actions[::-1]
                return path, (i, j, di, dj), previous_pos, actions

            # If this cell was not visually observed, don't expand from it
            if not self.vis_mask[i, j]:
                continue



            # Location to which the bot can get without turning
            # are put in the queue first
            for action in self.action_space_type.get_legal_actions():
                if action == Actions.left:
                    new_pos = (i, j)
                    new_dir_vec = rotate_left((di, dj))

                # Rotate right
                elif action == Actions.right:
                    new_pos = (i, j)
                    new_dir_vec = rotate_right((di, dj))

                # Move forward
                elif action == Actions.forward:
                    new_pos = front_pos(pos=(i, j), dir_vec=(di, dj))
                    new_dir_vec = (di, dj)

                # Move backward
                elif action == Actions.backward:
                    new_pos = back_pos(pos=(i, j), dir_vec=(di, dj))
                    new_dir_vec = (di, dj)

                # Move left while keeping the direction
                elif action == Actions.left_move_no_turn:
                    new_pos = left_pos(pos=(i, j), dir_vec=(di, dj))
                    new_dir_vec = (di, dj)

                # Move right while keeping the direction
                elif action == Actions.right_move_no_turn:
                    new_pos = right_pos(pos=(i, j), dir_vec=(di, dj))
                    new_dir_vec = (di, dj)

                # Move diagonal right
                elif action == Actions.diagonal_right:
                    new_pos = front_right_pos(pos=(i, j), dir_vec=(di, dj))
                    new_dir_vec = (di, dj)

                # Move diagonal left
                elif action == Actions.diagonal_left:
                    new_pos = front_left_pos(pos=(i, j), dir_vec=(di, dj))
                    new_dir_vec = (di, dj)

                # Move diagonal backwards left
                elif action == Actions.diagonal_backwards_left:
                    new_pos = back_left_pos(pos=(i, j), dir_vec=(di, dj))
                    new_dir_vec = (di, dj)

                # Move diagonal backwards right
                elif action == Actions.diagonal_backwards_right:
                    new_pos = back_right_pos(pos=(i, j), dir_vec=(di, dj))
                    new_dir_vec = (di, dj)

                # Move diagonal backwards left
                elif action == Actions.diagonal_backwards_left:
                    new_pos = back_left_pos(pos=(i, j), dir_vec=(di, dj))
                    new_dir_vec = (di, dj)

                # Turn 180 degrees
                elif action == Actions.turn_around:
                    new_pos = (i, j)
                    new_dir_vec = rotate_180((di, dj))

                # WSAD actions
                # Move to one dir in 1 step (combines turn + forward)
                elif action == Actions.right_move:
                    new_dir_vec = (1, 0)
                    new_pos = front_pos(pos=(i, j), dir_vec=new_dir_vec)

                elif action == Actions.down_move:
                    new_dir_vec = (0, 1)
                    new_pos = front_pos(pos=(i, j), dir_vec=new_dir_vec)

                elif action == Actions.left_move:
                    new_dir_vec = (-1, 0)
                    new_pos = front_pos(pos=(i, j), dir_vec=new_dir_vec)

                elif action == Actions.up_move:
                    new_dir_vec = (0, -1)
                    new_pos = front_pos(pos=(i, j), dir_vec=new_dir_vec)

                # Pick up an object
                elif action == Actions.pickup:
                    continue

                # Drop an object
                elif action == Actions.drop:
                    continue

                # Toggle/activate an object
                elif action == Actions.toggle:
                    continue

                # Done action (not used by default)
                elif action == Actions.done:
                    continue

                else:
                    raise ValueError(f"Unknown action: {action}")

                next_state = (*new_pos, *new_dir_vec)
                next_cell = grid.get(*new_pos)

                # Check whether the position is not previously visited
                if next_state not in previous_pos:
                    dont_add = False
                    if next_cell:
                        if next_cell.type == "wall":
                            dont_add = True
                        # If this is a door
                        elif next_cell.type == "door":
                            # If the door is closed, don't visit neighbors
                            if not next_cell.is_open:
                                if not ignore_blockers or action not in (Actions.forward, Actions.up_move, Actions.right_move, Actions.left_move, Actions.down_move):
                                    dont_add = True
                        elif not ignore_blockers or action not in (Actions.forward, Actions.up_move, Actions.right_move, Actions.left_move, Actions.down_move):
                            dont_add = True
                    if dont_add:
                        # Check special case (if the cell is not walkable x_move actions still turn the agent)
                        if action in (Actions.forward, Actions.up_move, Actions.right_move, Actions.left_move, Actions.down_move):
                            next_state = (i, j, *new_dir_vec)
                            if next_state in previous_pos:
                                continue
                        else:
                            continue
                    previous_pos[next_state] = ((i, j, di, dj), action)
                    queue.append(next_state)

        # Path not found
        return None, None, previous_pos, None

    def _shortest_path(self, accept_fn, try_with_blockers=False):
        """
        Finds the path to any of the locations that satisfy `accept_fn`.
        Prefers the paths that avoid blockers for as long as possible.
        """

        # Initial states to visit (BFS)
        initial_states = [
            (*self.mission.unwrapped.agent_pos, *self.mission.unwrapped.dir_vec)
        ]

        path = finish = None
        with_blockers = False
        path, finish, previous_pos, actions = self._breadth_first_search(
            initial_states, accept_fn, ignore_blockers=False
        )

        if not path and try_with_blockers:
            with_blockers = True
            path, finish, _, actions = self._breadth_first_search(
                [(i, j, di, dj) for i, j, di, dj in previous_pos], accept_fn, ignore_blockers=True, previous_pos=previous_pos
            )
        if path:
            # And the starting position is not required
            path = path[1:]  # Cut first state

        # Note, that with_blockers only makes sense if path is not None
        return path, finish, with_blockers, actions

    def _find_drop_pos(self, except_pos=None):
        """
        Find a position where an object can be dropped, ideally without blocking anything.
        """

        grid = self.mission.unwrapped.grid

        def match_unblock(pos, dir, cell):
            # Consider the region of 8 neighboring cells around the candidate cell.
            # If dropping the object in the candidate makes this region disconnected,
            # then probably it is better to drop elsewhere.

            i, j = pos
            agent_pos = tuple(self.mission.unwrapped.agent_pos)

            if np.array_equal(pos, agent_pos):
                return False

            if except_pos and np.array_equal(pos, except_pos):
                return False

            if not self.vis_mask[i, j] or grid.get(i, j):
                return False

            # We distinguish cells of three classes:
            # class 0: the empty ones, including open doors
            # class 1: those that are not interesting (just walls so far)
            # class 2: all the rest, including objects and cells that are current not visible,
            #          and hence may contain objects, and also `except_pos` at it may soon contain
            #          an object
            # We want to ensure that empty cells are connected, and that one can reach
            # any object cell from any other object cell.
            cell_class = []
            for k, l in [
                (-1, -1),
                (0, -1),
                (1, -1),
                (1, 0),
                (1, 1),
                (0, 1),
                (-1, 1),
                (-1, 0),
            ]:
                nb_pos = (i + k, j + l)
                cell = grid.get(*nb_pos)
                # completely blocked
                if self.vis_mask[nb_pos] and cell and cell.type == "wall":
                    cell_class.append(1)
                # empty
                elif (
                        self.vis_mask[nb_pos]
                        and (
                                not cell
                                or (cell.type == "door" and cell.is_open)
                                or nb_pos == agent_pos
                        )
                        and nb_pos != except_pos
                ):
                    cell_class.append(0)
                # an object cell
                else:
                    cell_class.append(2)

            # Now we need to check that empty cells are connected. To do that,
            # let's check how many times empty changes to non-empty
            changes = 0
            for i in range(8):
                if bool(cell_class[(i + 1) % 8]) != bool(cell_class[i]):
                    changes += 1

            # Lastly, we need check that every object has an adjacent empty cell
            for i in range(8):
                next_i = (i + 1) % 8
                prev_i = (i + 7) % 8
                if (
                        cell_class[i] == 2
                        and cell_class[prev_i] != 0
                        and cell_class[next_i] != 0
                ):
                    return False

            return changes <= 2

        def match_empty(pos, dir, cell):
            i, j = pos

            if np.array_equal(pos, self.mission.unwrapped.agent_pos):
                return False

            if except_pos and np.array_equal(pos, except_pos):
                return False

            if not self.vis_mask[pos] or grid.get(*pos):
                return False

            return True

        _, drop_pos, _, actions = self._shortest_path(match_unblock)

        if not drop_pos:
            _, drop_pos, _, actions = self._shortest_path(match_empty)

        if not drop_pos:
            _, drop_pos, _, actions = self._shortest_path(match_unblock, try_with_blockers=True)

        if not drop_pos:
            _, drop_pos, _, actions = self._shortest_path(match_empty, try_with_blockers=True)

        return drop_pos[:2]

    def _process_instr(self, instr):
        """
        Translate instructions into an internal form the agent can execute
        """

        if isinstance(instr, GoToInstr):
            self.stack.append(GoNextToSubgoal(self, instr.desc))
            return

        if isinstance(instr, OpenInstr):
            self.stack.append(OpenSubgoal(self))
            self.stack.append(GoNextToSubgoal(self, instr.desc, reason="Open"))
            return

        if isinstance(instr, PickupInstr):
            # We pick up and immediately drop so
            # that we may carry other objects
            self.stack.append(DropSubgoal(self))
            self.stack.append(PickupSubgoal(self))
            self.stack.append(GoNextToSubgoal(self, instr.desc))
            return

        if isinstance(instr, PutNextInstr):
            self.stack.append(DropSubgoal(self))
            self.stack.append(GoNextToSubgoal(self, instr.desc_fixed, reason="PutNext"))
            self.stack.append(PickupSubgoal(self))
            self.stack.append(GoNextToSubgoal(self, instr.desc_move))
            return

        if isinstance(instr, BeforeInstr) or isinstance(instr, AndInstr):
            self._process_instr(instr.instr_b)
            self._process_instr(instr.instr_a)
            return

        if isinstance(instr, AfterInstr):
            self._process_instr(instr.instr_a)
            self._process_instr(instr.instr_b)

            return

        assert False, "unknown instruction type"

    def _check_erroneous_box_opening(self, action):
        """
        When the agent opens a box, we raise an error and mark the task unsolvable.
        This is a tad conservative, because maybe the box is irrelevant to the mission.unwrapped.
        """
        if (
                action == self.mission.unwrapped.actions.toggle
                and self.prev_fwd_cell is not None
                and self.prev_fwd_cell.type == "box"
        ):
            raise DisappearedBoxError("A box was opened. I am not sure I can help now.")

# class BabyAICustom(BabyAIBot):
#
#     def _breadth_first_search(self, initial_states, accept_fn, ignore_blockers):
#         """Performs breadth first search.
#
#         This is pretty much your textbook BFS. The state space is agent's locations,
#         but the current direction is also added to the queue to slightly prioritize
#         going straight over turning.
#
#         """
#         self.bfs_counter += 1
#
#         queue = []
#         previous_pos = dict()
#         grid = self.mission.unwrapped.grid
#         for state in initial_states:
#             queue.append(state)
#             previous_pos[state] = (None, None)
#
#         while len(queue) > 0:
#             state = queue[0]
#             queue = queue[1:]
#             i, j, di, dj = state
#             self.bfs_step_counter += 1
#
#             cell = grid.get(i, j)
#
#             # If we reached a position satisfying the acceptance condition
#             if accept_fn((i, j), (di, dj), cell):
#                 path = []
#                 actions = []
#                 pos = (i, j, di, dj)
#                 path.append(pos)
#                 while pos:
#                     pos, a = previous_pos[pos]
#                     if pos is not None:
#                         path.append(pos)
#                     actions.append(a)
#                 return path, (i, j, di, dj), previous_pos, actions
#
#             # If this cell was not visually observed, don't expand from it
#             if not self.vis_mask[i, j]:
#                 continue
#
#             if cell:
#                 if cell.type == "wall":
#                     continue
#                 # If this is a door
#                 elif cell.type == "door":
#                     # If the door is closed, don't visit neighbors
#                     if not cell.is_open:
#                         if not ignore_blockers:
#                             continue
#                 elif not ignore_blockers:
#                     continue
#
#             # Location to which the bot can get without turning
#             # are put in the queue first
#             for action in self.action_space_type.get_legal_actions():
#                 if action == Actions.left:
#                     new_pos = (i, j)
#                     new_dir_vec = rotate_left((di, dj))
#
#                 # Rotate right
#                 elif action == Actions.right:
#                     new_pos = (i, j)
#                     new_dir_vec = rotate_right((di, dj))
#
#                 # Move forward
#                 elif action == Actions.forward:
#                     new_pos = front_pos(pos=(i, j), dir_vec=(di, dj))
#                     new_dir_vec = (di, dj)
#
#                 # Move backward
#                 elif action == Actions.backward:
#                     new_pos = back_pos(pos=(i, j), dir_vec=(di, dj))
#                     new_dir_vec = (di, dj)
#
#                 # Move left while keeping the direction
#                 elif action == Actions.left_move_no_turn:
#                     new_pos = left_pos(pos=(i, j), dir_vec=(di, dj))
#                     new_dir_vec = (di, dj)
#
#                 # Move right while keeping the direction
#                 elif action == Actions.right_move_no_turn:
#                     new_pos = right_pos(pos=(i, j), dir_vec=(di, dj))
#                     new_dir_vec = (di, dj)
#
#                 # Move diagonal right
#                 elif action == Actions.diagonal_right:
#                     new_pos = front_right_pos(pos=(i, j), dir_vec=(di, dj))
#                     new_dir_vec = (di, dj)
#
#                 # Move diagonal left
#                 elif action == Actions.diagonal_left:
#                     new_pos = front_left_pos(pos=(i, j), dir_vec=(di, dj))
#                     new_dir_vec = (di, dj)
#
#                 # Move diagonal backwards left
#                 elif action == Actions.diagonal_backwards_left:
#                     new_pos = back_left_pos(pos=(i, j), dir_vec=(di, dj))
#                     new_dir_vec = (di, dj)
#
#                 # Move diagonal backwards right
#                 elif action == Actions.diagonal_backwards_right:
#                     new_pos = back_right_pos(pos=(i, j), dir_vec=(di, dj))
#                     new_dir_vec = (di, dj)
#
#                 # Move diagonal backwards left
#                 elif action == Actions.diagonal_backwards_left:
#                     new_pos = back_left_pos(pos=(i, j), dir_vec=(di, dj))
#                     new_dir_vec = (di, dj)
#
#                 # Turn 180 degrees
#                 elif action == Actions.turn_around:
#                     new_pos = (i, j)
#                     new_dir_vec = rotate_180((di, dj))
#
#                 # WSAD actions
#                 # Move to one dir in 1 step (combines turn + forward)
#                 elif action == Actions.right_move:
#                     # TODO
#                     continue
#
#                 elif action == Actions.down_move:
#                     # TODO
#                     continue
#
#                 elif action == Actions.left_move:
#                     # TODO
#                     continue
#
#                 elif action == Actions.up_move:
#                     # TODO
#                     continue
#
#                 # Pick up an object
#                 elif action == Actions.pickup:
#                     continue
#
#                 # Drop an object
#                 elif action == Actions.drop:
#                     continue
#
#                 # Toggle/activate an object
#                 elif action == Actions.toggle:
#                     continue
#
#                 # Done action (not used by default)
#                 elif action == Actions.done:
#                     continue
#
#                 else:
#                     raise ValueError(f"Unknown action: {action}")
#
#                 next_state = (*new_pos, *new_dir_vec)
#                 if next_state not in previous_pos:
#                     previous_pos[next_state] = ((i, j, di, dj), action)
#                     queue.append(next_state)
#
#         # Path not found
#         return None, None, previous_pos
#
#     def _shortest_path(self, accept_fn, try_with_blockers=False):
#         """
#         Finds the path to any of the locations that satisfy `accept_fn`.
#         Prefers the paths that avoid blockers for as long as possible.
#         """
#
#         # Initial states to visit (BFS)
#         initial_states = [
#             (*self.mission.unwrapped.agent_pos, *self.mission.unwrapped.dir_vec)
#         ]
#
#         path = finish = None
#         with_blockers = False
#         path, finish, previous_pos, actions = self._breadth_first_search(
#             initial_states, accept_fn, ignore_blockers=False
#         )
#
#         if not path and try_with_blockers:
#             with_blockers = True
#             path, finish, _, actions = self._breadth_first_search(
#                 [(i, j, di, dj) for i, j, di, dj in previous_pos], accept_fn, ignore_blockers=True
#             )
#             if path:
#                 # `path` now contains the path to a cell that is reachable without
#                 # blockers. Now let's add the path to this cell
#                 pos = path[-1]
#                 extra_path = []
#                 while pos:
#                     extra_path.append(pos)
#                     pos = previous_pos[pos]
#                 path = path + extra_path[1:]
#
#         if path:
#             # And the starting position is not required
#             path = path[::-1]
#             path = path[1:]
#             actions = actions[::-1]
#             actions = actions[1:]
#
#         # Note, that with_blockers only makes sense if path is not None
#         return path, finish, with_blockers, actions
#
#     def _find_obj_pos(self, obj_desc, adjacent=False):
#         """Find the position of the closest visible object matching a given description."""
#
#         assert len(obj_desc.obj_set) > 0
#
#         best_distance_to_obj = 999
#         best_pos = None
#         best_obj = None
#
#         for i in range(len(obj_desc.obj_set)):
#             if obj_desc.obj_set[i].type == "wall":
#                 continue
#             try:
#                 if obj_desc.obj_set[i] == self.mission.unwrapped.carrying:
#                     continue
#                 obj_pos = obj_desc.obj_poss[i]
#
#                 if self.vis_mask[obj_pos]:
#                     shortest_path_to_obj, _, with_blockers, actions = self._shortest_path(
#                         lambda pos, dir, cell: (pos[0] + dir[0] == obj_pos[0]) and (pos[1] + dir[1] == obj_pos[1]),
#                         try_with_blockers=True
#                     )
#                     assert shortest_path_to_obj is not None
#                     distance_to_obj = len(shortest_path_to_obj)
#
#                     if with_blockers:
#                         # The distance should take into account the steps necessary
#                         # to unblock the way. Instead of computing it exactly,
#                         # we can use a lower bound on this number of steps
#                         # which is 4 when the agent is not holding anything
#                         # (pick, turn, drop, turn back
#                         # and 7 if the agent is carrying something
#                         # (turn, drop, turn back, pick,
#                         # turn to other direction, drop, turn back)
#                         distance_to_obj = len(shortest_path_to_obj) + (
#                             7 if self.mission.unwrapped.carrying else 4
#                         )
#
#                     # If we looking for a door and we are currently in that cell
#                     # that contains the door, it will take us at least 2
#                     # (3 if `adjacent == True`) steps to reach the goal.`
#                     if distance_to_obj == 0:
#                         distance_to_obj = 3 if adjacent else 2
#
#                     # If what we want is to face a location that is adjacent to an object,
#                     # and if we are already right next to this object,
#                     # then we should not prefer this object to those at distance 2
#                     if adjacent and distance_to_obj == 1:
#                         distance_to_obj = 3
#
#                     if distance_to_obj < best_distance_to_obj:
#                         best_distance_to_obj = distance_to_obj
#                         best_pos = obj_pos
#                         best_obj = obj_desc.obj_set[i]
#             except IndexError:
#                 # Suppose we are tracking red keys, and we just used a red key to open a door,
#                 # then for the last i, accessing obj_desc.obj_poss[i] will raise an IndexError
#                 # -> Solution: Not care about that red key we used to open the door
#                 pass
#
#         return best_obj, best_pos


def front_pos(pos, dir_vec):
    """
    Get the position of the pos that in front of the agent
    """

    return pos[0] + dir_vec[0], pos[1] + dir_vec[1]


def back_pos(pos, dir_vec):
    """
    Get the position of the pos that is behind the agent
    """

    return pos[0] - dir_vec[0], pos[1] - dir_vec[1]


def right_pos(pos, dir_vec):
    """
    Get the position of the pos that is right from the agent
    """
    direction = VEC_TO_DIR[dir_vec]
    right = DIR_TO_VEC[(direction + 1) % 4]
    return pos[0] + right[0], pos[1] + right[1]


def left_pos(pos, dir_vec):
    """
    Get the position of the pos that is left from the agent
    """
    direction = VEC_TO_DIR[dir_vec]
    left = DIR_TO_VEC[(direction - 1) % 4]
    return pos[0] + left[0], pos[1] + left[1]


def front_left_pos(pos, dir_vec):
    """
    Get the position of the pos that is in front and one pos to the left.
    """
    return front_pos(left_pos(pos, dir_vec), dir_vec)


def front_right_pos(pos, dir_vec):
    """
    Get the position of the pos that is in front and one pos to the right.
    """
    return front_pos(right_pos(pos, dir_vec), dir_vec)


def back_left_pos(pos, dir_vec):
    """
    Get the position of the pos that is behind and one pos to the left.
    """
    return back_pos(left_pos(pos, dir_vec), dir_vec)


def back_right_pos(pos, dir_vec):
    """
    Get the position of the pos that is behind and one pos to the right.
    """

    return back_pos(right_pos(pos, dir_vec), dir_vec)


def rotate_right(dir_vec):
    direction = VEC_TO_DIR[dir_vec]
    new_dir = DIR_TO_VEC[(direction + 1) % 4]
    return new_dir[0], new_dir[1]


def rotate_left(dir_vec):
    direction = VEC_TO_DIR[dir_vec]
    new_dir = DIR_TO_VEC[(direction - 1) % 4]
    return new_dir[0], new_dir[1]


def rotate_180(dir_vec):
    direction = VEC_TO_DIR[dir_vec]
    new_dir = DIR_TO_VEC[(direction + 2) % 4]
    return new_dir[0], new_dir[1]

