# Enumeration of possible actions
from __future__ import annotations

from enum import IntEnum


class Effect(object):
    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.direction = direction


class Actions(IntEnum):
    # Turn left, turn right, move forward
    left = 0
    right = 1
    forward = 2
    # Pick up an object
    pickup = 3
    # Drop an object
    drop = 4
    # Toggle/activate an object
    toggle = 5

    # Done completing task
    done = 6

    # Step diagonal left
    diagonal_left = 7

    # Step diagonal right
    diagonal_right = 8

    # Turn and go left
    right_move = 9
    down_move = 10
    left_move = 11
    up_move = 12

    # Turn 180 degrees
    turn_around = 13

    diagonal_backwards_left = 14
    diagonal_backwards_right = 15
    left_move_no_turn = 16
    right_move_no_turn = 17
    backward = 18


class ActionSpace(IntEnum):
    # Standard
    standard = 0

    # No left turns
    no_left = 1

    # No right turns
    no_right = 2

    # Diagonal steps possible
    diagonal = 3

    # WSAD movement (up down left right) turn + move with single action
    wsad = 4

    # All adjacent cells (includes diagonals) + right turn for turning
    dir8 = 5

    # Can go left and right immediately but not forward and backward, but can turn
    left_right = 6

    # Get the legal actions for the action space
    all_diagonal = 7

    # An action space where any action is possible
    all = 8

    def get_legal_actions(self) -> tuple(Actions, ...):
        if self == ActionSpace.standard:
            return (
                Actions.left,
                Actions.right,
                Actions.forward,
                Actions.pickup,
                Actions.drop,
                Actions.toggle,
                Actions.done,
            )
        elif self == ActionSpace.no_left:
            return (
                Actions.right,
                Actions.forward,
                Actions.pickup,
                Actions.drop,
                Actions.toggle,
                Actions.done,
            )
        elif self == ActionSpace.no_right:
            return (
                Actions.left,
                Actions.forward,
                Actions.pickup,
                Actions.drop,
                Actions.toggle,
                Actions.done,
            )
        elif self == ActionSpace.diagonal:
            return (
                Actions.left,
                Actions.right,
                Actions.forward,
                Actions.pickup,
                Actions.drop,
                Actions.toggle,
                Actions.done,
                Actions.diagonal_left,
                Actions.diagonal_right,
            )
        elif self == ActionSpace.wsad:
            return (
                Actions.right_move,
                Actions.down_move,
                Actions.left_move,
                Actions.up_move,
                Actions.pickup,
                Actions.drop,
                Actions.toggle,
                Actions.done,
            )
        elif self == ActionSpace.dir8:
            return (
                Actions.right_move_no_turn,
                Actions.backward,
                Actions.left_move_no_turn,
                Actions.forward,
                Actions.diagonal_backwards_left,
                Actions.diagonal_backwards_right,
                Actions.diagonal_right,
                Actions.diagonal_left,
                Actions.right,
                Actions.pickup,
                Actions.drop,
                Actions.toggle,
                Actions.done,
            )
        elif self == ActionSpace.left_right:
            return (
                Actions.left_move_no_turn,
                Actions.right_move_no_turn,
                Actions.right,
                Actions.pickup,
                Actions.drop,
                Actions.toggle,
                Actions.done,
            )
        elif self == ActionSpace.all_diagonal:
            return (
                Actions.right,
                Actions.right_move_no_turn,
                Actions.diagonal_backwards_left,
                Actions.diagonal_backwards_right,
                Actions.diagonal_right,
                Actions.diagonal_left,
                Actions.pickup,
                Actions.drop,
                Actions.toggle,
                Actions.done,
            )
        elif self == ActionSpace.all:
            return (
                Actions.left,
                Actions.right,
                Actions.forward,
                Actions.pickup,
                Actions.drop,
                Actions.toggle,
                Actions.done,
                Actions.diagonal_left,
                Actions.diagonal_right,
                Actions.right_move,
                Actions.down_move,
                Actions.left_move,
                Actions.up_move,
                Actions.turn_around,
                Actions.diagonal_backwards_left,
                Actions.diagonal_backwards_right,
                Actions.left_move_no_turn,
                Actions.right_move_no_turn,
                Actions.backward,
            )
        else:
            raise RuntimeError(f'Unknown actionspace {self}')


    def get_agent_color(self) -> tuple[int, int, int]:
        if self == ActionSpace.standard:
            return (255, 0, 0)
        elif self == ActionSpace.no_left:
            return (0, 255, 0)
        elif self == ActionSpace.no_right:
            return (0, 0, 255)
        elif self == ActionSpace.diagonal:
            return (255, 255, 0)
        elif self == ActionSpace.wsad:
            return (0, 255, 255)
        elif self == ActionSpace.dir8:
            return (255, 0, 255)
        elif self == ActionSpace.left_right:
            return (100, 100, 100)
        elif self == ActionSpace.all_diagonal:
            return (100, 200, 125)
        elif self == ActionSpace.all:
            return (200, 100, 125)
        else:
            raise RuntimeError(f'Unknown actionspace {self}')


