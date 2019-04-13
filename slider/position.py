from collections import namedtuple
from datetime import timedelta
import operator as op
from typing import NamedTuple, Any


class Position(NamedTuple):
    """A position on the osu! screen.

    Parameters
    ----------
    x : int or float
        The x coordinate in the range.
    y : int or float
        The y coordinate in the range.

    Notes
    -----
    The visible region of the osu! standard playfield is [0, 512] by [0, 384].
    Positions may fall outside of this range for slider curve control points.
    """
    x: float
    y: float
    x_max = 512
    y_max = 384

    def __repr__(self):
        return (
            f'<{type(self).__qualname__}: ({self.x:.4f}, {self.y:.4f})>'
        )

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def _vec_scl_ops(self, opf, other):
        if isinstance(other, (Position)):
            return Position(opf(self.x, other.x), opf(self.y, other.y))
        elif isinstance(other, (int, float)):
            return Position(opf(self.x, other), opf(self.y, other))
        else:
            return NotImplemented

    def __add__(self, other):
        return self._vec_scl_ops(op.add, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        return self._vec_scl_ops(op.mul, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self._vec_scl_ops(op.sub, other)

    def __rsub__(self, other):
        if isinstance(other, (Position)):
            return other._vec_scl_ops(op.sub, self)
        elif isinstance(other, (int, float)):
            return Position(op.sub(other, self.x), op.sub(other, self.y))
        else:
            return NotImplemented

    def __neg__(self):
        return Position(-self.x, -self.y)

    def __pow__(self, power, modulo=None):
        return Position(pow(self.x, power, modulo), pow(self.y, power, modulo))


class Point(namedtuple('Point', 'x y offset')):
    """A position and time on the osu! screen.

    Parameters
    ----------
    x : int or float
        The x coordinate in the range.
    y : int or float
        The y coordinate in the range.
    offset : int or float
        The time

    Notes
    -----
    The visible region of the osu! standard playfield is [0, 512] by [0, 384].
    Positions may fall outside of this range for slider curve control points.
    """

    def __repr__(self):
        return (
            f'<{type(self).__qualname__}: ({self.x:.4f}, {self.y:.4f}), {self.offset.total_seconds() * 1000:g}ms>'
        )


class Tick(NamedTuple):
    """A position and time on the osu! screen.

    Parameters
    ----------
    position : Position
        The Position of the tick.
    time : timedelta
        The time of the tick.
    parent : Any
        The parent object of the tick
    is_note : bool = False
        Whether the tick is considered a note in osu!catch or not
    """
    position: Position
    time: timedelta
    parent: Any
    is_note: bool = False

    def __repr__(self):
        return (
            f'<{type(self).__qualname__}{" note" if self.is_note else ""}: {self.position}, '
            f'{self.time.total_seconds() * 1000:g}ms>'
        )

    # parent not used due to Slider having no __eq__ and __hash__
    # a tick cannot be in the same position and time anyway
    def __eq__(self, other):
        return self.position == other.position and self.time == other.time and self.is_note == other.is_note

    def __hash__(self):
        return hash((self.position, self.time, self.is_note))
