from enum import IntEnum, unique


@unique
class GameMode(IntEnum):
    """The various game modes in osu!.
    """
    standard = 0
    taiko = 1
    ctb = 2
    mania = 3

    @classmethod
    def parse(cls, mode: str):
        if mode.lower() in ["standard", "std", "osu", "o", "0"]:
            mode_enum = cls.standard
        elif mode.lower() in ["taiko", "t", "1"]:
            mode_enum = cls.taiko
        elif mode.lower() in ["catch", "ctb", "c", "2"]:
            mode_enum = cls.ctb
        elif mode.lower() in ["mania", "m", "3"]:
            mode_enum = cls.mania
        else:
            raise Exception
        return mode_enum

    @classmethod
    def serialize(cls, mode):
        if mode == cls.ctb:
            return "fruits"
        else:
            return str(mode.name)
