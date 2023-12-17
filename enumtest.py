import enum


class Role(enum.Enum):
    ATTACKER = enum.auto()
    DEFENDER = enum.auto()


print(Role.ATTACKER.value)
print(Role.DEFENDER.value)

x = {Role.ATTACKER: 4}
print(x)
print(list(Role))
