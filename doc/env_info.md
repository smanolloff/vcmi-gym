# Environment documentation

> [!WARNING]
> This project is still in active development and its documentation may be
> outdated. It reflects the project's state as of March 2024, but frequent
> code changes make it hard to maintain an up-to-date documentation at
> this stage.

## API

`vcmi-gym` implements the Gym API, please refer to the
[Gymnasium](https://gymnasium.farama.org/) documentation for reference.

## Observation space

vcmi-gym uses a
[`Box`](https://gymnasium.farama.org/api/spaces/fundamental/#box)
observation space with shape `(11, 15, E)`, corresponding to the battlefield's
11x15 hex grid (165 hexes total), where `E` is the hex encoding size (as of
this writing, `E` is 574).

<p align="center"><img src="hexes.jpg" alt="hexes" height="300"></p>

Every hex is described by `N` attributes, each of which is one-hot encoded as
follows:


| Attribute name                            | Encoding type     | Encoded size |
| ----------------------------------------- | ----------------- | ------------ |
| HEX_Y_COORD                               | CATEGORICAL       | 11           |
| HEX_X_COORD                               | CATEGORICAL       | 15           |
| HEX_STATE \*                              | CATEGORICAL       | 3            |
| HEX_ACTION_MASK_FOR_ACT_STACK             | BINARY            | 14           |
| HEX_ACTION_MASK_FOR_L_STACK_0             | BINARY            | 14           |
| HEX_ACTION_MASK_FOR_L_STACK_1             | BINARY            | 14           |
| HEX_ACTION_MASK_FOR_L_STACK_2             | BINARY            | 14           |
| HEX_ACTION_MASK_FOR_L_STACK_3             | BINARY            | 14           |
| HEX_ACTION_MASK_FOR_L_STACK_4             | BINARY            | 14           |
| HEX_ACTION_MASK_FOR_L_STACK_5             | BINARY            | 14           |
| HEX_ACTION_MASK_FOR_L_STACK_6             | BINARY            | 14           |
| HEX_ACTION_MASK_FOR_R_STACK_0             | BINARY            | 14           |
| HEX_ACTION_MASK_FOR_R_STACK_1             | BINARY            | 14           |
| HEX_ACTION_MASK_FOR_R_STACK_2             | BINARY            | 14           |
| HEX_ACTION_MASK_FOR_R_STACK_3             | BINARY            | 14           |
| HEX_ACTION_MASK_FOR_R_STACK_4             | BINARY            | 14           |
| HEX_ACTION_MASK_FOR_R_STACK_5             | BINARY            | 14           |
| HEX_ACTION_MASK_FOR_R_STACK_6             | BINARY            | 14           |
| HEX_MELEEABLE_BY_ACT_STACK                | CATEGORICAL       | 3            |
| HEX_MELEEABLE_BY_L_STACK_0                | CATEGORICAL       | 3            |
| HEX_MELEEABLE_BY_L_STACK_1                | CATEGORICAL       | 3            |
| HEX_MELEEABLE_BY_L_STACK_2                | CATEGORICAL       | 3            |
| HEX_MELEEABLE_BY_L_STACK_3                | CATEGORICAL       | 3            |
| HEX_MELEEABLE_BY_L_STACK_4                | CATEGORICAL       | 3            |
| HEX_MELEEABLE_BY_L_STACK_5                | CATEGORICAL       | 3            |
| HEX_MELEEABLE_BY_L_STACK_6                | CATEGORICAL       | 3            |
| HEX_MELEEABLE_BY_R_STACK_0                | CATEGORICAL       | 3            |
| HEX_MELEEABLE_BY_R_STACK_1                | CATEGORICAL       | 3            |
| HEX_MELEEABLE_BY_R_STACK_2                | CATEGORICAL       | 3            |
| HEX_MELEEABLE_BY_R_STACK_3                | CATEGORICAL       | 3            |
| HEX_MELEEABLE_BY_R_STACK_4                | CATEGORICAL       | 3            |
| HEX_MELEEABLE_BY_R_STACK_5                | CATEGORICAL       | 3            |
| HEX_MELEEABLE_BY_R_STACK_6                | CATEGORICAL       | 3            |
| HEX_SHOOT_DISTANCE_FROM_ACT_STACK         | CATEGORICAL       | 3            |
| HEX_SHOOT_DISTANCE_FROM_L_STACK_0         | CATEGORICAL       | 3            |
| HEX_SHOOT_DISTANCE_FROM_L_STACK_1         | CATEGORICAL       | 3            |
| HEX_SHOOT_DISTANCE_FROM_L_STACK_2         | CATEGORICAL       | 3            |
| HEX_SHOOT_DISTANCE_FROM_L_STACK_3         | CATEGORICAL       | 3            |
| HEX_SHOOT_DISTANCE_FROM_L_STACK_4         | CATEGORICAL       | 3            |
| HEX_SHOOT_DISTANCE_FROM_L_STACK_5         | CATEGORICAL       | 3            |
| HEX_SHOOT_DISTANCE_FROM_L_STACK_6         | CATEGORICAL       | 3            |
| HEX_SHOOT_DISTANCE_FROM_R_STACK_0         | CATEGORICAL       | 3            |
| HEX_SHOOT_DISTANCE_FROM_R_STACK_1         | CATEGORICAL       | 3            |
| HEX_SHOOT_DISTANCE_FROM_R_STACK_2         | CATEGORICAL       | 3            |
| HEX_SHOOT_DISTANCE_FROM_R_STACK_3         | CATEGORICAL       | 3            |
| HEX_SHOOT_DISTANCE_FROM_R_STACK_4         | CATEGORICAL       | 3            |
| HEX_SHOOT_DISTANCE_FROM_R_STACK_5         | CATEGORICAL       | 3            |
| HEX_SHOOT_DISTANCE_FROM_R_STACK_6         | CATEGORICAL       | 3            |
| HEX_MELEE_DISTANCE_FROM_ACT_STACK         | CATEGORICAL       | 3            |
| HEX_MELEE_DISTANCE_FROM_L_STACK_0         | CATEGORICAL       | 3            |
| HEX_MELEE_DISTANCE_FROM_L_STACK_1         | CATEGORICAL       | 3            |
| HEX_MELEE_DISTANCE_FROM_L_STACK_2         | CATEGORICAL       | 3            |
| HEX_MELEE_DISTANCE_FROM_L_STACK_3         | CATEGORICAL       | 3            |
| HEX_MELEE_DISTANCE_FROM_L_STACK_4         | CATEGORICAL       | 3            |
| HEX_MELEE_DISTANCE_FROM_L_STACK_5         | CATEGORICAL       | 3            |
| HEX_MELEE_DISTANCE_FROM_L_STACK_6         | CATEGORICAL       | 3            |
| HEX_MELEE_DISTANCE_FROM_R_STACK_0         | CATEGORICAL       | 3            |
| HEX_MELEE_DISTANCE_FROM_R_STACK_1         | CATEGORICAL       | 3            |
| HEX_MELEE_DISTANCE_FROM_R_STACK_2         | CATEGORICAL       | 3            |
| HEX_MELEE_DISTANCE_FROM_R_STACK_3         | CATEGORICAL       | 3            |
| HEX_MELEE_DISTANCE_FROM_R_STACK_4         | CATEGORICAL       | 3            |
| HEX_MELEE_DISTANCE_FROM_R_STACK_5         | CATEGORICAL       | 3            |
| HEX_MELEE_DISTANCE_FROM_R_STACK_6         | CATEGORICAL       | 3            |
| STACK_QUANTITY                            | NUMERIC_SQRT      | 31           |
| STACK_ATTACK                              | NUMERIC_SQRT      | 7            |
| STACK_DEFENSE                             | NUMERIC_SQRT      | 7            |
| STACK_SHOTS                               | NUMERIC_SQRT      | 5            |
| STACK_DMG_MIN                             | NUMERIC_SQRT      | 8            |
| STACK_DMG_MAX                             | NUMERIC_SQRT      | 8            |
| STACK_HP                                  | NUMERIC_SQRT      | 28           |
| STACK_HP_LEFT                             | NUMERIC_SQRT      | 28           |
| STACK_SPEED                               | NUMERIC           | 23           |
| STACK_WAITED                              | CATEGORICAL       | 2            |
| STACK_QUEUE_POS                           | NUMERIC           | 15           |
| STACK_RETALIATIONS_LEFT                   | NUMERIC           | 3            |
| STACK_SIDE                                | CATEGORICAL       | 2            |
| STACK_SLOT                                | CATEGORICAL       | 7            |
| STACK_CREATURE_TYPE                       | CATEGORICAL       | 145          |
| STACK_AI_VALUE_TENTH                      | NUMERIC_SQRT      | 62           |
| STACK_IS_ACTIVE                           | CATEGORICAL       | 2            |
| STACK_IS_WIDE                             | CATEGORICAL       | 2            |
| STACK_FLYING                              | CATEGORICAL       | 2            |
| STACK_NO_MELEE_PENALTY                    | CATEGORICAL       | 2            |
| STACK_TWO_HEX_ATTACK_BREATH               | CATEGORICAL       | 2            |
| STACK_BLOCKS_RETALIATION                  | CATEGORICAL       | 2            |
| STACK_DEFENSIVE_STANCE                    | CATEGORICAL       | 2            |


\* The 3 hex states are: obstacle / occupied / free

\*\* "meleeable" indicates if stack can reach (or already stands on) a
neighbouring hex. The 3 values are "none" (can't reach), "half dmg" and
"full dmg".

> [!NOTE]
> The above table is only a snapshot of the observation as of the time of this
> writing. For an up-to-date info, you can call `env.attribute_mapping`
> which returns a `name: (encoding, offset, size, max_value)` mapping.

```python
from vcmi_gym import VcmiEnv

env = VcmiEnv("gym/A2.vmap")
obs0, _info = env.reset()

""" Decoded observation data is 2-D list of Hex objects:
    simple namedtuples with fields as per the `HexEncoding` attributes here:
    https://github.com/smanolloff/vcmi/blob/mmai/AI/MMAI/export.h """
obs = VcmiEnv.decode_obs(obs0)  # equivalent to `obs = env.decode()`

""" Get hex 44 (Y=2, X=14). """
x = obs.get(44)
# or obs.get(2, 14)
# or obs.get[2][14]

""" Get the speed of the stack on that hex. """
x.STACK_SPEED
# => 12

""" Get the quantity of the stack on Y=2, X=14.
    STACK_QUANTITY is encoded via a NUMERIC_SQRT encoding.
    It is a lossy encoding which can only give us a value range. """
x.STACK_QUANTITY
# => (1, 4)

""" Check if the stack on Y=5, X=5 can fly. """
x.STACK_FLYING
# => 1

""" Dump hex data in a human-friendly format """
x.dump()
# HEX_Y_COORD                       | 2
# HEX_X_COORD                       | 14
# HEX_STATE                         | OCCUPIED
# HEX_ACTION_MASK_FOR_R_STACK_2     | MOVE
# HEX_MELEEABLE_BY_R_STACK_2        | FULL
# HEX_MELEEABLE_BY_R_STACK_3        | FULL
# STACK_QUANTITY                    | (1, 4)
# STACK_ATTACK                      | (16, 25)
# STACK_DEFENSE                     | (16, 25)
# STACK_SHOTS                       | (0, 1)
# STACK_DMG_MIN                     | (49, 64)
# STACK_DMG_MAX                     | (49, 64)
# STACK_HP                          | (196, 225)
# STACK_HP_LEFT                     | (196, 225)
# STACK_SPEED                       | 12
# STACK_WAITED                      | 0
# STACK_QUEUE_POS                   | 1
# STACK_RETALIATIONS_LEFT           | 1
# ...
```

## Action space

vcmi-gym uses a
[`Discrete`](https://gymnasium.farama.org/api/spaces/fundamental/#discrete)
action space with a total of 2311 actions which is better thought of
`1` non-hex action + `2310` hex actions.

The non-hex action is `WAIT` and has value `0`. The remaining values are
used for one of the 14 actions for each hex (total for 165 hexes:
`165 * 14 = 2310`)

For a given Hex ID (0..164), the action value is: `hex_id * 14 + (1 + action_index)`:

|Action index|Description|
|------|-----------|
|0..11|Move to hex and attack at direction 0..11\*|
|12|Move to hex|
|13|Shoot at hex|

e.g. Moving to hex with ID=2 (X=2, Y=0) is described by the action `41`.

```python
bf = env.decode()

# the numeric action for "move to hex 2":
bf.get(2).action(Action.MOVE)

# the numeric action for "move to Y=8, X=12 and attack right"
action = env.decode().get(8, 12).action(Action.AMOVE_R)

# execute the action
env.step(action)
env.render()
```


\* The 12 attack directions are as follows: 0..5 are the hexes that surround
the current unit, while 7..11 are special cases for 2-hex units (3 per side):

<p align="center">
<img src="attacks1.jpg" alt="attacks1" height=200>
<img src="attacks2.jpg" alt="attacks3" height=200>
<img src="attacks3.jpg" alt="attacks2" height=200>
</p>

## Action masking

The env object also exposes the `action_mask()` method which is not part of
the Gym API, but is useful for certain Reinforcement Learning scenarios where
invalid actions are masked in order to improve learning performance.

The method returns an `np.array` with 2311 `bool` values, indicating the
validity of the corresponding action (`True` means the action is valid).


## Rendering

The gym env supports only one type of rendering: the ANSI render.

It is intended to be rendered in terminals with ANSI color code support,
unicode support and monospaced font:

```python
from vcmi_gym import VcmiEnv, Action

env = VcmiEnv("gym/A2.vmap")
env.reset()
env.render()
```

<img src="render.jpg" alt="render">

> [!TIP]
> If your output looks unaligned, try changing the font of your terminal


## Test Helper

If you want to test the env by playing manually, a convenient helper is provided:

```python
from vcmi_gym import VcmiEnv, Action, TestHelper

env = VcmiEnv("gym/A2.vmap");
h = TestHelper(env)

h.wait()
h.amove(2, 1, Action.AMOVE_R)
h.defend()
# ... etc
```
