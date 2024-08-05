# Connector documentation

Here you will find general information about the VCMI connector, why it is
needed and how it works.

## What is it?

The connector is the "man-in-the-middle" for all communication between
`VcmiGym` (Python gym environment) and VCMI (C++ library):

<img src="components-connector.png" alt="components-connector" height="150px">

Internally, the Connector is composed of two parts: 
* PyConnector
* CppConnector

Together, they form the "link" between the gym env and VCMI.

## Why is it needed?

If we take VcmiGym's point of view, communication with VCMI is as simple as
calling `vcmi.get_state(action)` to obtain the new environment state:

<img src="connector-pov-gym.png" alt="connector-pov-gym" height="300px">

Doing the same with VCMI's point of view, communication with VcmiGym
(the "AI") is as simple as calling `ai.getAction(state)` to obtain next action:

<img src="connector-pov-vcmi.png" alt="connector-pov-vcmi" height="250px">

The problem is: both components assume the role of the *caller* in this
state-action exchange and that simply can't happen in practice without some
kind of a buffer in between. That "buffer" is the Connector and here's how it
works:

<img src="connector-pov.png" alt="connector-pov" height="350px">

Although more stuff is happening under the hood, in a nutshell, two different
threads are synchronized with the help of locks and condvars. Here's the same
call to `step()` with some implementation details:

<img src="connector-details-step.png" alt="connector-details-step" height="500px">

There's several more of those diagrams. They really helped me during the
connector development phase, so others (and future me) might find them useful
as well:

#### Connector details - VCMI init:

<img src="connector-details-init.png" alt="connector-details-init" height="500px">

#### Connector details - render:

<img src="connector-details-render.png" alt="connector-details-render" height="500px">

#### Connector details - reset (after battle end):

<img src="connector-details-reset-endbattle.png" alt="connector-details-reset-endbattle" height="500px">

#### Connector details - reset (mid-battle):

<img src="connector-details-reset-midbattle.png" alt="connector-details-reset-midbattle" height="500px">
