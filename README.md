# Agentic Explanations with Simulations

Welcome to the code repository of the AXS framework: an agentic system to generate causal explanations for any multi-agent system.

AXS comes with out-of-the-box support for existing RL environments that are implemented in either `gymansium` or `pettingzoo`. To get started in these environments, see [Usage](#usage).

While the AXS framework may work out-of-the box, it is best to use the object-oriented abstractions to override built-in behaviour, so you can customise your explanations to your need. For more on this, [Customisation](#customisation).

For our implementation and experiments with autonomous driving and the [IGP2](https://github.com/uoe-agents/IGP2)  planner, see [Autonomous driving](#autonomous-driving).

## Requirements

The AXS framework requires Python>=3.9 and various additional dependencies as described in the [pyproject.toml](pyproject.toml) file.

## Installation

_Note: to install dependencies for autonomous driving, use the [igp2] dependency flag after `axs`, such as `pip install axs[igp2]`._

You can install the current stable version of AXS with:
```bash
pip install axs
```

If you prefer to use the latest version, we recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) to install AXS.
If you are using uv, then the following command installs the latest version into your current virutal environment:

```bash
uv add git+https://github.com/gyevnarb/axs
```

You can alternatively clone the repository then install the package into your environment using `pip` and the following command:

```bash
pip install git+https://github.com/gyevnarb/axs
```

## Usage

To use AXS, you will need an environment and some working agents in the environment. If you are using the `gymnasium` or the `pettingzoo` libraries then you can follow the snippets to get started with AXS.

### Use with `gymnasium`

### Use with `pettingzoo`

## Customisation

## Autonomous driving

### Scenarios

The AXS framework comes with 10 unique driving scenarios based on the [IGP2](https://github.com/uoe-agents/IGP2) planning algorithm.
These may be categorised into three types:

- Rational: in which all vehicles plan and act rationally.
- Irrational: in which a non-ego agent is acting irrationally or erratically.
- Occluded: in which some vehicles are ocluded from the ego's viewpoint.

The scenario road layout are defined in the [maps](data/igp2/maps/) folder using the [OpenDRIVE](https://www.asam.net/standards/detail/opendrive/) standard.
Some scenarios might share the same map.
The scenario behaviour are defined in the [configuration](data/igp2/configs/) using a non-standard description format collected into a JSON-file.
The behaviour description format is described in detail in the IGP2 [documentation](https://uoe-agents.github.io/IGP2/).

In each scenario, the ego vehicle, i.e. the agent being explained, is Vehicle 0.
The table below gives a full description summary of each scenario:

|ID|Road Layout|Description|Category|
|---|---|---|---|
|0|Road with two parallel lanes and a T-junction coming up ahead.|Two vehicles. The ego vehicle starts in the left lane then changes lanes right to prepare for an exit right. Vehicle 1 in the left lane continues, maintaining its speed.|Rational|
|1|Same as Scenario 0|Three vehicles. The ego vehicle starts in right lane, when Vehicle 1 cuts in front of it from the left lane and begins to slow down rapidly, indicating its intention to exit right at the junction. In response, the ego vehicle changes lanes left to continue to its goal unimpeded and more safely. Vehicle 2 is giving way at the T-junction to both the ego vehicle and vehicle 1|Rational|
|2|Four-way crossroads between a main road and two lower-priority roads. Each road has two lanes, with one lane in each driving direction. The roads are perpendicular to one another.|Three vehicles. The ego vehicle is approaching the crossroads from a low-priority road. Vehicle 1 is on the main road, slowing down to a stop as it approaches the crossroads. Vehicle 2 is going down the main road in the opposite direction of vehicle 1 while mainting its speed close to the speed limit. The ego vehicle realises that vehicle 1 is trying to turn left and is giving way to vehicle 2 on the main road. The ego vehicle can use this opportunity to turn right early without waiting for vehicle 1 to turn.|Rational|
|3|Roundabout with three exits and two lans.|Two vehicles. The ego vehicle is approaching the roundabout from the south. Vehicle 1 is in the roundabout in the inner lane at the start but changes lanes to the left. The ego vehicle in reaction enters the rounadbout without giving way to vehicle 1 as it infers the intention of vehicle 1 to exit the roundabout which was indiciated by vehicle 1's left lane change.|Rational|
|4|T-junction followed by a four-way crossroads.|Five vehicles. Two vehicles are waiting in line at the four-way crossroads at a traffic light. Vehicle 3 is approaching behind. Vehicle 4 is passing through the four-way crossroads in the opposite direction of the waiting cars. The ego vehicle is approaching from the T-junction and is aiming to merge behind the waiting line of cars. Vehicle 3 sees this and slows down to stop, leaving a gap for the ego vehicle to merge. The ego vehicle realises this and uses the gap to merge behind the waiting line of cars.|Rational|
|5|Same as scenario 2.|The scenario is the same as scenario 2, however vehicle 1 after slowing down decided to speed back and head straight which results in a collision with the turining ego vehicle, which thought vehicle 1 is about to turn left.|Irrational|
|6|Same as scenario 1.|The scenario is the same as scenario 1, however vehicle 1 is changing lanes back and forth on the main road. In response, the ego vehicle stays in the right lane instead of changing lane to the left and keeps some distance from vehicle 1.|Irrational|
|7|Same as scenario 1.|Three vehicles. There is a parked vehicle 2 in the left lane of the two-lane main road. Behind it is vehicle 1 and then the ego vehicle. Vehicle 1 is blocking the view of vehicle 2 from the perspective of the ego vehicle. However, the ego vehicle observes vehicle 1 changing lanes to the right which would otherwise not be rational unless vehicle 2 were present, so to avoid the inferred parked vehicle, the ego vehicle also changes lanes.|Occlusion|
|8|Same as scenario 2 with the addition of a large building that blocks the view of the ego vehicle to the left.|Three vehicles. The ego vehicle is approaching from a low-priority road and can observe vehicle 1 on the main road from the right coming to a rolling stop. However, ego vehicle cannot see what is on the left because the building blocks its view. Still, from the actions of vehicle 1, it infers that there is a oncoming vehicle on the main road from the left. Therefore, the ego vehicle stops to give way to the inferred vehicle.|Occlusion|
|9|Same as scenario 2 with the addition of a large building that blocks the view of the ego vehicle to the right.|Three vehicles. The ego vehicle is approaching the main road from a low-priority road. It observes vehicle 1 on the main road from the left coming to a full stop for a longer amount of time. As vehicle 1 does not seem to want to turn and there is no vehicle appearing from the right side of the main road, the ego vehicle infers that the only rational reason for vehicle 1 to stop would be that there is a vehicle blocking its path occluded by the building on the right. Once the ego inferrs this, it decides to turn left, using the gap that vehicle 1 has left at the crossroads.|Occlusion|

