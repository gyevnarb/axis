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
|0|Road with two parallel lanes and a T-junction coming up ahead.|The ego vehicle starts in the left lane then changes lanes right to prepare for an exit right. Vehicle 1 in the left lane continues, maintaining its speed.|Rational|
|1|Same as Scenario 0|The ego vehicle starts in right lane, when Vehicle 1 cuts in front of it from the left lane and begins to slow down rapidly, indicating its intention to exit right at the junction. In response, the ego vehicle changes lanes left to continue to its goal unimpeded and more safely.|Rational|
