"""Baseline heuristic strategies for wildfire suppression resource allocation.

Each strategy is a callable ``(state, node) -> float`` compatible with :func:`wildfireGP.evaluate.evaluate`. Higher
score means higher treatment priority.

Two tiers
---------
**Naive comparators** --- intentionally weak; included as the "what a non-specialist would propose first" baseline
for a GP audience:

    random_score, score_by_fuel, score_by_burning_neighbors

**Doctrine baselines** --- operationally grounded in real wildfire suppression tactics or fire-behaviour modelling:

    score_by_fire_proximity      (direct attack)
    score_indirect_attack        (indirect attack on fuel ahead of the front)
    score_anchor_flank           (anchor-and-flank: extend line from existing barriers)
    score_bottleneck             (gateway protection: shield large connected unburned regions)
    score_ridgeline              (topographic defence: absolute high ground in range)
    score_uphill_interception    (topographic defence: ground above the fire)
    score_head_fire              (wind-aligned interception)
    score_fire_run               (wind-aligned interception, fuel-weighted)
    score_flank_attack           (lateral attack: perpendicular to wind)
    score_composite_threat       (combined fuel/wind/slope behavioural baseline)

If GP cannot outperform :func:`score_by_fire_proximity` --- the strongest single-feature doctrine baseline --- it is not
producing useful strategies.

True no-treatment baseline
--------------------------
There is no ``no_treatment`` strategy function. A strategy that scores all nodes equally is indistinguishable from
``random_score`` after the shuffle in ``_apply_treatments`` --- treatments are still placed, just randomly. The true
lower bound is a run with ``treatments_per_step=0``, which ``compare_strategies.py`` includes automatically as the
``no_treatment`` row.

References
----------
National Wildfire Coordinating Group. (2004). Fireline Handbook. NWCG Handbook 3, PMS 410-1.
National Wildfire Coordinating Group. (2022). Incident Response Pocket Guide. PMS 461.
Pais, C., Carrasco, J., Martell, D. L., Weintraub, A., & Woodruff, D. L. (2021). Cell2Fire: A Cell-Based Forest
    Fire Growth Model to Support Strategic Landscape Management Planning. Frontiers in Forests and Global Change,
    4, 692706. https://doi.org/10.3389/ffgc.2021.692706
Rothermel, R.C. (1972). A mathematical model for predicting fire spread in wildland fuels. USDA Forest Service
    Research Paper INT-115.
"""

import random

from wildfireGP.evaluate import annotate_required_precomputes
from wildfireGP.features import (
    burning_neighbour_count,
    distance_to_fire,
    elevation,
    elevation_delta_to_fire,
    fuel_level,
    has_treated_neighbour,
    mean_neighbour_fuel,
    reachable_unburned_area,
    slope,
    treated_neighbour_count,
    unburnable_neighbour_count,
    wind_fire_alignment,
)
from wildfireGP.network import GraphState

ANCHOR_WEIGHT = 0.1


def random_score(state: GraphState, node: tuple) -> float:
    """Assign a random priority score --- treatment order is arbitrary.

    Naive comparator: the noise floor. Anything that fails to beat this is not learning.
    """
    return random.random()


def score_by_fuel(state: GraphState, node: tuple) -> float:
    """Prioritise nodes with the highest fuel load.

    Naive comparator: a layperson's first instinct ("treat the heavy fuel"). Ignores fire location, so it spends budget
    on remote high-fuel ground rather than the front. Not an active-suppression heuristic.
    """
    return fuel_level(state, node) + ANCHOR_WEIGHT * has_treated_neighbour(state, node)


def score_by_fire_proximity(state: GraphState, node: tuple) -> float:
    """Prioritise nodes closest to the active fire front (direct attack).

    Reference: NWCG Fireline Handbook (PMS 410-1), 2004 --- direct attack doctrine.
    """
    return -distance_to_fire(state, node) + ANCHOR_WEIGHT * has_treated_neighbour(state, node)


def score_by_burning_neighbors(state: GraphState, node: tuple) -> float:
    """Prioritise nodes already surrounded by burning neighbours.

    Naive comparator: targets cells about to be overrun --- a last-ditch defensive action rather than active firebreak
    placement. Included as a weak doctrine-shaped floor.
    """
    return float(burning_neighbour_count(state, node)) + ANCHOR_WEIGHT * has_treated_neighbour(state, node)


def score_indirect_attack(state: GraphState, node: tuple, min_distance: int = 2, max_distance: int = 10) -> float:
    """Prioritise fuel-rich ground just ahead of the fire front (indirect attack).

    Reference: NWCG Fireline Handbook (PMS 410-1), 2004 --- indirect attack doctrine.
    """
    d = distance_to_fire(state, node)
    if d < min_distance or d > max_distance:
        return 0.0
    return mean_neighbour_fuel(state, node) / (1.0 + d) + ANCHOR_WEIGHT * has_treated_neighbour(state, node)


def score_anchor_flank(state: GraphState, node: tuple, min_distance: int = 2, max_distance: int = 10) -> float:
    """Prioritise burnable cells adjacent to existing barriers (anchor-and-flank line construction).

    Real-world doctrine: build line from an anchor point (water, rock, burned ground, existing line) and extend it
    along the flanks so the fire cannot outflank crews. This baseline rewards cells in fuel, within engagement range
    of the fire, adjacent to unburnable or already-treated neighbours --- i.e. cells that extend a continuous barrier
    into burnable ground.

    References: NWCG Fireline Handbook (PMS 410-1), 2004; NWCG Incident Response Pocket Guide (PMS 461), 2022.
    """
    d = distance_to_fire(state, node)
    if d < min_distance or d > max_distance:
        return 0.0
    barriers = float(unburnable_neighbour_count(state, node) + treated_neighbour_count(state, node))
    return fuel_level(state, node) * barriers / (1.0 + d)


def score_bottleneck(state: GraphState, node: tuple, min_distance: int = 2, max_distance: int = 10) -> float:
    """Prioritise cells that gate access to large connected unburned regions (gateway protection).

    Indirect-attack reasoning generalised to landscape topology: a cell whose treatment denies the fire access to a
    large connected unburned component is worth more than one fronting a small pocket. Approximates the
    betweenness-centrality / minimum-travel-time placement heuristics used in landscape fuel-management planning.

    Reference: Pais et al. (2021), Cell2Fire --- discusses graph-theoretic placement heuristics including
    centrality-based approaches for strategic fuel treatment.
    """
    d = distance_to_fire(state, node)
    if d < min_distance or d > max_distance:
        return 0.0
    return reachable_unburned_area(state, node) * fuel_level(state, node) / (
        1.0 + d
    ) + ANCHOR_WEIGHT * has_treated_neighbour(state, node)


def score_ridgeline(state: GraphState, node: tuple, min_distance: int = 2, max_distance: int = 10) -> float:
    """Prioritise topographic high points within engagement range of the fire (ridgeline defence).

    Reference: NWCG Fireline Handbook (PMS 410-1), 2004 --- ridge-top line construction.
    """
    d = distance_to_fire(state, node)
    if d < min_distance or d > max_distance:
        return 0.0
    return elevation(state, node) + slope(state, node) + ANCHOR_WEIGHT * has_treated_neighbour(state, node)


def score_uphill_interception(state: GraphState, node: tuple, min_distance: int = 2, max_distance: int = 10) -> float:
    """Prioritise ground above the fire within engagement range (upslope interception).

    Fire-behaviour reasoning: spread rate increases roughly exponentially with slope, so the fire will run uphill
    preferentially. Defending ground above the fire intercepts it where it accelerates. Distinct from
    :func:`score_ridgeline`, which rewards absolute elevation regardless of fire position.

    Reference: Rothermel (1972) --- slope as a multiplicative spread-rate factor.
    """
    d = distance_to_fire(state, node)
    if d < min_distance or d > max_distance:
        return 0.0
    return max(elevation_delta_to_fire(state, node), 0.0) / (1.0 + d) + ANCHOR_WEIGHT * has_treated_neighbour(
        state, node
    )


def score_fire_run(state: GraphState, node: tuple, min_distance: int = 2, max_distance: int = 10) -> float:
    """Prioritise fuel-rich nodes downwind of the fire within engagement range.

    Reference: Rothermel (1972) --- wind and fuel as multiplicative spread-rate factors.
    """
    d = distance_to_fire(state, node)
    if d < min_distance or d > max_distance:
        return 0.0
    return fuel_level(state, node) * wind_fire_alignment(state, node) / (
        1.0 + d
    ) + ANCHOR_WEIGHT * has_treated_neighbour(state, node)


def score_head_fire(state: GraphState, node: tuple, min_distance: int = 2, max_distance: int = 10) -> float:
    """Prioritise nodes downwind and within engagement range of the fire (head-fire interception).

    Reference: NWCG Fireline Handbook (PMS 410-1), 2004 --- head-fire identification and engagement.
    """
    d = distance_to_fire(state, node)
    if d < min_distance or d > max_distance:
        return 0.0
    return wind_fire_alignment(state, node) / (1.0 + d) + ANCHOR_WEIGHT * has_treated_neighbour(state, node)


def score_flank_attack(state: GraphState, node: tuple, min_distance: int = 2, max_distance: int = 10) -> float:
    """Prioritise lateral positions perpendicular to wind (flank attack).

    Real-world doctrine: when the head fire is too intense to engage directly, attack the flanks --- which spread
    laterally rather than downwind and are therefore lower-intensity. This baseline rewards fuel-rich cells in range
    of the fire where the fire-to-cell direction is perpendicular to the wind (alignment near zero).

    Reference: NWCG Fireline Handbook (PMS 410-1), 2004 --- flanking action.
    """
    d = distance_to_fire(state, node)
    if d < min_distance or d > max_distance:
        return 0.0
    return fuel_level(state, node) * (1.0 - abs(wind_fire_alignment(state, node))) / (
        1.0 + d
    ) + ANCHOR_WEIGHT * has_treated_neighbour(state, node)


def score_composite_threat(state: GraphState, node: tuple, min_distance: int = 2, max_distance: int = 10) -> float:
    """Prioritise cells combining high fuel, steep slope, and downwind alignment (composite threat).

    Fire-behaviour baseline rather than a named field tactic: combines the three spread drivers in the Rothermel
    model multiplicatively. A serious comparator for GP, since it uses the same drivers a fire-behaviour predictor
    would weight.

    Reference: Rothermel (1972) --- fuel, wind, and slope as multiplicative spread-rate factors.
    """
    d = distance_to_fire(state, node)
    if d < min_distance or d > max_distance:
        return 0.0
    return fuel_level(state, node) * slope(state, node) * max(wind_fire_alignment(state, node), 0.0) / (
        1.0 + d
    ) + ANCHOR_WEIGHT * has_treated_neighbour(state, node)


ALL_STRATEGIES = [
    random_score,
    score_by_fuel,
    score_by_fire_proximity,
    score_by_burning_neighbors,
    score_indirect_attack,
    score_anchor_flank,
    score_bottleneck,
    score_ridgeline,
    score_uphill_interception,
    score_head_fire,
    score_fire_run,
    score_flank_attack,
    score_composite_threat,
]

annotate_required_precomputes(random_score, [])
annotate_required_precomputes(score_by_fuel, ["has_treated_neighbour"])
annotate_required_precomputes(score_by_fire_proximity, ["distance_to_fire", "has_treated_neighbour"])
annotate_required_precomputes(score_by_burning_neighbors, ["burning_neighbour_count", "has_treated_neighbour"])
annotate_required_precomputes(
    score_indirect_attack, ["distance_to_fire", "mean_neighbour_fuel", "has_treated_neighbour"]
)
annotate_required_precomputes(
    score_anchor_flank, ["distance_to_fire", "unburnable_neighbour_count", "treated_neighbour_count"]
)
annotate_required_precomputes(
    score_bottleneck, ["distance_to_fire", "reachable_unburned_area", "has_treated_neighbour"]
)
annotate_required_precomputes(score_ridgeline, ["distance_to_fire", "has_treated_neighbour"])
annotate_required_precomputes(
    score_uphill_interception, ["distance_to_fire", "elevation_delta_to_fire", "has_treated_neighbour"]
)
annotate_required_precomputes(score_fire_run, ["distance_to_fire", "wind_fire_alignment", "has_treated_neighbour"])
annotate_required_precomputes(score_head_fire, ["distance_to_fire", "wind_fire_alignment", "has_treated_neighbour"])
annotate_required_precomputes(score_flank_attack, ["distance_to_fire", "wind_fire_alignment", "has_treated_neighbour"])
annotate_required_precomputes(
    score_composite_threat, ["distance_to_fire", "wind_fire_alignment", "has_treated_neighbour"]
)
