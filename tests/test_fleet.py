import pytest

from sim.fleet import FleetConfig, generate_fleet, lookalike_groups


def test_plates_unique():
    fleet = generate_fleet()
    plates = [v.plate for v in fleet]
    assert len(plates) == len(set(plates))


def test_lookalike_clusters_share_class_attrs():
    fleet = generate_fleet()
    groups = lookalike_groups(fleet)
    assert len(groups) == 3
    for members in groups.values():
        attrs = {tuple(sorted(m.class_attrs.items())) for m in members}
        assert len(attrs) == 1, "cluster members must be visually identical at class level"


def test_each_cluster_has_confusable_unmarked_pair():
    # The reasoning layer needs pairs that appearance alone cannot separate:
    # same class attrs AND no distinguishing instance attributes.
    fleet = generate_fleet()
    for members in lookalike_groups(fleet).values():
        unmarked = [m for m in members if not m.instance_attrs]
        assert len(unmarked) >= 1
    sizes = [len(m) for m in lookalike_groups(fleet).values()]
    assert sum(sizes) >= 8


def test_deterministic_by_seed():
    a = generate_fleet(FleetConfig(seed=42))
    b = generate_fleet(FleetConfig(seed=42))
    c = generate_fleet(FleetConfig(seed=43))
    assert a == b
    assert a != c


def test_rejects_fully_marked_cluster():
    with pytest.raises(ValueError):
        generate_fleet(FleetConfig(lookalike_clusters=((2, 2),)))
