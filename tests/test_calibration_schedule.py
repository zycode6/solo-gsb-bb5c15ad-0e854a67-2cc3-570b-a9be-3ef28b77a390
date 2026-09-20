"""校准排期规划纯算法单测：禁排规范化、可排域切分、单段最少点数与字典序择优。"""
from __future__ import annotations

import itertools
import random

import pytest

from app.calibration import (
    ScheduleInfeasibleError,
    free_segments,
    normalize_blackouts,
    plan_gap_points,
)


# ---------- 禁排时段规范化合并 ----------

def test_normalize_blackouts_sorts_and_merges_overlaps():
    assert normalize_blackouts([(10, 20), (5, 15), (30, 40), (14, 25)]) == [
        (5, 25), (30, 40),
    ]


def test_normalize_blackouts_keeps_touching_intervals_separate():
    # 端点相接：公共端点不在任一区间内部，必须保持可排，故不合并
    assert normalize_blackouts([(10, 20), (20, 30)]) == [(10, 20), (20, 30)]


def test_normalize_blackouts_drops_zero_length_intervals():
    assert normalize_blackouts([(10, 10), (5, 15), (20, 20)]) == [(5, 15)]


def test_normalize_blackouts_nested_and_unsorted():
    assert normalize_blackouts([(0, 100), (10, 20), (50, 60)]) == [(0, 100)]
    assert normalize_blackouts([]) == []


# ---------- 可排域切分（禁排取内部，边界可排） ----------

def test_free_segments_basic():
    assert free_segments(0, 100, [(30, 70)]) == [(0, 30), (70, 100)]


def test_free_segments_blackout_starting_at_window_left_leaves_isolated_point():
    # 禁排从窗口边界开始：左边界点 0 是孤立可排点
    assert free_segments(0, 100, [(0, 50)]) == [(0, 0), (50, 100)]


def test_free_segments_blackout_ending_at_window_right_leaves_isolated_point():
    assert free_segments(0, 100, [(50, 100)]) == [(0, 50), (100, 100)]


def test_free_segments_touching_blackouts_keep_shared_endpoint_schedulable():
    assert free_segments(0, 100, [(30, 50), (50, 80)]) == [(0, 30), (50, 50), (80, 100)]


def test_free_segments_fully_covered_window():
    assert free_segments(0, 100, [(0, 100)]) == [(0, 0), (100, 100)]


def test_free_segments_without_blackouts():
    assert free_segments(0, 100, []) == [(0, 100)]


def test_free_segments_blackout_outside_window():
    assert free_segments(0, 100, [(100, 200)]) == [(0, 100)]
    assert free_segments(0, 100, [(-50, 0)]) == [(0, 100)]


def test_free_segments_degenerate_domain():
    assert free_segments(50, 50, []) == [(50, 50)]


# ---------- 单段排期：最少点数 ----------

def test_plan_gap_no_points_when_gap_within_limit():
    assert plan_gap_points(0, 1000, 1000, []) == []
    assert plan_gap_points(0, 900, 1000, []) == []


def test_plan_gap_points_are_evenly_spaced_when_unconstrained():
    assert plan_gap_points(0, 3000, 1000, []) == [1000.0, 2000.0]
    assert plan_gap_points(0, 5000, 1000, []) == [1000.0, 2000.0, 3000.0, 4000.0]


def test_plan_gap_single_point_takes_earliest_feasible():
    # 间隔 1500 只需 1 点：该点须 ∈ [500,1000]（距右锚 ≤1000、距左锚 ≤1000），
    # 字典序最早取 500
    assert plan_gap_points(0, 1500, 1000, []) == [500.0]


def test_plan_gap_tie_break_picks_lexicographically_earliest_set():
    # 间隔 2500 需 2 点；多组可行方案（如 [500,1500] 与 [1000,2000] 都使间隔 ≤ 1000），
    # 字典序最早为 [500,1500]（朴素“尽量靠右”贪心会得到 [1000,2000]）
    assert plan_gap_points(0, 2500, 1000, []) == [500.0, 1500.0]


def test_plan_gap_tie_break_with_three_points():
    # 间隔 3500 需 3 点；字典序最早为 [500,1500,2500]
    assert plan_gap_points(0, 3500, 1000, []) == [500.0, 1500.0, 2500.0]


def test_plan_gap_blackout_endpoints_are_schedulable():
    # 禁排 (1000,2000)：两个端点都可排，方案 [1000,2000] 恰好达标
    assert plan_gap_points(0, 3000, 1000, [(1000, 2000)]) == [1000.0, 2000.0]


def test_plan_gap_schedules_around_blackout():
    # 禁排 (500,1500)：从右向左，最右点取 [2000,3000) 最早 2000，再取 (0,1000] 内
    # 能接住 2000 的最早点——禁排端点 1500 可排，1500 ∈ [1000,2000) → 1500；
    # 仍需一点接住 a=0：q ∈ (0,500] 且能到 1500（距离 ≤1000）→ q=500（端点可排）
    assert plan_gap_points(0, 3000, 1000, [(500, 1500)]) == [500.0, 1500.0, 2000.0]


def test_plan_gap_schedules_at_blackout_boundaries_longer_span():
    # 禁排 (1000,2000) 跨越 4000 间隔：端点 1000/2000 可排
    assert plan_gap_points(0, 4000, 1000, [(1000, 2000)]) == [1000.0, 2000.0, 3000.0]


def test_plan_gap_infeasible_when_blackout_blocks_first_hop():
    # 禁排 (400,1500)：a=0 之后 max_gap=1000 内可排点至多到端点 400，而 400→1500
    # =1100 > 1000，无法跨过禁排内部，无解
    with pytest.raises(ScheduleInfeasibleError):
        plan_gap_points(0, 3000, 1000, [(400, 1500)])


def test_plan_gap_infeasible_when_blackout_longer_than_max_gap():
    # 禁排 (200,1300) 长 1100 > max_gap：任何可排点对都无法跨过其内部
    with pytest.raises(ScheduleInfeasibleError):
        plan_gap_points(0, 3000, 1000, [(200, 1300)])


def test_plan_gap_infeasible_when_entire_gap_is_blackout_interior():
    # 禁排覆盖整个间隔内部：仅端点 a、b 可排，但它们本身是固定锚点
    with pytest.raises(ScheduleInfeasibleError):
        plan_gap_points(0, 2000, 1000, [(0, 2000)])


def test_plan_gap_domain_clipping_to_window():
    # 外部锚点 0、5000，可排域裁剪到窗口 [1000,4000]
    assert plan_gap_points(0, 5000, 1000, [], lo=1000, hi=4000) == [
        1000.0, 2000.0, 3000.0, 4000.0,
    ]


def test_plan_gap_domain_clipping_infeasible_at_left_boundary():
    # 外部锚点 0 距窗口边界 1500 > max_gap，而窗口内禁止在 1000 之前补点 → 无解
    with pytest.raises(ScheduleInfeasibleError):
        plan_gap_points(0, 5000, 1000, [], lo=1500, hi=4000)


def test_plan_gap_domain_clipping_infeasible_at_right_boundary():
    with pytest.raises(ScheduleInfeasibleError):
        plan_gap_points(0, 5000, 1000, [], lo=1000, hi=3500)


def test_plan_gap_blackout_crossing_domain_left_boundary_forbids_it():
    # 禁排 (900,1100) 越过可排域左边界 1000：1000 严格位于禁排内部，不可排；
    # 首个可排点至少 1100，左锚 0 无法在 max_gap=1000 内到达 → 无解
    with pytest.raises(ScheduleInfeasibleError):
        plan_gap_points(0, 5000, 1000, [(900, 1100)], lo=1000, hi=4000)


def test_plan_gap_blackout_ending_exactly_at_domain_left_boundary_keeps_it():
    # 禁排恰止于 1000：边界点 1000 是端点，仍可排，方案与无禁排时一致
    assert plan_gap_points(0, 5000, 1000, [(900, 1000)], lo=1000, hi=4000) == [
        1000.0, 2000.0, 3000.0, 4000.0,
    ]


def test_plan_gap_blackout_crossing_domain_right_boundary_forbids_it():
    # 禁排 (3900,4100) 越过可排域右边界 4000：4000 不可排；末点至多 3900，
    # 无法在 1000 内接到右锚 5000 → 无解
    with pytest.raises(ScheduleInfeasibleError):
        plan_gap_points(0, 5000, 1000, [(3900, 4100)], lo=1000, hi=4000)


def test_plan_gap_blackout_extending_well_beyond_domain():
    # 左锚 500（在窗口外）、禁排 (-100,1200) 越过窗口边界 1000：1000 不可排，
    # 首个可排点 1200，500→1200=700 ≤ 1000 仍可行；验证远伸出域左侧的禁排按交集生效
    points = plan_gap_points(500, 5000, 1000, [(-100, 1200)], lo=1000, hi=4000)
    chain = [500.0, *points, 5000.0]
    assert points[0] == 1200.0
    assert all(not -100 < t < 1200 for t in points)
    assert all(e - s <= 1000 + 1e-9 for s, e in zip(chain, chain[1:]))


def test_plan_gap_points_never_inside_blackout():
    # 任意输入下规划出的点都严格不在禁排内部且间隔达标（禁排内部长度均 < max_gap）
    blackouts = normalize_blackouts([(800, 1400), (2400, 2600)])
    points = plan_gap_points(0, 4000, 700, blackouts)
    chain = [0.0, *points, 4000.0]
    assert [e - s for s, e in zip(chain, chain[1:])] == [700, 700, 500, 700, 700, 700]
    assert points == [700.0, 1400.0, 1900.0, 2600.0, 3300.0]
    for s, e in zip(chain, chain[1:]):
        assert e - s <= 700 + 1e-9
        for t in (s, e):
            assert not any(a < t < b for a, b in blackouts)


def test_plan_gap_minimum_point_count():
    # 禁排把可排域切碎时仍应取最少点数：6000 间隔、max_gap 1000 无禁排需 5 点
    assert len(plan_gap_points(0, 6000, 1000, [])) == 5


def test_plan_gap_merged_blackouts_consistent_with_piecewise():
    # 传入交叠禁排与先合并后等价的单个禁排，结果一致（合并后长度恰等于 max_gap，
    # 靠两端点跨越）
    merged = normalize_blackouts([(800, 1500), (1400, 1800)])
    assert merged == [(800, 1800)]
    assert plan_gap_points(0, 4000, 1000, merged) == plan_gap_points(
        0, 4000, 1000, [(800, 1800)]
    )
    assert plan_gap_points(0, 4000, 1000, [(800, 1800)]) == [
        800.0, 1800.0, 2000.0, 3000.0,
    ]


# ---------- 与网格暴力枚举对照：最少点数 + 字典序最早 ----------

def _brute_force_lex_min(a, b, g, blackouts, step):
    """整数网格上枚举所有可行时点子集，返回（最少且字典序最早的）方案；无解返回 None。"""
    forbidden = set()
    for lo, hi in blackouts:
        for t in range(int(lo) + 1, int(hi)):  # 禁排内部不可排，端点可排
            forbidden.add(t)
    candidates = [t for t in range(int(a) + step, int(b), step) if t not in forbidden]
    n = len(candidates)

    def feasible(subset):
        chain = [a, *subset, b]
        return all(y - x <= g for x, y in zip(chain, chain[1:]))

    for size in range(0, n + 1):
        for combo in itertools.combinations(candidates, size):
            if feasible(combo):
                return list(combo)
    return None


@pytest.mark.parametrize(
    "a,b,g,blackouts",
    [
        (0, 6, 2, []),
        (0, 7, 3, []),
        (0, 10, 3, [(2, 4)]),
        (0, 10, 3, [(3, 6)]),
        (0, 12, 4, [(2, 5), (8, 9)]),
        (0, 8, 2, [(1, 4)]),    # 禁排内部长度 3 > g：无解
        (0, 15, 4, [(3, 7), (10, 12)]),
        (0, 9, 5, [(0, 4)]),    # 禁排从锚点开始：边界点可排
        (0, 9, 5, [(5, 9)]),    # 禁排止于锚点
        (0, 20, 6, [(4, 10), (10, 14)]),  # 端点相接
    ],
)
def test_plan_gap_matches_brute_force_lex_min(a, b, g, blackouts):
    merged = normalize_blackouts(blackouts)
    expected = _brute_force_lex_min(a, b, g, merged, 1)
    if expected is None:
        with pytest.raises(ScheduleInfeasibleError):
            plan_gap_points(float(a), float(b), float(g), merged)
    else:
        result = plan_gap_points(float(a), float(b), float(g), merged)
        assert result == pytest.approx([float(t) for t in expected])
        # 解确实可行：所有间隔 ≤ max_gap，且无点落在禁排内部
        chain = [float(a), *result, float(b)]
        assert all(e - s <= g + 1e-9 for s, e in zip(chain, chain[1:]))
        assert all(not any(lo < t < hi for lo, hi in merged) for t in result)


# ---------- 随机化对照：贪心解必须与网格暴力枚举的字典序最优解一致 ----------

def test_plan_gap_randomized_matches_brute_force():
    rng = random.Random(20260920)
    feasible = infeasible = 0
    for _ in range(300):
        a = 0
        b = rng.randint(6, 16)
        g = rng.randint(2, 7)
        # 随机生成端点为整数的禁排区间，单个宽度不超过 g（保证大多实例可行），
        # 但区间可能相接/嵌套（合并后内部仍可能超过 g，从而覆盖无解分支）
        blackouts = []
        for _ in range(rng.randint(0, 2)):
            x = rng.randint(0, b)
            y = x + rng.randint(0, max(1, g))
            blackouts.append((float(min(x, y)), float(max(x, y))))
        merged = normalize_blackouts(blackouts)
        expected = _brute_force_lex_min(a, b, g, merged, 1)
        try:
            result = plan_gap_points(float(a), float(b), float(g), merged)
        except ScheduleInfeasibleError:
            assert expected is None, f"greedy infeasible but brute force found {expected}"
            infeasible += 1
            continue
        assert expected is not None, "greedy returned a plan where brute force saw none"
        assert result == pytest.approx([float(t) for t in expected]), (
            f"a={a} b={b} g={g} blackouts={merged}: got {result}, expected {expected}"
        )
        chain = [float(a), *result, float(b)]
        assert all(e - s <= g + 1e-9 for s, e in zip(chain, chain[1:]))
        assert all(not any(lo < t < hi for lo, hi in merged) for t in result)
        feasible += 1
    # 绝大多数随机实例应可行，确保确实逐解对照了大量构造
    assert feasible > 200, feasible
