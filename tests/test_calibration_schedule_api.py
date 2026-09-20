"""校准排期规划 API 端到端测试：POST /calibration-schedules/plan。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIB_DB_PATH", str(tmp_path / "test.db"))
    from app.main import app  # 环境变量设置后再导入

    with TestClient(app) as c:
        yield c


def make_observation(client, ts, bias=1.0):
    """在 ts 登记一条 bias 约为给定值的观测（读数中位数 = 10 - bias）。"""
    med = 10.0 - bias
    resp = client.post(
        "/observations",
        json={"timestamp": ts, "reference": 10.0, "tolerance": 0.5,
              "readings": [med, med + 0.1, med - 0.1]},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def plan(client, **body):
    return client.post("/calibration-schedules/plan", json=body)


# ---------- 基本规划 ----------

def test_plan_basic_with_external_anchors(client):
    # 两条观测覆盖窗口 [1000,4000]：0 与 5000
    make_observation(client, 0.0)
    make_observation(client, 5000.0)
    resp = plan(client, **{"from": 1000.0, "to": 4000.0, "max_gap": 1000.0})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["from"] == pytest.approx(1000.0)
    assert body["to"] == pytest.approx(4000.0)
    assert body["max_gap"] == pytest.approx(1000.0)
    assert body["blackouts"] == []

    anchors = body["fixed_anchors"]
    assert [(a["timestamp"], a["in_window"]) for a in anchors] == [
        (0.0, False), (5000.0, False),
    ]
    assert [a["observation_id"] for a in anchors] == [1, 2]
    assert anchors[0]["timestamp_iso"].startswith("1970-01-01T00:00:00")

    # 建议时点落在窗口边界与内部，每段间隔恰为 1000
    suggestions = [s["timestamp"] for s in body["suggestions"]]
    assert suggestions == [1000.0, 2000.0, 3000.0, 4000.0]
    assert body["suggestions"][0]["timestamp_iso"].startswith("1970-01-01T00:16:40")

    gaps = [(seg["t_start"], seg["t_end"], seg["gap_seconds"]) for seg in body["segments"]]
    assert gaps == [
        (0.0, 1000.0, 1000.0),
        (1000.0, 2000.0, 1000.0),
        (2000.0, 3000.0, 1000.0),
        (3000.0, 4000.0, 1000.0),
        (4000.0, 5000.0, 1000.0),
    ]
    assert all(seg["within_max_gap"] for seg in body["segments"])
    # 段数 = 锚点数 + 建议点数 - 1
    assert len(body["segments"]) == len(anchors) + len(suggestions) - 1


def test_plan_no_suggestions_when_existing_observations_satisfy_gap(client):
    make_observation(client, 1000.0)
    make_observation(client, 2000.0)
    body = plan(client, **{"from": 1000.0, "to": 2000.0, "max_gap": 1000.0}).json()
    assert body["suggestions"] == []
    assert [a["timestamp"] for a in body["fixed_anchors"]] == [1000.0, 2000.0]
    assert all(a["in_window"] for a in body["fixed_anchors"])
    assert len(body["segments"]) == 1
    assert body["segments"][0]["gap_seconds"] == pytest.approx(1000.0)
    assert body["segments"][0]["within_max_gap"] is True


def test_plan_includes_in_window_observations_as_anchors(client):
    # 窗口 [1000,4000] 内已有观测 2000，加上两侧覆盖观测
    make_observation(client, 0.0)
    make_observation(client, 2000.0)
    make_observation(client, 5000.0)
    body = plan(client, **{"from": 1000.0, "to": 4000.0, "max_gap": 1000.0}).json()
    anchors = [(a["timestamp"], a["in_window"]) for a in body["fixed_anchors"]]
    assert anchors == [(0.0, False), (2000.0, True), (5000.0, False)]
    suggestions = [s["timestamp"] for s in body["suggestions"]]
    # (0,2000) 段窗口裁剪到 [1000,...]：需点 1000；(2000,5000) 段止于 4000：需 3000、4000
    assert suggestions == [1000.0, 3000.0, 4000.0]


def test_plan_observations_exactly_on_boundaries_are_anchors(client):
    make_observation(client, 1000.0)
    make_observation(client, 4000.0)
    body = plan(client, **{"from": 1000.0, "to": 4000.0, "max_gap": 1500.0}).json()
    anchors = [(a["timestamp"], a["in_window"]) for a in body["fixed_anchors"]]
    assert anchors == [(1000.0, True), (4000.0, True)]
    # 间隔 3000、max_gap 1500：恰需一点，且其位置被两侧约束唯一确定为 2500
    assert [s["timestamp"] for s in body["suggestions"]] == [2500.0]


def test_plan_lexicographically_earliest_when_ties(client):
    # 间隔 2500、max_gap 1000：两组最少方案如 [500,1500] 与 [1000,2000]，取最早
    make_observation(client, 0.0)
    make_observation(client, 2500.0)
    body = plan(client, **{"from": 0.0, "to": 2500.0, "max_gap": 1000.0}).json()
    assert [s["timestamp"] for s in body["suggestions"]] == [500.0, 1500.0]
    gaps = [seg["gap_seconds"] for seg in body["segments"]]
    assert gaps == [500.0, 1000.0, 1000.0]
    assert all(seg["within_max_gap"] for seg in body["segments"])


# ---------- 禁排时段 ----------

def test_plan_normalizes_and_merges_blackouts(client):
    make_observation(client, 0.0)
    make_observation(client, 5000.0)
    resp = plan(client, **{
        "from": 0.0, "to": 5000.0, "max_gap": 1000.0,
        "blackouts": [
            {"start": 2400.0, "end": 2800.0},   # 与下一时段交叠，合并为 [2000,2800]
            {"start": 2000.0, "end": 2500.0},
            {"start": 4500.0, "end": 4500.0},   # 零长时段直接丢弃，不影响规划
        ],
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [(b["t_start"], b["t_end"]) for b in body["blackouts"]] == [(2000.0, 2800.0)]
    suggestions = [s["timestamp"] for s in body["suggestions"]]
    # 合并后禁排端点 2000/2800 可排；方案不在禁排内部
    assert not any(2000.0 < t < 2800.0 for t in suggestions)
    chain = [0.0, *suggestions, 5000.0]
    assert all(b - a <= 1000.0 for a, b in zip(chain, chain[1:]))


def test_plan_touching_blackouts_keep_shared_boundary_schedulable(client):
    # 端点相接的两个禁排不合并：公共端点可排
    make_observation(client, 0.0)
    make_observation(client, 3000.0)
    body = plan(client, **{
        "from": 0.0, "to": 3000.0, "max_gap": 1000.0,
        "blackouts": [
            {"start": 1000.0, "end": 2000.0},
            {"start": 2000.0, "end": 2800.0},
        ],
    }).json()
    assert [(b["t_start"], b["t_end"]) for b in body["blackouts"]] == [
        (1000.0, 2000.0), (2000.0, 2800.0),
    ]
    suggestions = [s["timestamp"] for s in body["suggestions"]]
    assert 2000.0 in suggestions  # 相接端点 2000 可排


def test_plan_blackout_boundaries_are_schedulable(client):
    # 禁排 (1000,2000) 端点可排：方案直接复用两个端点
    make_observation(client, 0.0)
    make_observation(client, 3000.0)
    body = plan(client, **{
        "from": 0.0, "to": 3000.0, "max_gap": 1000.0,
        "blackouts": [{"start": 1000.0, "end": 2000.0}],
    }).json()
    assert [s["timestamp"] for s in body["suggestions"]] == [1000.0, 2000.0]
    # 建议时点恰为禁排端点，但不在其内部
    assert body["blackouts"] == [
        {"t_start": 1000.0, "t_end": 2000.0,
         "start_iso": body["blackouts"][0]["start_iso"],
         "end_iso": body["blackouts"][0]["end_iso"]},
    ]


def test_plan_blackouts_clipped_to_window(client):
    make_observation(client, 0.0)
    make_observation(client, 5000.0)
    body = plan(client, **{
        "from": 1000.0, "to": 4000.0, "max_gap": 2000.0,
        "blackouts": [{"start": -1000.0, "end": 2000.0}],  # 伸出窗口左侧，裁剪为 [1000,2000]
    }).json()
    assert [(b["t_start"], b["t_end"]) for b in body["blackouts"]] == [(1000.0, 2000.0)]


def test_plan_blackout_makes_schedule_infeasible_409(client):
    # 禁排内部长度超过 max_gap：无解
    make_observation(client, 0.0)
    make_observation(client, 5000.0)
    resp = plan(client, **{
        "from": 0.0, "to": 5000.0, "max_gap": 1000.0,
        "blackouts": [{"start": 500.0, "end": 2000.0}],  # 内部长 1500 > 1000
    })
    assert resp.status_code == 409
    assert "max_gap" in resp.json()["detail"]


def test_plan_blackout_crossing_window_left_boundary_409(client):
    # 禁排从窗口左侧外延伸、越过 from：from 点严格在禁排内部、不可排，
    # 左外部锚点 0 无法在 max_gap 内到达首个可排点 → 409（裁剪不得伪造可排端点）
    make_observation(client, 0.0)
    make_observation(client, 5000.0)
    resp = plan(client, **{
        "from": 1000.0, "to": 4000.0, "max_gap": 1000.0,
        "blackouts": [{"start": 900.0, "end": 1100.0}],
    })
    assert resp.status_code == 409


def test_plan_blackout_crossing_window_right_boundary_409(client):
    # 禁排越过 to：to 点严格在禁排内部、不可排
    make_observation(client, 0.0)
    make_observation(client, 5000.0)
    resp = plan(client, **{
        "from": 1000.0, "to": 4000.0, "max_gap": 1000.0,
        "blackouts": [{"start": 3900.0, "end": 4100.0}],
    })
    assert resp.status_code == 409


def test_plan_blackout_ending_exactly_at_from_keeps_boundary_schedulable(client):
    # 禁排恰止于 from：from 是端点、可排，规划正常
    make_observation(client, 0.0)
    make_observation(client, 5000.0)
    resp = plan(client, **{
        "from": 1000.0, "to": 4000.0, "max_gap": 1000.0,
        "blackouts": [{"start": 900.0, "end": 1000.0}],
    })
    assert resp.status_code == 200, resp.text
    assert [s["timestamp"] for s in resp.json()["suggestions"]] == [
        1000.0, 2000.0, 3000.0, 4000.0,
    ]


# ---------- 边界锚点缺失 ----------

def test_plan_missing_left_boundary_anchor_409(client):
    make_observation(client, 2000.0)
    make_observation(client, 3000.0)
    resp = plan(client, **{"from": 1000.0, "to": 3000.0, "max_gap": 1000.0})
    assert resp.status_code == 409
    assert "cover both window boundaries" in resp.json()["detail"]


def test_plan_missing_right_boundary_anchor_409(client):
    make_observation(client, 0.0)
    make_observation(client, 2000.0)
    resp = plan(client, **{"from": 1000.0, "to": 4000.0, "max_gap": 1000.0})
    assert resp.status_code == 409
    assert "cover both window boundaries" in resp.json()["detail"]


def test_plan_without_any_observation_409(client):
    resp = plan(client, **{"from": 1000.0, "to": 4000.0, "max_gap": 1000.0})
    assert resp.status_code == 409


def test_plan_single_observation_cannot_cover_both_sides_409(client):
    make_observation(client, 2000.0)
    resp = plan(client, **{"from": 1000.0, "to": 4000.0, "max_gap": 1000.0})
    assert resp.status_code == 409


# ---------- 参数校验 ----------

def test_plan_rejects_inverted_or_empty_window(client):
    make_observation(client, 0.0)
    make_observation(client, 5000.0)
    for frm, to in ((2000.0, 1000.0), (1000.0, 1000.0)):
        resp = plan(client, **{"from": frm, "to": to, "max_gap": 1000.0})
        assert resp.status_code == 400
        assert "must be less than" in resp.json()["detail"]


def test_plan_rejects_non_positive_max_gap(client):
    make_observation(client, 0.0)
    make_observation(client, 5000.0)
    for gap in (0.0, -100.0):
        resp = plan(client, **{"from": 0.0, "to": 5000.0, "max_gap": gap})
        assert resp.status_code == 400
        assert "must be positive" in resp.json()["detail"]


def test_plan_rejects_inverted_blackout(client):
    make_observation(client, 0.0)
    make_observation(client, 5000.0)
    resp = plan(client, **{
        "from": 0.0, "to": 5000.0, "max_gap": 1000.0,
        "blackouts": [{"start": 3000.0, "end": 1000.0}],
    })
    assert resp.status_code == 400
    assert "start <= end" in resp.json()["detail"]


def test_plan_missing_required_fields_422(client):
    resp = client.post("/calibration-schedules/plan", json={"from": 0.0, "to": 1000.0})
    assert resp.status_code == 422


def test_plan_accepts_iso8601_inputs(client):
    make_observation(client, 0.0)
    make_observation(client, 5000.0)
    resp = plan(client, **{
        "from": "1970-01-01T00:16:40Z",   # 1000
        "to": "1970-01-01T01:06:40Z",     # 4000
        "max_gap": 1000.0,
        "blackouts": [
            {"start": "1970-01-01T00:33:20Z", "end": "1970-01-01T00:50:00Z"},  # 2000..3000
        ],
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["from"] == pytest.approx(1000.0)
    assert body["to"] == pytest.approx(4000.0)
    assert [(b["t_start"], b["t_end"]) for b in body["blackouts"]] == [(2000.0, 3000.0)]


def test_plan_blackouts_default_to_empty(client):
    make_observation(client, 1000.0)
    make_observation(client, 2000.0)
    resp = plan(client, **{"from": 1000.0, "to": 2000.0, "max_gap": 1000.0})
    assert resp.status_code == 200
    assert resp.json()["blackouts"] == []


# ---------- 只读性 ----------

def test_plan_is_read_only(client):
    make_observation(client, 0.0)
    make_observation(client, 5000.0)
    before = (
        client.get("/observations").json(),
        client.get("/intervals").json(),
    )
    # 成功规划与无解（409）规划都不得改动任何现状数据
    assert plan(client, **{"from": 1000.0, "to": 4000.0, "max_gap": 1000.0}).status_code == 200
    assert plan(client, **{
        "from": 0.0, "to": 5000.0, "max_gap": 1000.0,
        "blackouts": [{"start": 500.0, "end": 2000.0}],
    }).status_code == 409
    after = (
        client.get("/observations").json(),
        client.get("/intervals").json(),
    )
    assert after == before
