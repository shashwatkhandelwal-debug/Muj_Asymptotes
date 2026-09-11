"""
Regression test for Red-Team API contract and scenario parsing.
Ensures GET /api/redteam/scenarios returns a properly structured payload
with exactly 6 distinct, correctly-keyed scenario definitions.
"""
import pytest
from fastapi.testclient import TestClient
from api.orchestrator import app

client = TestClient(app)

EXPECTED_SCENARIO_IDS = [
    "deepfake_face",
    "voice_spoof",
    "genai_doc",
    "tampered_doc",
    "invalid_qr",
    "clean_control",
]

def test_redteam_scenarios_response_shape():
    """
    Regression Test:
    Prevents bug where frontend did Object.entries() on top-level object
    when endpoint returned {'ok': True, 'scenarios': [...]}, producing 2 bad entries ['ok', 'scenarios'].
    """
    response = client.get("/api/redteam/scenarios")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert isinstance(data, dict), f"Expected dict root, got {type(data)}"
    assert "ok" in data and data["ok"] is True, "Expected 'ok': True in response"
    assert "scenarios" in data, "Expected 'scenarios' key in response"
    
    scenarios_raw = data["scenarios"]
    assert isinstance(scenarios_raw, list), f"Expected list of scenarios, got {type(scenarios_raw)}"
    assert len(scenarios_raw) == 6, f"Expected exactly 6 scenarios, got {len(scenarios_raw)}"
    
    # Simulate the frontend normalization logic:
    # const list = Array.isArray(data) ? data : (data.scenarios || Object.values(data));
    normalized_list = scenarios_raw if isinstance(scenarios_raw, list) else (data.get("scenarios") or list(data.values()))
    
    scenario_map = {}
    for item in normalized_list:
        assert isinstance(item, dict), f"Scenario item must be dict, got {item}"
        s_id = item.get("id")
        assert s_id, f"Scenario item missing 'id': {item}"
        assert "title" in item or "name" in item, f"Scenario item missing 'title' or 'name': {item}"
        assert "category" in item, f"Scenario item missing 'category': {item}"
        scenario_map[s_id] = item
        
    assert len(scenario_map) == 6, f"Expected 6 unique scenario IDs, got {len(scenario_map)}"
    for exp_id in EXPECTED_SCENARIO_IDS:
        assert exp_id in scenario_map, f"Missing expected scenario ID '{exp_id}' in {list(scenario_map.keys())}"


def test_redteam_run_scenarios():
    """Test running all 6 scenarios via /api/redteam/run returns valid signal lists and decision."""
    for s_id in EXPECTED_SCENARIO_IDS:
        res = client.post(f"/api/redteam/run?scenario={s_id}")
        assert res.status_code == 200, f"Scenario {s_id} returned HTTP {res.status_code}: {res.text}"
        payload = res.json()
        assert payload.get("ok") is True, f"Scenario {s_id} ok != True: {payload}"
        assert "decision" in payload, f"Scenario {s_id} missing decision: {payload}"
        assert "signals" in payload["decision"] or "lines" in payload, f"Scenario {s_id} missing signals: {payload}"
        assert "audit" in payload, f"Scenario {s_id} missing audit: {payload}"
