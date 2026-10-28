def test_health_and_model_info(client):
    assert client.get("/health").get_json()["model_loaded"] is True

    info = client.get("/api/model").get_json()
    assert info["loaded"] is True
    assert len(info["checkpoint_sha256"]) == 64
    assert info["version"] == info["checkpoint_sha256"][:12]
    assert info["temperature"] == 1.5
    assert info["classes"] == ["hydrolase", "oxidoreductase"]


def test_predict_single(client, lysozyme):
    response = client.post("/api/predict", json={"sequence": f">lyz\n{lysozyme}"})
    assert response.status_code == 200
    body = response.get_json()

    assert body["input"]["id"] == "lyz"
    probabilities = body["prediction"]["probabilities"]
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6
    assert body["prediction"]["calibrated"] is True
    assert body["model"]["version"]
    assert body["model"]["trained_on"]["hydrolase_count"] == 10

    attribution = body["attribution"]
    assert attribution["length"] == 64
    assert len(attribution["scores"]) == 64
    assert max(abs(score) for score in attribution["scores"]) <= 1.0
    assert {w["code"] for w in body["warnings"]} == {"truncated"}


def test_predict_rejects_bad_bodies(client):
    assert client.post("/api/predict", json=[1, 2]).status_code == 400
    assert client.post("/api/predict", json={"sequence": "   "}).status_code == 400
    assert client.post("/api/predict", json={"sequence": 5}).status_code == 400

    no_standard = client.post("/api/predict", json={"sequence": "12345"})
    assert no_standard.status_code == 400
    assert no_standard.get_json()["invalid_residues"]


def test_predict_rejects_multiple_records(client):
    response = client.post("/api/predict", json={"sequence": ">a\nMKV\n>b\nGHK"})
    assert response.status_code == 400
    assert "batch" in response.get_json()["error"]


def test_batch_mixed_records(client, lysozyme):
    text = f">good\n{lysozyme}\n>short\nMKV\n>bad\n12345"
    response = client.post("/api/predict/batch", json={"sequence": text})
    assert response.status_code == 200

    body = response.get_json()
    assert body["count"] == 3
    good, short, bad = body["results"]
    assert good["id"] == "good" and good["class_name"] in {"hydrolase", "oxidoreductase", "uncertain"}
    assert "attribution" not in good
    assert [w["code"] for w in short["warnings"]] == ["short_sequence"]
    assert "error" in bad and "class_name" not in bad


def test_batch_from_records_list(client):
    response = client.post(
        "/api/predict/batch",
        json={"records": [{"id": "x", "sequence": "MKVLAAGHKT"}, {"sequence": ">y\nGGGGAAAA"}]},
    )
    assert [r["id"] for r in response.get_json()["results"]] == ["x", "y"]


def test_batch_limit(client):
    records = [{"id": str(i), "sequence": "MKV"} for i in range(501)]
    assert client.post("/api/predict/batch", json={"records": records}).status_code == 400


def test_unknown_api_route_is_json_404(client):
    response = client.get("/api/missing")
    assert response.status_code == 404
    assert response.get_json()["error"]


def test_predict_without_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv("SEQUENCEVIEW_CHECKPOINT_PATH", str(tmp_path / "absent.pt"))
    from sequenceview.app import create_app

    client = create_app().test_client()
    assert client.post("/api/predict", json={"sequence": "MKVLAAGHKT"}).status_code == 503
    assert client.get("/api/model").get_json()["loaded"] is False


def test_predict_includes_motif_hits(client):
    sequence = "A" * 30 + "LTAAHC" + "A" * 30
    body = client.post("/api/predict", json={"sequence": sequence}).get_json()
    assert "PS00134" in {hit["id"] for hit in body["motifs"]}


def test_batch_projection_needs_three_scored_records(client, lysozyme):
    two = client.post("/api/predict/batch", json={"sequence": f">a\n{lysozyme}\n>b\n{lysozyme[::-1]}"}).get_json()
    assert two["projection"] is None

    text = "\n".join(f">r{i}\n{lysozyme[i:]}" for i in range(4)) + "\n>bad\n123"
    body = client.post("/api/predict/batch", json={"sequence": text}).get_json()
    projection = body["projection"]
    assert [point["index"] for point in projection["points"]] == [1, 2, 3, 4]
    assert len(projection["explained_variance"]) == 2


def test_mutation_scan(client, lysozyme):
    response = client.post("/api/mutscan", json={"sequence": lysozyme, "start": 3, "end": 8})
    assert response.status_code == 200
    body = response.get_json()

    assert body["alphabet"] == "ACDEFGHIKLMNPQRSTVWY"
    assert [p["index"] for p in body["positions"]] == [3, 4, 5, 6, 7]
    for position in body["positions"]:
        assert len(position["delta_log_odds"]) == 20
        wild_type = body["alphabet"].index(position["residue"]) if position["residue"] in body["alphabet"] else None
        if wild_type is not None:
            assert position["delta_log_odds"][wild_type] == 0
            assert position["delta_probability"][wild_type] == 0
    assert 0 <= body["baseline_oxidoreductase"] <= 1


def test_mutation_scan_validation(client, lysozyme):
    assert client.post("/api/mutscan", json={"sequence": lysozyme, "start": 5, "end": 5}).status_code == 400
    assert client.post("/api/mutscan", json={"sequence": lysozyme, "start": "a"}).status_code == 400
    assert client.post("/api/mutscan", json={"sequence": "A" * 300 + "C", "start": 0, "end": 300}).status_code == 400
    assert client.post("/api/mutscan", json={"sequence": ">a\nMKV\n>b\nGHK"}).status_code == 400
    assert client.post("/api/mutscan", json=[1]).status_code == 400


def test_compare(client, lysozyme):
    mutated = lysozyme[:60] + "AAA" + lysozyme[63:]
    response = client.post("/api/compare", json={"a": f">wild\n{lysozyme}", "b": f">mutant\n{mutated}"})
    assert response.status_code == 200
    body = response.get_json()

    assert body["a"]["id"] == "wild" and body["b"]["id"] == "mutant"
    assert body["alignment"]["identity"] < 1.0
    assert len(body["alignment"]["aligned_a"]) == len(body["alignment"]["aligned_b"])
    expected = body["b"]["prediction"]["probabilities"]["oxidoreductase"] - body["a"]["prediction"]["probabilities"]["oxidoreductase"]
    assert abs(body["delta_oxidoreductase_probability"] - expected) < 1e-3


def test_compare_validation(client, lysozyme):
    assert client.post("/api/compare", json={"a": lysozyme}).status_code == 400
    assert client.post("/api/compare", json={"a": ">x\nMK\n>y\nMK", "b": "MK"}).status_code == 400
    assert client.post("/api/compare", json={"a": "A" * 2001, "b": "MK"}).status_code == 400
