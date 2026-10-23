import numpy as np

from dialog2agent import Agent, InducedStructure, Session, Situation, SituationMemory, StructureDecider, Turn


def toy_structure():
    rng = np.random.default_rng(0)
    system = rng.normal(size=(4, 8))
    user = rng.normal(size=(3, 8))
    states = rng.normal(size=(2, 16))
    transitions = np.array([[10, 5, 0, 0], [0, 0, 7, 3]], float)
    s = InducedStructure(system / np.linalg.norm(system, axis=1, keepdims=True),
                         user / np.linalg.norm(user, axis=1, keepdims=True), states, transitions,
                         names=["greeting", "request: name", "inform: name", "inform: name"], tau=0.1)
    return s


def test_allowed_actions_are_sorted_and_thresholded():
    s = toy_structure()
    assert s.allowed(0) == [0, 1]
    assert s.allowed(1) == [2, 3]
    assert np.isclose(s.prior(0).sum(), 1.0)


def test_structure_roundtrip(tmp_path):
    s = toy_structure()
    s.save(tmp_path)
    t = InducedStructure.load(tmp_path)
    assert t.names == s.names and np.allclose(t.transitions, s.transitions)


def test_memory_search_conditioned_and_fallback(tmp_path):
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(6, 4))
    entries = [{"user": f"u{i}", "action": "greeting", "response": f"r{i}", "state": i % 2, "action_id": 0,
                "dialogue_id": f"d{i}"} for i in range(6)]
    m = SituationMemory(vectors, entries)
    q = vectors[3]
    assert m.search(q, k=1)[0].user == "u3"
    assert all(s.user in {"u1", "u3", "u5"} for s in m.search(q, k=3, state=1, allowed=[]))
    assert len(m.search(q, k=5, state=1, allowed=[])) == 5  # fewer than k in the state: whole memory
    m.add(rng.normal(size=4), Situation("u", "greeting", "r", 1.0, provenance="agent"), state=0, action_id=0)
    m.save(tmp_path)
    assert len(SituationMemory.load(tmp_path)) == 7


class FakeEncoder:
    def encode(self, *args):
        n = len(args[0])
        return np.ones((n, 8), np.float32)


class FakeDialogueEncoder:
    def encode(self, turns, e):
        return np.ones(8, np.float32)


class FakeBackend:
    def score(self, messages, continuations):
        return [float(len(c)) for c in continuations]

    def generate(self, messages, prefix="", max_new_tokens=120, sample=False):
        return " Hello, how can I help you?"


def test_agent_turn_decides_within_allowed_actions():
    s = toy_structure()
    m = SituationMemory(np.ones((3, 16)), [{"user": "hi", "action": "greeting", "response": "hello", "state": 0,
                                            "action_id": 0}] * 3)
    agent = Agent(s, m, FakeBackend(), FakeEncoder(), FakeEncoder(), FakeDialogueEncoder(), decider=StructureDecider(s), k=2)
    session = agent.new_session()
    r = agent.turn(session, "hello")
    assert r.action.id in s.allowed(r.state)
    assert abs(sum(o.probability for o in r.options) - 1) < 1e-6
    assert session.turns[-1] == Turn("system", "Hello, how can I help you?")
    assert isinstance(session, Session) and len(r.situations) == 2


def test_state_is_the_nearest_center():
    s = toy_structure()
    rng = np.random.default_rng(2)
    e, h = rng.normal(size=(20, 8)), rng.normal(size=(20, 8))
    x = np.hstack([e / np.linalg.norm(e, axis=1, keepdims=True),
                   s.context_weight * h / np.linalg.norm(h, axis=1, keepdims=True)])
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    brute = np.argmin(((x[:, None] - s.state_centers) ** 2).sum(-1), axis=1)
    assert (s.state(e, h) == brute).all()


def test_scoring_decider_breaks_name_ties_by_state_probability():
    from dialog2agent import LLMScoringDecider
    s = toy_structure()

    class ByName:
        def score(self, messages, continuations):
            return [1.0 if "inform" in c else 0.0 for c in continuations]

    options = LLMScoringDecider(ByName(), s).decide([], 1, s.allowed(1), [])
    assert [o.action.id for o in options] == [2, 3]  # same name, action 2 is more probable in state 1
    assert np.isclose(options[0].probability, options[1].probability)


def test_memory_search_excludes_a_dialogue():
    rng = np.random.default_rng(3)
    entries = [{"user": f"u{i}", "action": "greeting", "response": "r", "state": 0, "action_id": 0,
                "dialogue_id": f"d{i % 2}"} for i in range(6)]
    m = SituationMemory(rng.normal(size=(6, 4)), entries)
    assert all(x.dialogue_id == "d1" for x in m.search(rng.normal(size=4), k=3, exclude_dialogue="d0"))


def test_evaluation_predicts_with_the_real_history(tmp_path):
    from dialog2agent.evaluation import predict, report
    s = toy_structure()
    m = SituationMemory(np.ones((3, 16)), [{"user": "hi", "action": "greeting", "response": "hello", "state": 0,
                                            "action_id": 0}] * 3)
    agent = Agent(s, m, FakeBackend(), FakeEncoder(), FakeEncoder(), FakeDialogueEncoder(), decider=StructureDecider(s), k=2)
    turns = [{"speaker": "user", "text": "hi"}, {"speaker": "system", "text": "hello"},
             {"speaker": "user", "text": "a table"}, {"speaker": "user", "text": "for two"},
             {"speaker": "system", "text": "sure"}]
    out = tmp_path / "predictions_0.jsonl"
    assert predict(agent, [{"dialogue_id": "d", "turns": turns}], out) == 2
    assert predict(agent, [{"dialogue_id": "d", "turns": turns}], out) == 0  # resumed: nothing left
    r = report([out])
    assert r["turns"] == 2 and 0 <= r["action_accuracy"] <= 1


def test_server_responds_with_the_trace():
    import pytest
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from dialog2agent.server import create_app
    s = toy_structure()
    m = SituationMemory(np.ones((3, 16)), [{"user": "hi", "action": "greeting", "response": "hello", "state": 0,
                                            "action_id": 0}] * 3)
    agent = Agent(s, m, FakeBackend(), FakeEncoder(), FakeEncoder(), FakeDialogueEncoder(),
                  decider=StructureDecider(s), k=2)
    client = TestClient(create_app(agent))
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/about").json()["library"] == "dialog2agent"
    assert len(client.get("/structure").json()["states"]) == 2
    r = client.post("/respond", json={"turns": [{"speaker": "user", "text": "hello"}]}).json()
    assert r["text"] == "Hello, how can I help you?" and r["action"]["id"] in s.allowed(r["state"])
    assert len(r["situations"]) == 2
    bad = client.post("/respond", json={"turns": [{"speaker": "system", "text": "hi"}]})
    assert bad.status_code == 422
    g = client.post("/respond", json={"turns": [{"speaker": "user", "text": "hello"}], "decision": "generate"}).json()
    assert g["decision"] == "generate" and len(g["options"]) == 1 and len(g["allowed"]) >= 1
    assert client.post("/respond", json={"turns": [{"speaker": "user", "text": "hello"}]}).json()["decision"] == "score"
    chat = client.post("/v1/chat/completions", json={"model": "x", "messages": [
        {"role": "system", "content": "ignored"}, {"role": "user", "content": "hello"}]}).json()
    assert chat["choices"][0]["message"]["content"] == "Hello, how can I help you?"
    assert chat["trace"]["state"] == r["state"]


def test_generation_decision_projects_the_declared_action():
    s = toy_structure()
    m = SituationMemory(np.ones((3, 16)), [{"user": "hi", "action": "greeting", "response": "hello", "state": 0,
                                            "action_id": 0}] * 3)

    class Writer(FakeBackend):
        def generate(self, messages, prefix="", max_new_tokens=120, sample=False):
            return "Action: greeting\nResponse: Hello there."

    agent = Agent(s, m, Writer(), FakeEncoder(), FakeEncoder(), FakeDialogueEncoder(), k=2, decision="generate")
    r = agent.respond([Turn("user", "hi")])
    assert r.text == "Hello there." and len(r.options) == 1 and r.options[0].probability == 1.0


def test_scope_flags_turns_below_either_threshold():
    import pytest

    from dialog2agent import Scope
    s = toy_structure()
    rng = np.random.default_rng(4)
    m = SituationMemory(rng.normal(size=(5, 16)), [{"user": "hi", "action": "greeting", "response": "hello"}] * 5)

    class ByText:  # the embedding of a turn is the user centroid named by its text, or a random vector
        def encode(self, texts):
            return np.stack([s.user_centroids[int(t)] if t.isdigit() else rng.normal(size=8) for t in texts])

    agent = Agent(s, m, FakeBackend(), ByText(), FakeEncoder(), FakeDialogueEncoder(), k=2)
    with pytest.raises(ValueError):
        agent.check_scope("0")
    agent.scope = Scope.calibrate(agent, ["0", "1", "2"], percentile=0, language="en")
    assert agent.scope.calibration.endswith("3 user turns") and agent.scope.language == "en"
    c = agent.check_scope("1")
    assert c.in_scope and np.isclose(c.action_similarity, 1.0)
    agent.scope.memory_threshold = 2.0  # no situation is that similar
    assert not agent.check_scope("1").in_scope


def test_server_checks_the_scope():
    import pytest
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from dialog2agent import Scope
    from dialog2agent.server import create_app
    s = toy_structure()
    m = SituationMemory(np.ones((3, 16)), [{"user": "hi", "action": "greeting", "response": "hello"}] * 3)
    agent = Agent(s, m, FakeBackend(), FakeEncoder(), FakeEncoder(), FakeDialogueEncoder(), k=2)
    assert TestClient(create_app(agent)).post("/scope", json={"text": "hello"}).status_code == 404
    agent.scope = Scope(action_threshold=-1.0, memory_threshold=-1.0, language="en")
    client = TestClient(create_app(agent))
    assert client.get("/about").json()["scope"]["language"] == "en"
    r = client.post("/scope", json={"text": "hello"}).json()
    assert r["in_scope"] and set(r) == {"in_scope", "action_similarity", "memory_similarity"}


def test_artifacts_that_are_not_local_are_downloaded_from_the_hub(tmp_path, monkeypatch):
    import huggingface_hub
    import pytest

    from dialog2agent.domain import load_agent
    calls = []

    def snapshot_download(repo_id, revision=None):
        calls.append((repo_id, revision))
        return str(tmp_path)  # an empty directory: loading stops at domain.json

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    with pytest.raises(FileNotFoundError):
        load_agent("user/some-domain", revision="abc123")
    assert calls == [("user/some-domain", "abc123")]
