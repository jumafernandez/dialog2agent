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


class FakeBackend:
    def score(self, messages, continuations):
        return [float(len(c)) for c in continuations]

    def generate(self, messages, prefix="", max_new_tokens=120):
        return " Hello, how can I help you?"


def test_agent_turn_decides_within_allowed_actions():
    s = toy_structure()
    m = SituationMemory(np.ones((3, 16)), [{"user": "hi", "action": "greeting", "response": "hello", "state": 0,
                                            "action_id": 0}] * 3)
    agent = Agent(s, m, FakeBackend(), FakeEncoder(), FakeEncoder(), FakeEncoder(), decider=StructureDecider(s), k=2)
    session = agent.new_session()
    r = agent.turn(session, "hello")
    assert r.action.id in s.allowed(r.state)
    assert abs(sum(o.probability for o in r.options) - 1) < 1e-6
    assert session.turns[-1] == Turn("system", "Hello, how can I help you?")
    assert isinstance(session, Session) and len(r.situations) == 2
