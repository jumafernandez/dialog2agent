# Serving the agent

`dialog2agent serve` exposes an agent through an HTTP API that wraps the same `Agent.respond` used in the evaluation.
The service keeps no state between requests: the client sends the dialogue so far, which ends with a user turn, and
receives the next system turn with the full trace of the decision. Therefore, any replica of the service can answer
any request, and replicas can be started and stopped at any time.

```bash
pip install "dialog2agent[web] @ git+https://github.com/jumafernandez/dialog2agent"
dialog2agent serve jumafernandez/dialog2agent-spokenwoz --port 8000
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | whether the agent is loaded |
| GET | `/about` | library version, decision modes, scope and attribution of the data of the domain artifact |
| GET | `/structure` | induced actions and states, with the allowed actions of each state |
| POST | `/respond` | next system turn of a dialogue, with its trace; `decision` (`score` or `generate`) selects how the action is decided for that request |
| POST | `/scope` | whether a user turn is within the scope of the domain, when the artifact declares a scope |
| GET | `/v1/models` | model of the OpenAI-compatible API |
| POST | `/v1/chat/completions` | the agent through the chat completions API of OpenAI |

```bash
curl -X POST http://localhost:8000/respond -H "Content-Type: application/json" \
  -d '{"turns": [{"speaker": "user", "text": "i need a cheap restaurant in the centre"}]}'
```

The response contains the text, the state, the allowed actions, the decided action, the options with their
probability, the retrieved situations and the tool calls.

## OpenAI-compatible API

Any client of the chat completions API of OpenAI can converse with the agent. System messages are ignored, user and
assistant messages are the turns of the dialogue, and the trace of the decision is returned in the additional `trace`
field of the response.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
reply = client.chat.completions.create(
    model="dialog2agent",
    messages=[{"role": "user", "content": "i need a cheap restaurant in the centre"}])
print(reply.choices[0].message.content)
```

This is how the user simulators of SDialog talk to the agent (see [Evaluation](evaluation.md#closed-loop-simulation-with-sdialog)).

## Options

| Option | Effect |
|---|---|
| `--decision`, `--no-adapter` | decision mode and adapter of the agent, as in `chat` |
| `--allowed-origins` | origins allowed to call the API from a browser (any origin by default) |
| `--log FILE` | appends every response, with the dialogue and its trace, to a JSON lines file |
| `--model-id` | model name in the OpenAI-compatible API |

Requests are answered one at a time, since the agent holds a single model on the accelerator.

## Deployment

The `deploy/` directory of the repository deploys the service on [Modal](https://modal.com) with a serverless GPU that
scales to zero when nobody uses the agent. It is the service behind the
[web demo](https://jumafernandez.github.io/dialog2agent/demo.html); its design, steps and measurements are described in
[`deploy/README.md`](https://github.com/jumafernandez/dialog2agent/blob/main/deploy/README.md).
