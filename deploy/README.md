# Deployment

The agent is served as a stateless HTTP API (`dialog2agent.server`) that wraps the same `Agent.respond` used in the
evaluation. This directory deploys that API on [Modal](https://modal.com), with a serverless GPU that scales to zero
when nobody uses the agent.

## Design

- **Stateless service.** The client sends the dialogue so far and receives the next system turn with the trace of
  the decision (state, allowed actions with their probability, retrieved situations). Any replica can answer any
  request, so replicas can be started and stopped at any time.
- **Pinned behavior.** The image installs the library versions with which the agent was evaluated (torch 2.11.0,
  transformers 5.14.1, peft 0.21.2, sentence-transformers 5.6.0, accelerate 1.15.0, huggingface_hub 1.24.0,
  numpy 2.4.3). The base model is downloaded at the commit used in the evaluation
  (`Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`) and the TRACE encoders at the commits pinned in the
  domain artifact and in `dialog2agent.representations.REVISIONS`.
- **Two volumes.** `dialog2agent-models` holds the base model, the turn encoder and the Hugging Face cache of the
  TRACE encoders (about 20 GB). `dialog2agent-domains` holds the domain artifacts (the SpokenWOZ artifact takes
  about 1 GB: structure, memory and LoRA adapter). The containers run with `HF_HUB_OFFLINE=1`, so they only read
  what the volumes contain.
- **Bounded cost.** One NVIDIA L4 (24 GB), at most one replica (`max_containers=1`), and scale to zero after
  five minutes without requests (`scaledown_window=300`). Requests are answered one at a time, since the replica
  holds a single model on the GPU. A workspace budget in Modal caps the monthly spend.

## Steps

```bash
modal volume create dialog2agent-domains
modal volume put dialog2agent-domains DOMAIN_DIR /spokenwoz
modal run deploy/modal_app.py::download
modal serve deploy/modal_app.py      # temporary URL while the command runs
modal deploy deploy/modal_app.py     # permanent URL
```

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | whether the service is up |
| GET | `/structure` | induced actions and states, with the allowed actions of each state |
| POST | `/respond` | next system turn of a dialogue, with its trace |

```bash
curl -X POST "$URL/respond" -H "Content-Type: application/json" \
  -d '{"turns": [{"speaker": "user", "text": "i need a cheap restaurant in the centre"}]}'
```

## Measurements

First deployment of the SpokenWOZ agent on one NVIDIA L4 (October 2026), with the attention of the linear layers of
Qwen 3.5 computed by the reference implementation of transformers, since the optimized kernels
(`flash-linear-attention` and `causal-conv1d`) are not installed in the image.

| | Time |
|---|---|
| Image build (first time) | 61 s |
| Download of the base model to the volume (first time) | about 3 min |
| Cold start (container start and agent loading, of which 51 s are loading) | 61.5 s |
| `/health` with the replica running | 0.7 s |
| `/respond`, decision only (11 allowed actions) | 6.0 s |
| `/respond`, decision and response | 5.3 to 9.3 s |

The three turns of the example conversation of the documentation produced the same states, actions and responses as
the agent run on the Intel Max 1550 accelerators where it was evaluated, with probabilities that differ in the third
decimal place.
