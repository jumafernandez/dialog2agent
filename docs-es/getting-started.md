# Primeros pasos

## Instalación

La librería requiere Python 3.10 o posterior y se instala desde su repositorio:

```bash
pip install "dialog2agent @ git+https://github.com/jumafernandez/dialog2agent"
```

Las dependencias opcionales se instalan con extras: `web` para servir el agente, `eval` para la evaluación, `build`
para construir un dominio y `docs` para esta documentación.

```bash
pip install "dialog2agent[web] @ git+https://github.com/jumafernandez/dialog2agent"
```

## Artefactos de dominio

Un agente se carga a partir de un artefacto de dominio, un directorio con todo lo que el agente aprendió de los
diálogos de un dominio, es decir, la estructura inducida, la memoria de situaciones, el adaptador del modelo de
lenguaje y la configuración de `domain.json` (ver [Construir un dominio](build-a-domain.md)). El modelo de lenguaje y
los codificadores se descargan del Hub de Hugging Face la primera vez que se utilizan, en las revisiones fijadas por el
artefacto y por la librería.

El artefacto de SpokenWOZ está publicado en el Hub como
[`jumafernandez/dialog2agent-spokenwoz`](https://huggingface.co/jumafernandez/dialog2agent-spokenwoz), con la licencia
CC BY-NC 4.0 porque contiene datos derivados de SpokenWOZ, y tanto `Agent.from_pretrained` como la línea de comandos
aceptan un directorio local o un repositorio del Hub. El agente también puede probarse sin instalar nada en la
[demostración web](https://jumafernandez.github.io/dialog2agent/demo.html).

## Conversar con un agente

```bash
dialog2agent chat jumafernandez/dialog2agent-spokenwoz
```

`--device` elige el acelerador (por ejemplo `cuda`, `mps`, `xpu` o `cpu`), `--decision generate` hace que el modelo
genere la acción en lugar de puntuar las acciones permitidas y `--no-adapter` carga el modelo de base sin el
adaptador del artefacto.

## Uso desde Python

```python
from dialog2agent import Agent

agent = Agent.from_pretrained("jumafernandez/dialog2agent-spokenwoz")
session = agent.new_session()
for text in ["hello, I need a restaurant in the centre", "something cheap, please"]:
    result = agent.turn(session, text)
    print(f"[estado {result.state} · {result.action.name}] {result.text}")
```

`Agent.turn` agrega el turno del usuario y la respuesta a la sesión. `Agent.respond` calcula el turno siguiente del
sistema para cualquier diálogo que termine con un turno del usuario, sin modificar ningún estado, y es el método que
invoca la evaluación, de modo que el agente que se utiliza es el mismo que se evaluó.

Cada resultado contiene la traza de la decisión:

| Campo | Contenido |
|---|---|
| `text` | la respuesta |
| `action` | la acción decidida, con su `id` y su `name` |
| `state` | el estado inducido de la conversación |
| `options` | las acciones permitidas con su probabilidad, de la más a la menos probable |
| `situations` | las situaciones recuperadas de la memoria, con su similitud |
| `tool_calls` | las llamadas a herramientas que disparó la acción |

## Hardware

El agente de SpokenWOZ ejecuta el modelo de 9 mil millones de parámetros de la familia Qwen 3.5 en bfloat16, por lo
cual requiere un acelerador con alrededor de 24 GB de memoria, como una NVIDIA L4. Los codificadores son pequeños y
también funcionan en CPU.
