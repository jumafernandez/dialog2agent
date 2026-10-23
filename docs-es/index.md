# dialog2agent

**Agentes de diálogo orientados a tareas que deciden sobre estructuras inducidas de tus propias conversaciones.**

[Probar la demostración web](https://jumafernandez.github.io/dialog2agent/demo.html){ .md-button .md-button--primary }
[Primeros pasos](getting-started.md){ .md-button }

`dialog2agent` convierte una colección de diálogos sin anotar en un agente. A partir de los diálogos induce una
estructura de estados y acciones, construye una memoria de situaciones conversacionales y adapta un modelo de
lenguaje. En cada turno, el agente ubica la conversación en un estado, recupera situaciones parecidas de otros
diálogos, decide la acción siguiente puntuando las acciones que la estructura admite en ese estado y redacta la
respuesta.

```python
from dialog2agent import Agent

agent = Agent.from_pretrained("jumafernandez/dialog2agent-spokenwoz")
session = agent.new_session()
result = agent.turn(session, "I'm looking for a cheap hotel in the north")
print(result.state, result.action.name, result.options[0].probability)
print(result.text)
```

Cada turno devuelve la traza completa de la decisión, es decir, el estado, las acciones permitidas con su
probabilidad, las situaciones recuperadas y las llamadas a herramientas. De este modo, cada respuesta puede
atribuirse a un estado y a una acción de la estructura.

## Componentes

| Componente | Función |
|---|---|
| Representaciones de turno ([Dialog2Flow](https://huggingface.co/sergioburdisso/dialog2flow-joint-bert-base)) | caracterizan la acción que realiza cada turno |
| Representaciones contextuales ([TRACE](models.md)) | ubican la conversación en un estado |
| Representaciones del diálogo ([TRACE-Dialog](models.md)) | recuperan situaciones parecidas de otros diálogos |
| Estructura inducida | define las acciones, los estados y las acciones permitidas en cada estado |
| Memoria conversacional | situaciones de otros diálogos, con lo que hizo el sistema a continuación |
| Modelo de lenguaje | decide entre las acciones permitidas y redacta la respuesta |
| Herramientas | actúan sobre el entorno, por ejemplo con la búsqueda en una base de datos |

## Origen

La librería implementa el agente híbrido de la tesis doctoral de Juan Manuel Fernández en la Universidad Nacional de
San Luis (Argentina). El agente se evaluó en SpokenWOZ sin utilizar las anotaciones del corpus, y la librería
reproduce esa evaluación (ver [Evaluación](evaluation.md)).

## Licencia

La librería y los codificadores TRACE se distribuyen con la licencia Apache 2.0. Los artefactos de dominio, y
cualquier dato derivado de un corpus, conservan la licencia de ese corpus: el artefacto de SpokenWOZ y los datos de
la demostración web se distribuyen con la licencia CC BY-NC 4.0, para uso no comercial y con atribución.
