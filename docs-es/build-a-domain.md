# Construir un dominio

Un artefacto de dominio contiene todo lo que el agente aprende de una colección de diálogos, de modo que el mismo
código del agente sirve para cualquier dominio del cual se disponga de su artefacto.

```
mi-dominio/
  domain.json        # modelo de base, codificadores y configuración del agente
  structure/         # estructura inducida (structure.npz, structure.json)
  memory/            # situaciones (vectors.npy, entries.jsonl, memory.json)
  adapter/           # adaptador LoRA del modelo de lenguaje (opcional)
  db/                # base de datos de las herramientas (opcional, <dominio>_db.json y booking.json)
```

## domain.json

| Clave | Contenido |
|---|---|
| `base_model` | modelo de lenguaje, por ejemplo `Qwen/Qwen3.5-9B` |
| `turn_encoder` | codificador de turnos de Dialog2Flow |
| `context_encoder`, `context_encoder_revision` | codificador TRACE y su commit |
| `dialogue_encoder`, `dialogue_encoder_revision` | codificador TRACE-Dialog y su commit |
| `k`, `conditioned` | cantidad de situaciones recuperadas y si la recuperación se condiciona por la estructura |
| `decision_scale` | calibración de las probabilidades de la decisión |
| `max_tokens` | presupuesto del prompt |
| `search_triggers` | palabras de los nombres de las acciones que invocan la herramienta de búsqueda |
| `tools` | acciones inducidas, por número, que invocan cada herramienta, por ejemplo `{"search": [43, 9], "book": [8]}` |
| `tool_results` | `"message"` para entregar los resultados al modelo como un mensaje con el rol de herramienta |
| `scope` | umbrales del alcance del dominio (ver más abajo) |
| `data` | nombre, cita, URL y licencia de los diálogos con los cuales se construyó el artefacto |

`db/booking.json` declara los parámetros de la reserva de cada dominio, por ejemplo
`{"restaurant": ["people", "day", "time"]}`, los cuales pertenecen a la interfaz de reservas y no a la base. Un archivo
de la base que contiene un único objeto de listas de valores, como el de los taxis de MultiWOZ, describe un servicio
sin entidades, cuyas reservas se arman a partir de esas listas.

Las revisiones de los codificadores quedan fijadas porque la memoria se calcula con una versión determinada de cada
uno. El servicio web devuelve la entrada `data`, de modo que cualquier interfaz construida sobre él pueda mostrar la
atribución.

## Construcción

```bash
dialog2agent build mis-dialogos.jsonl --output dominios/mi-dominio
```

Cada línea del archivo de entrada es un diálogo con sus turnos:

```json
{"dialogue_id": "d1", "turns": [{"speaker": "user", "text": "hi"}, {"speaker": "system", "text": "hello"}]}
```

!!! note "Nota"
    El comando `build` se encuentra en desarrollo. El artefacto de SpokenWOZ se exportó a partir de la estructura, la
    memoria y el adaptador de la tesis.

## Alcance

El alcance de un dominio se calibra sobre turnos del usuario que no se utilizaron para construir el artefacto, como
los de sus diálogos de desarrollo. Cada umbral es un percentil bajo de su señal sobre esos turnos, y un turno queda
fuera del alcance cuando cualquiera de las dos señales cae por debajo de su umbral.

```python
from dialog2agent import Agent, Scope

agent = Agent.from_pretrained("dominios/mi-dominio")
scope = Scope.calibrate(agent, turnos_de_usuario_de_desarrollo, percentile=1, language="en",
                        description="calls to a town information centre")
print(scope.to_dict())   # se guarda en la clave "scope" de domain.json
print(agent.check_scope("what is the weather like tomorrow?"))
```

En SpokenWOZ, con el percentil 1 sobre los 9.169 turnos del usuario de los diálogos de desarrollo, quedan fuera del
alcance el 1,7 % de 5.000 turnos del usuario de la partición de prueba, en su mayoría cierres y turnos con errores de
transcripción. También quedan fuera todos los mensajes en español escritos para la prueba, dado que los codificadores
se entrenaron con diálogos en inglés, y 16 de 30 pedidos en inglés ajenos al dominio.
