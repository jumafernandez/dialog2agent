# Servir el agente

`dialog2agent serve` expone un agente mediante una API HTTP que envuelve el mismo `Agent.respond` que se utiliza en
la evaluación. El servicio no conserva estado entre solicitudes: el cliente envía el diálogo hasta el momento, el cual
termina con un turno del usuario, y recibe el turno siguiente del sistema con la traza completa de la decisión. Por
lo tanto, cualquier réplica del servicio puede atender cualquier solicitud, y las réplicas pueden iniciarse y
detenerse en cualquier momento.

```bash
pip install "dialog2agent[web] @ git+https://github.com/jumafernandez/dialog2agent"
dialog2agent serve jumafernandez/dialog2agent-spokenwoz --port 8000
```

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | si el agente está cargado |
| GET | `/about` | versión de la librería, modos de decisión, alcance y atribución de los datos del artefacto |
| GET | `/structure` | acciones y estados inducidos, con las acciones permitidas de cada estado |
| POST | `/respond` | turno siguiente del sistema de un diálogo, con su traza; `decision` (`score` o `generate`) elige cómo se decide la acción en esa solicitud |
| POST | `/scope` | si un turno del usuario está dentro del alcance del dominio, cuando el artefacto declara un alcance |
| GET | `/v1/models` | modelo de la API compatible con OpenAI |
| POST | `/v1/chat/completions` | el agente a través de la API de chat completions de OpenAI |

```bash
curl -X POST http://localhost:8000/respond -H "Content-Type: application/json" \
  -d '{"turns": [{"speaker": "user", "text": "i need a cheap restaurant in the centre"}]}'
```

La respuesta contiene el texto, el estado, las acciones permitidas, la acción decidida, las opciones con su
probabilidad, las situaciones recuperadas y las llamadas a herramientas.

## API compatible con OpenAI

Cualquier cliente de la API de chat completions de OpenAI puede conversar con el agente. Los mensajes de sistema se
ignoran, los mensajes del usuario y del asistente son los turnos del diálogo, y la traza de la decisión se devuelve en
el campo adicional `trace` de la respuesta.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
reply = client.chat.completions.create(
    model="dialog2agent",
    messages=[{"role": "user", "content": "i need a cheap restaurant in the centre"}])
print(reply.choices[0].message.content)
```

De este modo conversan con el agente los usuarios simulados de SDialog (ver
[Evaluación](evaluation.md#simulacion-en-ciclo-cerrado-con-sdialog)).

## Opciones

| Opción | Efecto |
|---|---|
| `--decision`, `--no-adapter` | modo de decisión y adaptador del agente, como en `chat` |
| `--allowed-origins` | orígenes autorizados a invocar la API desde un navegador (cualquiera por defecto) |
| `--log ARCHIVO` | agrega cada respuesta, con el diálogo y su traza, a un archivo de líneas JSON |
| `--model-id` | nombre del modelo en la API compatible con OpenAI |

Las solicitudes se atienden de a una, dado que el agente mantiene un único modelo en el acelerador.

## Despliegue

El directorio `deploy/` del repositorio despliega el servicio en [Modal](https://modal.com), con una GPU bajo demanda
que se detiene cuando nadie utiliza el agente. Es el servicio detrás de la
[demostración web](https://jumafernandez.github.io/dialog2agent/demo.html), y su diseño, sus pasos y sus mediciones
se describen en [`deploy/README.md`](https://github.com/jumafernandez/dialog2agent/blob/main/deploy/README.md).
