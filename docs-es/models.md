# Modelos

El agente utiliza tres codificadores. [Dialog2Flow](https://huggingface.co/sergioburdisso/dialog2flow-joint-bert-base)
(Burdisso et al., 2024) representa cada turno por la acción que realiza. TRACE y TRACE-Dialog, entrenados en la tesis
doctoral en la cual se basa la librería, leen los embeddings de Dialog2Flow de los turnos y se publicaron en el Hub de
Hugging Face con la licencia Apache 2.0.

| Modelo | Repositorio | Salida | Uso en el agente |
|---|---|---|---|
| TRACE, profundo, autorregresivo | [`jumafernandez/trace-deep-ar`](https://huggingface.co/jumafernandez/trace-deep-ar) | un vector `h_t` por turno, a partir de ese turno y de los anteriores | ubicar el estado |
| TRACE-Dialog, profundo, con la etapa de similitud | [`jumafernandez/trace-dialog-deep-tod`](https://huggingface.co/jumafernandez/trace-dialog-deep-tod) | un vector `h_CLS` por diálogo | consultar la memoria |

Ambos modelos incluyen su propio código, por lo cual se cargan con `trust_remote_code=True`, y calculan los
embeddings de Dialog2Flow de los turnos con el codificador de base declarado en su configuración.

## TRACE con transformers

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("jumafernandez/trace-deep-ar", trust_remote_code=True)

turns = ["hi, I need a cheap restaurant in the centre", "sure, what kind of food would you like?", "italian"]
speakers = ["user", "system", "user"]
h = model.encode_dialogue(turns, speakers)   # (3, 768), un h_t por turno
```

## TRACE-Dialog con sentence-transformers

Cada entrada es un diálogo completo, como una lista de mensajes (`role` y `content`) o de turnos (`speaker` y `text`).

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("jumafernandez/trace-dialog-deep-tod", trust_remote_code=True)

dialogue = [{"role": "user", "content": "hi, I need a cheap restaurant in the centre"},
            {"role": "assistant", "content": "sure, what kind of food would you like?"},
            {"role": "user", "content": "italian"}]
embeddings = model.encode([dialogue])   # (1, 768), un h_CLS por diálogo
```

Los embeddings pueden compararse con las funciones de similitud de sentence-transformers, por ejemplo para recuperar
los diálogos de una colección más cercanos a uno en curso. TRACE-Dialog también se carga con `AutoModel`, cuyo
método `encode_dialogue(turns, speakers)` devuelve `h_CLS`.

## Dentro de la librería

`TraceEncoder` y `TraceDialogEncoder` cargan los modelos en las revisiones fijadas en
`dialog2agent.representations.REVISIONS`, de modo que una actualización posterior de los repositorios no modifique el
comportamiento de un agente. Un artefacto de dominio puede fijar otras revisiones en `domain.json`.

## Verificación

Los modelos publicados se compararon con los utilizados en la tesis sobre diálogos de SpokenWOZ:

| Modelo | Formato | Comparación | Diferencia absoluta máxima |
|---|---|---|---|
| TRACE | transformers | `h_t`, 50 diálogos | 1.2e-5 |
| TRACE-Dialog | transformers | `h_CLS`, 1.882 prefijos | 7.4e-6 |
| TRACE-Dialog | sentence-transformers | `h_CLS` desde el texto, 500 prefijos | 1.4e-5 |
