# Evaluación

El agente de la tesis se evaluó en SpokenWOZ en tres niveles: la acción (exactitud de la acción decidida y coherencia
con las acciones permitidas), la anotación humana (coincidencia con los actos de diálogo anotados, que el agente nunca
utiliza) y la tarea (Inform, Success y BLEU con el evaluador oficial del corpus). La librería predice los turnos de un
corpus y calcula las métricas de la acción, y sus predicciones contienen las respuestas a partir de las cuales el
evaluador oficial calcula las métricas de la tarea. Una segunda evaluación pone al agente en conversaciones completas
con usuarios simulados.

## Predicción de un corpus

`dialog2agent evaluate` predice cada turno del sistema que sigue a un turno del usuario con `Agent.respond`, a partir
del diálogo real hasta ese turno, de modo que cada turno se evalúa con la misma historia que la referencia. Los
diálogos son líneas JSON:

```json
{"dialogue_id": "MUL0001", "turns": [{"speaker": "user", "text": "..."}, {"speaker": "system", "text": "..."}]}
```

```bash
dialog2agent evaluate mi-dominio dialogos.jsonl --output predicciones/
```

Cada predicción registra el estado, las acciones permitidas, la acción decidida y su probabilidad, la respuesta y la
acción inducida del turno real del sistema, es decir, la acción del sistema de la estructura más cercana a su
embedding de Dialog2Flow. Por lo tanto, la evaluación no requiere anotaciones.

Con `--no-write` solo se deciden las acciones, lo cual resulta mucho más rápido. Las ejecuciones pueden interrumpirse
y retomarse, dado que se omiten los turnos ya presentes en la salida.

## Varios aceleradores

Cada proceso toma los diálogos de su rango y escribe su propio archivo, de modo que la evaluación se ejecuta con
`torch.distributed.run` sin comunicación entre procesos:

```bash
python -m torch.distributed.run --standalone --nproc_per_node 8 -m dialog2agent.cli evaluate \
    mi-dominio dialogos.jsonl --output predicciones/ --require-accelerator
```

En un clúster sin acceso al Hub, `--base-model` y `--turn-encoder` apuntan a copias locales de los modelos, y los
codificadores TRACE se leen del caché de Hugging Face en sus revisiones fijadas.

## Métricas

```bash
dialog2agent report predicciones/
```

informa la exactitud de la acción decidida por región y por nombre, y la cobertura de la acción real por parte de las
acciones permitidas, sobre todos los turnos predichos. Los turnos cuya acción real no está permitida cuentan como
errores.

## Reproducción de la tesis

La evaluación de la tesis se repitió con la librería sobre los 17.782 turnos del sistema de los 1.000 diálogos de
prueba de SpokenWOZ. El estado, las acciones permitidas y la acción inducida de la referencia coincidieron con los de
la tesis en todos los turnos, y la acción decidida coincidió en el 96,6 % de ellos. Las diferencias restantes provienen
de dos detalles de implementación que solo afectan los empates entre opciones equivalentes: la tesis calculó las
similitudes de la memoria en bfloat16, y desempató las regiones con el mismo nombre y el mismo conteo por su orden de
aparición.

| | Exactitud (región) | Exactitud (nombre) | Inform | Success | BLEU | Comb |
|---|---|---|---|---|---|---|
| Tesis | 0,557 | 0,570 | 70,9 | 54,0 | 23,2 | 85,6 |
| dialog2agent | 0,556 | 0,569 | 70,6 | 53,7 | 23,2 | 85,4 |

La diferencia de exactitud, de −0,0005 (IC 95 %: −0,0024 a 0,0014, bootstrap pareado por diálogo), no resultó
significativa.

## Simulación en ciclo cerrado con SDialog

La evaluación anterior predice cada turno a partir de la historia real, de modo que una decisión errónea no modifica
los turnos siguientes. Con el fin de observar el agente en conversaciones completas, usuarios simulados de
[SDialog](https://github.com/idiap/sdialog) conversan con el agente a través de la API compatible con OpenAI del
servicio web (ver [Servir el agente](serving.md)). Cada usuario recibe una persona cuyo trasfondo enuncia la meta de
un diálogo de prueba, es decir, las restricciones de cada servicio, la información a solicitar, la reserva y los datos
personales.

```python
from sdialog.agents import Agent
from sdialog.personas import Persona

persona = Persona(name="Alex Smith", language="English",
                  role="a customer who calls the Cambridge town information centre on the phone",
                  background="You are looking for a train from ely to cambridge on wednesday that arrives by 08:15. "
                             "You want to know its departure time.")
user = Agent(persona=persona, name="User", model="openai:user-model", openai_api_base="http://localhost:8101/v1",
             openai_api_key="EMPTY", temperature=0)
system = Agent(name="System", model="openai:dialog2agent", openai_api_base="http://localhost:8000/v1",
               openai_api_key="EMPTY", temperature=0, can_finish=False)
dialog = user.dialog_with(system, max_turns=40)
dialog.to_file("dialog.json")
```

En la tesis, el usuario simulado fue el modelo de 27 mil millones de parámetros de la familia Qwen 3.5, las
conversaciones las juzgó Claude Opus 5.5 sin conocer la variante que las produjo, y las diferencias entre variantes se
estimaron con bootstrap pareado por meta. La
[demostración web](https://jumafernandez.github.io/dialog2agent/demo.html) reproduce algunas de estas conversaciones
con la traza de cada decisión.
