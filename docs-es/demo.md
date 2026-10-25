# Demostración web

[Abrir la demostración web](https://jumafernandez.github.io/dialog2agent/demo.html){ .md-button .md-button--primary }

La demostración muestra al agente en funcionamiento. La conversación transcurre junto a la estructura inducida: los
estados se dibujan sobre el plano de las dos primeras componentes principales de sus centros, el estado actual se
resalta y el recorrido de la conversación se traza sobre ellos. Para cada turno del agente, el panel de decisión
muestra las acciones permitidas en el estado con la probabilidad que el modelo asigna a cada una, y el panel de
memoria muestra las situaciones de otras conversaciones que recuperó el agente. Al pasar sobre un estado, o al
tocarlo, se muestran sus acciones más probables y dos turnos típicos del usuario en ese estado.

## Cómo decide el agente

Un selector en la parte superior alterna entre tres formas de decidir la acción, con sus métricas de referencia. El
agente decide como los modelos de decisión de
[TypeSafe AI](https://typesafe.ai/blog/introducing-system-one-models-and-jev), por ejemplo Jev: en lugar de escribir
la acción, puntúa cada opción de un conjunto definido de antemano, en este caso las acciones permitidas en el estado,
y devuelve una probabilidad para cada una, con el mismo modelo adaptado y pesos abiertos. Las otras dos variantes son
el mismo modelo adaptado cuando genera la acción y el modelo de base sin el adaptador. El replay muestra la misma meta
resuelta por cada una de ellas.

## Replay y conversación en vivo

Por defecto, la demostración reproduce conversaciones de la simulación en ciclo cerrado con
[SDialog](https://github.com/idiap/sdialog), en la cual usuarios simulados con metas de la partición de prueba de
SpokenWOZ conversan con las tres variantes (ver [Evaluación](evaluation.md#simulacion-en-ciclo-cerrado-con-sdialog)).
Las metas se eligieron por dominio, sin observar los juicios, y para cada conversación se muestra el veredicto del
juez.

El botón "conversar con el agente" despierta el agente servido en Modal (ver [Servir el agente](serving.md)) y abre
una conversación en vivo con las dos primeras variantes, que comparten el modelo adaptado. La página nunca despierta
el servicio por sí misma, dado que cada solicitud a un servicio detenido inicia una GPU, y el primer arranque demora
alrededor de un minuto. En la conversación en vivo el agente tiene herramientas: cuando la acción que decide es de
herramienta, busca en la base de SpokenWOZ o hace una reserva simulada, y la llamada, con sus argumentos y su
resultado, se muestra debajo de la respuesta. El replay proviene de una simulación en la cual el agente todavía no
tenía herramientas, por lo cual los turnos en los cuales anuncia una búsqueda llevan una nota.

Antes de que un turno del visitante llegue al agente, la página controla que esté en inglés, por las palabras
funcionales de cada idioma, y el servicio controla que esté dentro del alcance del dominio (ver
[Construir un dominio](build-a-domain.md#alcance)). El dominio se controla hasta el primer turno que cae dentro de él,
es decir, sobre el pedido con el cual se abre la conversación, dado que los cierres y los turnos breves, como los
nombres deletreados, quedan con frecuencia por debajo de los umbrales. Un turno fuera del alcance queda en la
conversación, marcado, con un aviso que muestra el valor de cada señal y su umbral, y no se envía al agente.

## Construcción y servicio

Los datos del replay se construyen a partir de un directorio de simulación y del artefacto de dominio:

```bash
python demo/build_replay.py --domain DIR_DOMINIO --simulation DIR_SIMULACION --out demo/data/replay.json
python -m http.server --directory demo 8000    # luego abrir http://localhost:8000/demo.html
```

Los datos del replay contienen texto derivado de los diálogos de SpokenWOZ, por lo cual se distribuyen con la licencia
CC BY-NC 4.0, con la atribución que la página muestra al pie (ver `demo/data/LICENSE.md`). El sitio de esta
documentación publica la demostración en `demo.html`, con el workflow `.github/workflows/pages.yml`.
