# Conceptos

## Representaciones

- **Embeddings de turno** (Dialog2Flow, `e_t`): caracterizan la acción que realiza cada turno.
- **Representaciones contextuales** (TRACE, `h_t`): incorporan la historia del diálogo a cada turno, con solo los
  turnos anteriores, y ubican la conversación en un estado.
- **Representaciones del diálogo** (TRACE-Dialog, `h_CLS`): resumen el diálogo hasta el turno actual y se utilizan
  para recuperar situaciones parecidas.

## Estructura inducida

Las acciones son regiones del espacio de Dialog2Flow, y los estados son regiones de la combinación `[e_t ; λ h_t]` de
los embeddings de turno con las representaciones contextuales. Para cada estado, las acciones permitidas son las que
lo siguieron en los diálogos con una probabilidad superior a un umbral. La inducción de la estructura no requiere
anotaciones, y las acciones se nombran a partir de los turnos que agrupan.

## Memoria conversacional

Cada entrada de la memoria es un turno del usuario de un diálogo almacenado, representado por `[e_t ; λ h_CLS]`, junto
con la acción que realizó el sistema a continuación, su respuesta, el diálogo del cual proviene y el modo en que
terminó ese diálogo. La memoria recupera las situaciones más parecidas a la actual, con la opción de restringirlas al
mismo estado o a las situaciones cuya acción siguiente está permitida.

## Decidir puntuando

En lugar de generar una descripción libre de la acción siguiente, el modelo de lenguaje puntúa cada acción
permitida, es decir, la log-probabilidad de la salida `Action: <nombre>` seguida de `Response:` a continuación del
prompt, y el agente elige la más probable. La decisión es una elección entre opciones que define la estructura, por
lo cual nunca sale del conjunto permitido y devuelve una probabilidad para cada opción, en la línea de los modelos de
decisión como [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev). Las regiones que comparten un
nombre reciben la misma puntuación, y se elige la región más probable en el estado.

El agente también puede decidir por generación (`decision="generate"`), como referencia de la evaluación: el modelo
escribe la acción y la respuesta, y la acción declarada se proyecta sobre la acción inducida más cercana.

## Actuar y redactar

Cuando la acción decidida requiere información del entorno, se invoca una herramienta, como la búsqueda en una base de
datos. La respuesta se redacta a continuación de la acción decidida, con los resultados de la herramienta en el
prompt.

## Alcance

Un artefacto de dominio puede declarar su alcance, el cual indica si un turno del usuario se parece a los turnos de
los diálogos con los cuales se construyó el agente. El control utiliza la propia estructura y la memoria: un turno
queda fuera del alcance cuando su embedding de Dialog2Flow está lejos de todas las acciones inducidas del usuario o
cuando la memoria no contiene ninguna situación parecida (ver [Construir un dominio](build-a-domain.md#alcance)).
