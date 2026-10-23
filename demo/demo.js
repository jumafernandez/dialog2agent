"use strict";
// Demo of dialog2agent: replays conversations of the closed-loop simulation with SDialog, with the trace of every
// decision, and talks to the agent served on Modal when the visitor wakes it up. The page never wakes the service on
// its own, since every request to a stopped service starts a GPU.

const API = "https://jumafernandez--dialog2agent-spokenwoz-service-web.us-east.modal.run";
const MAX_TURNOS_VIVO = 12;
// Reference metrics of chapter X (SpokenWOZ test) and of the closed-loop simulation (exploratory sample, 100 goals).
const VARIANTES = {
  puntuacion: { decision: "score", vivo: true, exactitud: 0.557, comb: 85.6, meta: 0.68 },
  generacion: { decision: "generate", vivo: true, exactitud: 0.422, comb: 71.7, meta: 0.58 },
  sin_adaptar: { decision: null, vivo: false, exactitud: null, comb: 61.4, meta: 0.55 },
};
const DECIMALES = { exactitud: 3, comb: 1, meta: 2 };

const TEXTOS = {
  es: {
    lema: "Un agente de diálogo orientado a tareas que decide sobre acciones inducidas de conversaciones reales, con memoria de otras conversaciones y sin anotaciones.",
    decisor: "Cómo decide el agente", decisorNota: "métricas de referencia",
    decisorExplica: "El decisor tipo Jev sigue la idea de los modelos de decisión de TypeSafe AI: en lugar de escribir la acción, puntúa cada opción de un conjunto definido de antemano, acá las acciones permitidas en el estado de la conversación, y devuelve una probabilidad para cada una. Lo implementamos de forma abierta con el mismo LLM adaptado. Las otras dos variantes escriben la acción libremente. Exactitud de la acción y Comb (Inform, Success y BLEU) sobre los 1.000 diálogos de prueba de SpokenWOZ; meta cumplida según un juez (Claude Opus 5.5) en 100 conversaciones con usuarios simulados con SDialog.",
    nombres: { puntuacion: "Decisor tipo Jev (abierto)", generacion: "LLM que genera la acción", sin_adaptar: "LLM sin adaptar" },
    como: { puntuacion: "puntúa las acciones permitidas del estado y devuelve una probabilidad para cada una", generacion: "el mismo modelo adaptado escribe la acción", sin_adaptar: "el modelo base, solo con el prompt" },
    exactitud: "exactitud", comb: "Comb", meta: "meta cumplida",
    conversacion: "Conversación", metaSel: "Meta", enviar: "enviar", siguiente: "siguiente", reiniciar: "reiniciar",
    conversar: "conversar con el agente", pausa: "pausa", reproducir: "reproducir",
    estructura: "Estructura inducida", decision: "Decisión", memoria: "Memoria", memoriaNota: "situaciones parecidas",
    tesis: "tesis doctoral", replay: "replay · SDialog", vivo: "en vivo", despertandoChip: "despertando",
    usuario: "usuario", agente: "agente", accion: "acción", pensando: "ubicando el estado y decidiendo…",
    tam: (n, a) => `${n} estados · ${a} acciones`, estado: (s, a) => `estado <b>${s}</b> · acción más probable: <b>${a}</b>`,
    permitidas: (n) => `${n} permitidas`, notaPuntuacion: "probabilidad que el modelo asigna a cada acción permitida",
    generada: "acción generada", dentro: "La acción generada está entre las acciones permitidas del estado.", fuera: "La acción generada no está entre las acciones permitidas del estado.",
    veredicto: (m, c) => `juez: meta ${m ? '<span class="si">cumplida</span>' : '<span class="no">no cumplida</span>'} · calidad ${c}/5`,
    turno: (i, n) => `turno ${i} de ${n}`, simulada: "Meta del usuario simulado: ",
    despertando: (s) => `Despertando al agente en la GPU… ${s} s. El primer arranque tarda alrededor de un minuto.`,
    listo: "El agente está listo. Escribile en inglés como si llamaras al centro de información de Cambridge: restaurantes, hoteles, atracciones, trenes, taxis.",
    libre: "Conversación libre con el agente servido en Modal (una GPU L4 que se apaga sola).",
    error: "No se pudo contactar al agente. Probá de nuevo en un rato.", limite: "Llegaste al máximo de turnos de esta sesión.",
    soloReplay: "solo en el replay", ficha: (s, n) => `estado ${s} · ${n} turnos de entrenamiento`, tipicos: "turnos típicos del usuario",
    probables: "acciones más probables",
    idioma: "El agente conversa solo en inglés: su estructura y su memoria se construyeron con conversaciones de SpokenWOZ, que están en inglés. Probá, por ejemplo, «I need a cheap restaurant in the centre».",
    dominio: (a, m, ua, um) => `Ese pedido queda fuera de lo que el agente conoce, que son las llamadas al centro de información de Cambridge: restaurantes, hoteles, atracciones, trenes, taxis, hospital y policía. El turno no cae cerca de las regiones de la estructura: acción de usuario más cercana ${a} (umbral ${ua}), situación más parecida de la memoria ${m} (umbral ${um}).`,
  },
  en: {
    lema: "A task-oriented dialogue agent that decides over actions induced from real conversations, with a memory of other conversations and without annotations.",
    decisor: "How the agent decides", decisorNota: "reference metrics",
    decisorExplica: "The Jev-style decider follows the idea of the decision models of TypeSafe AI: instead of writing the action, it scores every option of a set defined in advance, here the actions allowed in the state of the conversation, and returns a probability for each one. We implemented it openly with the same adapted LLM. The other two variants write the action freely. Action accuracy and Comb (Inform, Success and BLEU) on the 1,000 SpokenWOZ test dialogues; goal completion according to a judge (Claude Opus 5.5) on 100 conversations with users simulated with SDialog.",
    nombres: { puntuacion: "Jev-style decider (open)", generacion: "LLM generating the action", sin_adaptar: "Unadapted LLM" },
    como: { puntuacion: "scores the allowed actions of the state and returns a probability for each one", generacion: "the same adapted model writes the action", sin_adaptar: "the base model, with the prompt only" },
    exactitud: "accuracy", comb: "Comb", meta: "goal completed",
    conversacion: "Conversation", metaSel: "Goal", enviar: "send", siguiente: "next", reiniciar: "restart",
    conversar: "talk to the agent", pausa: "pause", reproducir: "play",
    estructura: "Induced structure", decision: "Decision", memoria: "Memory", memoriaNota: "similar situations",
    tesis: "doctoral thesis", replay: "replay · SDialog", vivo: "live", despertandoChip: "waking up",
    usuario: "user", agente: "agent", accion: "action", pensando: "locating the state and deciding…",
    tam: (n, a) => `${n} states · ${a} actions`, estado: (s, a) => `state <b>${s}</b> · most probable action: <b>${a}</b>`,
    permitidas: (n) => `${n} allowed`, notaPuntuacion: "probability the model assigns to each allowed action",
    generada: "generated action", dentro: "The generated action is one of the actions allowed in the state.", fuera: "The generated action is not one of the actions allowed in the state.",
    veredicto: (m, c) => `judge: goal ${m ? '<span class="si">completed</span>' : '<span class="no">not completed</span>'} · quality ${c}/5`,
    turno: (i, n) => `turn ${i} of ${n}`, simulada: "Goal of the simulated user: ",
    despertando: (s) => `Waking up the agent on the GPU… ${s} s. The first start takes about a minute.`,
    listo: "The agent is ready. Write as if you were calling the Cambridge information centre: restaurants, hotels, attractions, trains, taxis.",
    libre: "Free conversation with the agent served on Modal (an L4 GPU that turns itself off).",
    error: "The agent could not be reached. Try again in a while.", limite: "You reached the maximum number of turns of this session.",
    soloReplay: "replay only", ficha: (s, n) => `state ${s} · ${n} training turns`, tipicos: "typical user turns",
    probables: "most probable actions",
    idioma: "The agent only talks in English: its structure and its memory were built from SpokenWOZ conversations, which are in English. Try, for example, “I need a cheap restaurant in the centre”.",
    dominio: (a, m, ua, um) => `That request is outside what the agent knows, which is calls to the Cambridge information centre: restaurants, hotels, attractions, trains, taxis, hospital and police. The turn does not fall near the regions of the structure: closest user action ${a} (threshold ${ua}), most similar memory situation ${m} (threshold ${um}).`,
  },
};

// Turns in Spanish get a notice without reaching the service, since the encoders and the memory of the domain are in
// English: a turn is taken as Spanish when it has more Spanish than English function words. Names of places of the
// domain such as "la raza" or "el shaddai" are not counted.
const PALABRAS_ES = new Set(("hola gracias buenas buenos buen dia día dias días necesito necesitaría quiero querría busco quisiera " +
  "podrías podria podría puede puedo hay un una uno unos unas los las del al con para por que qué donde dónde cuando cuándo " +
  "como cómo cual cuál cuánto cuesta es esta está estoy tengo mi mis tu su sus de en y pero muy mas más eso todo algo algún " +
  "alguna lugar comprar reservar reserva mesa personas mañana tarde noche hoy lunes martes miércoles jueves viernes sábado " +
  "domingo precio barato barata caro comida restaurante estación aeropuerto museo favor sí bueno dale che vos usted también ahora").split(" "));
const PALABRAS_EN = new Set(("i you the a an to is are am be for and of in on at with my me need want looking like would could " +
  "can please thanks thank hello hi yes no what where when how there it that this").split(" "));
function pareceEspanol(texto) {
  const palabras = texto.toLowerCase().match(/[a-záéíóúüñ]+/g) || [];
  return palabras.filter((w) => PALABRAS_ES.has(w)).length > palabras.filter((w) => PALABRAS_EN.has(w)).length;
}

const $ = (id) => document.getElementById(id);
const quieto = matchMedia("(prefers-reduced-motion: reduce)").matches;
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
let lang = (() => { try { return localStorage.getItem("d2a-lang") || (navigator.language.startsWith("es") ? "es" : "en"); } catch { return "es"; } })();
let DATOS = null, ALCANCE = null, variante = "puntuacion", metaIdx = 0, modo = "replay";
let paso = 0, reproduciendo = true, temporizador = null, recorrido = [], actual = null, hover = null, turnosVivo = [];
const T = () => TEXTOS[lang];

function aplicarIdioma() {
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-i18n]").forEach((e) => { const v = T()[e.dataset.i18n]; if (typeof v === "string") e.textContent = v; });
  document.querySelectorAll(".idiomas button").forEach((b) => b.classList.toggle("activo", b.dataset.lang === lang));
  $("texto").placeholder = "i need a cheap restaurant in the centre";
  $("play").textContent = reproduciendo ? T().pausa : T().reproducir;
  if (DATOS) {
    $("tam-estructura").textContent = T().tam(DATOS.structure.states.length, DATOS.structure.actions.length);
    $("atribucion").textContent = DATOS.data ? `${DATOS.data.name} · ${DATOS.data.license} · ${DATOS.data.citation.split(". ")[0]}.` : "";
    pintarDecisores(); pintarMeta(); pintarModo();
  }
}

function pintarDecisores() {
  const caja = $("decisores"); caja.textContent = "";
  for (const [v, m] of Object.entries(VARIANTES)) {
    const b = el("button", "decisor" + (v === variante ? " activo" : ""));
    const deshabilitado = modo !== "replay" && !m.vivo;
    b.setAttribute("aria-disabled", deshabilitado ? "true" : "false");
    b.append(el("span", "nombre", T().nombres[v]), el("span", "como", deshabilitado ? T().soloReplay : T().como[v]));
    const met = el("span", "metricas");
    for (const k of ["exactitud", "comb", "meta"]) {
      const val = m[k] === null ? "—" : m[k].toLocaleString(lang === "es" ? "es-AR" : "en-US", { minimumFractionDigits: DECIMALES[k], maximumFractionDigits: DECIMALES[k] });
      const s = el("span"); s.append(el("b", null, val), T()[k]); met.append(s);
    }
    b.append(met);
    b.onclick = () => { if (deshabilitado || v === variante) return; variante = v; pintarDecisores(); if (modo === "replay") reiniciarReplay(); };
    caja.append(b);
  }
}

function pintarModo() {
  const chip = $("modo");
  chip.className = "chip modo" + (modo === "vivo" ? " vivo" : "");
  chip.textContent = modo === "vivo" ? T().vivo : modo === "despertando" ? T().despertandoChip : T().replay;
  $("entrada").hidden = modo !== "vivo";
  for (const id of ["play", "sig"]) $(id).hidden = modo !== "replay";
  $("vivo").hidden = modo !== "replay";
  $("meta-sel").disabled = modo !== "replay";
}

function pintarMeta() {
  const sel = $("meta-sel");
  if (!sel.options.length) DATOS.goals.forEach((g, i) => sel.append(new Option(`${g.id} · ${g.domains.join(" + ")}`, i)));
  sel.value = metaIdx;
  if (modo !== "replay") { $("meta").textContent = T().libre; $("veredicto").textContent = ""; return; }
  const g = DATOS.goals[metaIdx], j = g.variants[variante].judge;
  $("meta").textContent = T().simulada + g.goal;
  $("veredicto").innerHTML = j && j.goal !== null && j.goal !== undefined ? T().veredicto(j.goal, j.quality) : "";
}

// ---- background: values of the state coordinates floating up
const fondo = $("fondo"), fx = fondo.getContext("2d");
let nums = [];
function medirFondo() {
  fondo.width = innerWidth * devicePixelRatio; fondo.height = innerHeight * devicePixelRatio;
  fx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  const estados = DATOS ? DATOS.structure.states : [{ x: 0.3, y: 0.7 }];
  nums = Array.from({ length: Math.round(innerWidth * innerHeight / 22000) }, (_, i) => {
    const e = estados[i % estados.length], v = (e.x - 0.5) * 0.12 + (e.y - 0.5) * 0.05;
    return { t: (v >= 0 ? "+" : "") + v.toFixed(4), x: Math.random() * innerWidth, y: Math.random() * innerHeight, vy: -(0.08 + Math.random() * 0.2), a: 0.05 + Math.random() * 0.12 };
  });
}
function pintarFondo() {
  fx.clearRect(0, 0, innerWidth, innerHeight); fx.font = "12px 'DM Mono', monospace";
  for (const n of nums) { if (!quieto) { n.y += n.vy; if (n.y < -20) n.y = innerHeight + 20; } fx.fillStyle = `rgba(77,226,255,${n.a})`; fx.fillText(n.t, n.x, n.y); }
  requestAnimationFrame(pintarFondo);
}

// ---- induced structure: states on the plane, path of the conversation and current state
const g = $("grafo"), gx = g.getContext("2d");
let GW = 0, GH = 0;
function medirGrafo() { GW = g.clientWidth; GH = g.clientHeight; g.width = GW * devicePixelRatio; g.height = GH * devicePixelRatio; gx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0); }
const pos = (e) => [24 + e.x * (GW - 48), 20 + (1 - e.y) * (GH - 40)];
const radio = (e, maxN) => 1.5 + 4 * Math.sqrt(e.n / maxN);
function pintarGrafo(t) {
  gx.clearRect(0, 0, GW, GH);
  if (DATOS) {
    const estados = DATOS.structure.states, maxN = Math.max(...estados.map((e) => e.n));
    estados.forEach((e, i) => {
      const [x, y] = pos(e);
      gx.beginPath(); gx.arc(x, y, radio(e, maxN) + (i === hover ? 2.5 : 0), 0, 6.283);
      gx.fillStyle = i === hover ? "#ffffff" : recorrido.includes(i) ? "rgba(255,91,216,0.9)" : "rgba(138,125,255,0.45)"; gx.fill();
    });
    if (recorrido.length > 1) {
      gx.beginPath(); recorrido.forEach((s, k) => { const [x, y] = pos(estados[s]); k ? gx.lineTo(x, y) : gx.moveTo(x, y); });
      gx.strokeStyle = "rgba(255,91,216,0.55)"; gx.lineWidth = 1.5; gx.setLineDash([4, 4]); gx.lineDashOffset = -(t || 0) / 60; gx.stroke(); gx.setLineDash([]);
    }
    if (actual !== null) {
      const [x, y] = pos(estados[actual]), r = 9 + 4 * Math.sin((t || 0) / 300);
      gx.beginPath(); gx.arc(x, y, r, 0, 6.283); gx.strokeStyle = "rgba(77,226,255,0.9)"; gx.lineWidth = 2; gx.stroke();
      gx.beginPath(); gx.arc(x, y, 4, 0, 6.283); gx.fillStyle = "#4de2ff"; gx.fill();
    }
  }
  requestAnimationFrame(pintarGrafo);
}
function fichaEstado(ev) {
  if (!DATOS) return;
  const r = g.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
  let mejor = null, dmin = 12;
  DATOS.structure.states.forEach((e, i) => { const [x, y] = pos(e), d = Math.hypot(x - mx, y - my); if (d < dmin) { dmin = d; mejor = i; } });
  hover = mejor; const f = $("ficha");
  if (mejor === null) { f.hidden = true; return; }
  const e = DATOS.structure.states[mejor], acc = DATOS.structure.actions;
  f.textContent = "";
  f.append(el("h4", null, T().ficha(mejor, e.n)), el("div", "acc", T().probables + ":"));
  for (const [a, p] of e.allowed.slice(0, 4)) { const d = el("div", "acc", acc[a] + " "); d.append(el("span", null, p.toFixed(2))); f.append(d); }
  if (e.examples.length) { f.append(el("div", "acc", T().tipicos + ":")); for (const x of e.examples) f.append(el("div", "ej", `“${x}”`)); }
  f.hidden = false;
  f.style.left = Math.max(0, Math.min(mx + 14, GW - 300)) + "px"; f.style.top = Math.max(0, my - 10) + "px";
}
g.addEventListener("mousemove", fichaEstado);
g.addEventListener("click", fichaEstado);  // touch screens have no hover
g.addEventListener("mouseleave", () => { hover = null; $("ficha").hidden = true; });

// ---- decision and memory panels
function nombreAccion(id) { return id >= 0 ? DATOS.structure.actions[id] : "—"; }
function mostrarPaso(p) {
  actual = p.state; recorrido.push(p.state);
  const e = DATOS.structure.states[p.state];
  $("estado").innerHTML = T().estado(p.state, nombreAccion(e.allowed[0][0]));
  $("permitidas").textContent = T().permitidas(p.allowed.length);
  const caja = $("opciones"); caja.textContent = "";
  if (p.decision === "score") {
    // regions that share a name receive the same score: they are shown once, with their total probability
    const porNombre = [];
    for (const o of p.options) {
      const n = nombreAccion(o.id), x = porNombre.find((y) => y.nombre === n);
      if (x) { x.p += o.p; x.regiones++; } else porNombre.push({ nombre: n, p: o.p, regiones: 1 });
    }
    porNombre.forEach((o, k) => {
      const d = el("div", "opcion" + (k === 0 ? " elegida" : "")), fila = el("div", "fila");
      fila.append(el("span", null, o.nombre + (o.regiones > 1 ? ` ×${o.regiones}` : "")), el("span", null, o.p.toFixed(3)));
      const barra = el("div", "barra"), i = el("i"); i.dataset.w = Math.max(2, o.p * 100); barra.append(i);
      d.append(fila, barra); caja.append(d);
    });
    requestAnimationFrame(() => caja.querySelectorAll(".barra i").forEach((i) => (i.style.width = i.dataset.w + "%")));
    $("nota-decision").textContent = T().notaPuntuacion;
  } else {
    const gen = el("div", "generada", T().generada + ": "); gen.append(el("b", null, nombreAccion(p.action))); caja.append(gen);
    const elegida = nombreAccion(p.action);
    for (const n of [...new Set(p.allowed.map(nombreAccion))]) caja.append(el("span", "permitida" + (n === elegida ? " elegida" : ""), n));
    $("nota-decision").textContent = p.allowed.includes(p.action) ? T().dentro : T().fuera;
  }
  const mem = $("memoria"); mem.textContent = "";
  for (const s of p.situations) {
    const d = el("div", "situacion"); d.append(el("span", "sim", s.similarity.toFixed(3)), el("div", null, s.user), el("div", "a", "→ " + s.action), el("div", "r", s.response)); mem.append(d);
  }
}
function burbuja(quien, texto, accion) {
  const b = el("div", "burbuja " + (quien === "User" ? "user" : "system"));
  b.append(el("span", "quien", quien === "User" ? T().usuario : T().agente));
  const txt = el("span"); b.append(txt);
  if (accion) { b.append(el("br"), el("span", "accion", `${T().accion}: ${accion}`)); }
  $("chat").append(b);
  if (quieto) txt.textContent = texto;
  else { let i = 0; const f = () => { txt.textContent = texto.slice(0, ++i); $("chat").scrollTop = 1e9; if (i < texto.length) setTimeout(f, 12); }; f(); }
  $("chat").scrollTop = 1e9;
  return b;
}
function limpiar() {
  clearTimeout(temporizador); paso = 0; recorrido = []; actual = null;
  for (const id of ["chat", "opciones", "memoria", "estado", "nota-decision", "permitidas"]) $(id).textContent = "";
}

// ---- replay of the simulated conversations
function pasoReplay(p) {
  return { state: p.state, action: p.action, allowed: p.allowed, options: p.options, situations: p.situations,
           decision: variante === "puntuacion" ? "score" : "generate" };
}
function avanzar() {
  const v = DATOS.goals[metaIdx].variants[variante], turnos = v.turns;
  if (paso >= turnos.length) { reproduciendo = false; $("play").textContent = T().reproducir; return; }
  const t = turnos[paso], p = v.steps.find((s) => s.turn === paso);
  if (t.speaker === "System" && p) {
    const pen = el("div", "pensando", T().pensando); $("chat").append(pen); $("chat").scrollTop = 1e9;
    mostrarPaso(pasoReplay(p));
    setTimeout(() => { pen.remove(); burbuja("System", t.text, nombreAccion(p.action)); }, quieto ? 0 : 900);
  } else burbuja(t.speaker, t.text);
  paso++; $("contador").textContent = T().turno(paso, turnos.length);
  if (reproduciendo) temporizador = setTimeout(avanzar, quieto ? 600 : 3200);
}
function reiniciarReplay() { limpiar(); pintarMeta(); reproduciendo = true; $("play").textContent = T().pausa; temporizador = setTimeout(avanzar, 400); }

// ---- live conversation with the agent served on Modal
async function despertar() {
  modo = "despertando"; limpiar(); pintarModo(); pintarDecisores(); pintarMeta();
  const aviso = el("div", "aviso"); $("chat").append(aviso);
  const inicio = Date.now(), reloj = setInterval(() => (aviso.textContent = T().despertando(Math.round((Date.now() - inicio) / 1000))), 500);
  try {
    const ctl = new AbortController(), corte = setTimeout(() => ctl.abort(), 180000);
    const r = await fetch(API + "/health", { signal: ctl.signal }); clearTimeout(corte);
    if (!r.ok) throw new Error(r.status);
    try { ALCANCE = (await (await fetch(API + "/about")).json()).scope; } catch { ALCANCE = null; }
    if (!VARIANTES[variante].vivo) variante = "puntuacion";
    modo = "vivo"; turnosVivo = []; aviso.textContent = T().listo;
  } catch { modo = "replay"; aviso.className = "aviso error"; aviso.textContent = T().error; }
  clearInterval(reloj); pintarModo(); pintarDecisores(); pintarMeta(); if (modo === "vivo") $("texto").focus();
}
// A turn out of the scope of the domain stays in the chat, marked, with a notice, and is not part of the dialogue. The
// domain is checked until the first turn within it, that is, on the request that opens the conversation, since closings
// and short turns such as spelled names often fall below the thresholds; the language is checked on every turn.
function fueraDeAlcance(b, aviso) {
  b.classList.add("fuera"); $("chat").append(el("div", "aviso fuera", aviso)); $("chat").scrollTop = 1e9;
}
const post = (ruta, cuerpo) => fetch(API + ruta, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(cuerpo) });
async function enviar(ev) {
  ev.preventDefault();
  const texto = $("texto").value.trim(); if (!texto) return;
  if (turnosVivo.filter((t) => t.speaker === "user").length >= MAX_TURNOS_VIVO) { $("chat").append(el("div", "aviso", T().limite)); return; }
  $("texto").value = ""; const b = burbuja("User", texto);
  if (pareceEspanol(texto)) { fueraDeAlcance(b, T().idioma); return; }
  const pen = el("div", "pensando", T().pensando); $("chat").append(pen); $("chat").scrollTop = 1e9;
  let agregado = false;
  try {
    if (ALCANCE && !turnosVivo.some((t) => t.speaker === "user")) {
      const r = await post("/scope", { text: texto }); if (!r.ok) throw new Error(r.status);
      const c = await r.json();
      if (!c.in_scope) {
        const f = (x) => x.toLocaleString(lang === "es" ? "es-AR" : "en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        pen.remove(); fueraDeAlcance(b, T().dominio(f(c.action_similarity), f(c.memory_similarity), f(ALCANCE.action_threshold), f(ALCANCE.memory_threshold)));
        return;
      }
    }
    turnosVivo.push({ speaker: "user", text: texto }); agregado = true;
    const r = await post("/respond", { turns: turnosVivo, decision: VARIANTES[variante].decision });
    if (!r.ok) throw new Error(r.status);
    const x = await r.json();
    mostrarPaso({ state: x.state, action: x.action.id, allowed: x.allowed.map((a) => a.id), situations: x.situations.slice(0, 3),
                  options: x.options.slice(0, 6).map((o) => ({ id: o.id, p: o.probability })), decision: x.decision });
    pen.remove(); burbuja("System", x.text, x.action.name); turnosVivo.push({ speaker: "system", text: x.text });
  } catch { pen.remove(); if (agregado) turnosVivo.pop(); $("chat").append(el("div", "aviso error", T().error)); }
}

// ---- controls
$("play").onclick = () => { reproduciendo = !reproduciendo; $("play").textContent = reproduciendo ? T().pausa : T().reproducir; clearTimeout(temporizador); if (reproduciendo) avanzar(); };
$("sig").onclick = () => { reproduciendo = false; $("play").textContent = T().reproducir; clearTimeout(temporizador); avanzar(); };
$("reinicio").onclick = () => { if (modo === "vivo") { limpiar(); turnosVivo = []; $("chat").append(el("div", "aviso", T().listo)); } else reiniciarReplay(); };
$("vivo").onclick = despertar;
$("entrada").onsubmit = enviar;
$("meta-sel").onchange = (e) => { metaIdx = +e.target.value; reiniciarReplay(); };
document.querySelectorAll(".idiomas button").forEach((b) => (b.onclick = () => { lang = b.dataset.lang; try { localStorage.setItem("d2a-lang", lang); } catch {} aplicarIdioma(); }));
addEventListener("resize", () => { medirFondo(); medirGrafo(); });

fetch("data/replay.json").then((r) => r.json()).then((d) => {
  DATOS = d; aplicarIdioma(); medirFondo(); medirGrafo(); pintarFondo(); requestAnimationFrame(pintarGrafo); reiniciarReplay();
});
