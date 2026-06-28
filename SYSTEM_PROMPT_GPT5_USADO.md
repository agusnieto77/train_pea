# SYSTEM PROMPT — usado para producir el training data (GPT-5.4-mini + validación humana)

**Origen:** `batch_requests_eventos_protesta.jsonl` (batch histórico)
**Verificado:** 1 system prompt único en el batch histórico (hash confirmado)
**Batch OpenAI:** `batch_6a27123bea948190be44a836334d74ff`
**Endpoint:** `POST /v1/responses`
**Modelo formal del training data:** `gpt-5.4-mini` + validación humana de Nico
**Reasoning:** `{"effort": "medium"}`
**max_output_tokens:** `16000`
**text.verbosity:** `low`
**Fecha de envío:** 2026-06-09 (timestamp `created_at: 1780945467`)
**Length:** 1873 caracteres

---

## ⚠️ Por qué importa para fine-tuning

Este es **exactamente** el system prompt histórico usado para generar las extracciones;
el dataset canónico actual es `entrenamiento.jsonl` con 317 ejemplos validados.
Si entrenamos a Qwen con cualquier variación de este
prompt, vamos a tener **prompt drift** entre entrenamiento e inferencia, lo que
degrada F1 y categorical accuracy.

**Regla:** usar este texto LITERAL (incluyendo encoding sin tildes) en el
`system` message del ChatML de entrenamiento Y en el `system` message de vLLM
al hacer inference post-fine-tuning.

**Aclaración de validación:** la frase histórica sobre `"S/D"` aplica a campos
textuales/categoriales desconocidos dentro de eventos reales. En el gold
validado, si `es_evento_protesta=false`, los campos de detalle del evento van en
`null`, no en `"S/D"`. No editar el bloque literal de abajo para evitar prompt
drift; aplicar esta aclaración como regla de validación externa.

---

## Prompt (texto exacto, sin tildes intencionalmente)

```
Eres un cientifico social experto en codificacion de eventos de protesta en notas periodisticas historicas.

Contrato de extraccion:
- Unidad de registro: la nota periodistica completa. Devuelve un unico objeto JSON para la nota.
- Unidad de analisis: el evento de protesta o accion conflictiva.
- Una nota puede contener cero, uno o varios eventos. Cada item de extraccion.eventos_protesta debe representar exactamente un evento/accion atomico.
- Si la nota identifica mas de una accion, separalas solo cuando puedan delimitarse espacial y/o temporalmente con claridad.
- No confundas acciones complementarias con eventos independientes. Ejemplo: un corte con volanteada, junta de firmas, carpa o quema de gomas suele ser un unico evento con accion principal y formatos complementarios, salvo delimitacion temporal/espacial clara.
- Un mismo evento puede tener multiples sujetos, organizaciones, demandas, destinatarios contra_quien y lugares; representalos como arrays sin duplicar eventos.
- Prioridad absoluta: extrae primero la cita textual exacta de la nota y luego clasifica segun las categorias del esquema.
- Si una variable no aparece en el texto, usa estrictamente "S/D" en campos textuales/categoriales y null en campos numericos no observados.
- Las fechas de salida deben estar en formato DD/MM/AAAA. La primera linea del input es la fecha de edicion/publicacion; usala para resolver expresiones relativas como "ayer", "hoy", "manana", "el jueves", etc.
- Si el texto anuncia una accion futura, registra como evento el anuncio, no la accion futura realizada, salvo que la nota tambien informe que esa accion efectivamente ocurrio.
- Para ahorrar salida, en nota.texto_original devuelve "S/D"; no repitas el texto completo de la nota.
- No inventes informacion. Cuando haya ambiguedad, registrala en calidad_extraccion.ambiguedades u observaciones_extraccion.
```

---

## Notas de encoding

- El texto no tiene tildes ni eñes (`manana` en lugar de `mañana`, `codigo`
  en lugar de `código`, `cientifico` en lugar de `científico`). Esto es
  **intencional**: así se envió al modelo y así lo consumimos. No normalizar.
- Las comillas tipográficas (`"`) están reemplazadas por ASCII (`"`).
- Hay saltos de línea `\n` literales como separadores de bullets.
- El texto termina sin newline final.

---

## Cuerpo completo (system + user + body params) — referencia para el script de fine-tuning

```json
{
  "model": "gpt-5.4-mini",
  "input": [
    {
      "role": "system",
      "content": "<este archivo: pegar SYSTEM PROMPT completo arriba>"
    },
    {
      "role": "user",
      "content": "<ver USER_MESSAGE_TEMPLATE_GPT5.md>"
    }
  ],
  "reasoning": {"effort": "medium"},
  "text": {
    "verbosity": "low",
    "format": <JSON Schema — pegar contenido de esquema_eventos_protesta_entrenamiento_MVS.json>
  },
  "max_output_tokens": 16000
}
```

> Para el fine-tuning de Qwen NO se envía `text.format` (Qwen no usa
> Structured Outputs nativo). En su lugar, el JSON Schema va al system prompt
> como anexo de paths MVS (ver `PLAN_ENTRENAMIENTO_QWEN.md` §5 Fase 1).

---

## Pie: origen formal del training data

- Los 317 ejemplos canónicos de `entrenamiento.jsonl` se tratan formalmente como producidos por **GPT-5.4-mini** y validados humanamente por Nico.
- Cualquier mención histórica a `gpt-5.5` en requests/scripts queda como ruido documental o configuración operativa para regeneraciones futuras; **no** define el origen ni el baseline del training set actual.
- Baseline correcto para evaluación: **GPT-5.4-mini + validación humana Nico**.

Recordatorio: las 317 filas canónicas son validadas por Nico; el bloque `validacion_humana`
solo indica edición humana, no mayor o menor calidad.
