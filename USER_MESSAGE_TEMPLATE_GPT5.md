# USER MESSAGE TEMPLATE — usado para producir el training data

**Origen:** primera fila de `batch_requests_eventos_protesta.jsonl`
**Nota:** cada fila tiene su propio `custom_id` (= txt_file) y su propio user
message. En el repo actual, la reconstrucción más consistente con
`extraer_eventos_protesta_batch.py` y `entrenamiento.jsonl` es:
`FECHA DE EDICIÓN DE LA NOTA: {fecha_iso}\n\n{texto}`. La columna `texto` del CSV
ya incluye título + cuerpo.

---

## Plantilla

```python
def build_user_message(fecha_iso: str, texto_csv: str) -> str:
    return f"FECHA DE EDICIÓN DE LA NOTA: {fecha_iso}\n\n{texto_csv}"
```

---

## Ejemplo real (primera fila)

```
FECHA DE EDICIÓN DE LA NOTA: 1989-01-03

Izquierda Unida repudia atentado

La Izquierda Unida de Mar del Plata expresó en un documento su "repudio al salvaje atentado perpetrado contra la sede central del Partido Comunista en la Capital Federal", apuntando que "esta agresión contra el PC se encuadra en el avance del proyecto yanqui aprobado por la burguesía criolla de democracia con seguridad, o más claramente combinar el consenso que da el voto por los partidos del sistema con la represión a los luchadores sociales y a la izquierda política en general".

El documento está firmado por Santiago Navone (PC), Analía Di Giovanni (MAS), Mari Sosa (Peronismo Base) y Luis Pasik (Socialismo Primero de Mayo) y apunta también que "en Villa Martelli hubo varios heridos de bala y dos muertos, uno militante del Partido Comunista. Sin embargo, el presidente Alfonsín habló de que se había concluido el conflicto "sin daños personales y sin sangre".

Añade que "este avance de la represión, llevado a cabo por manos expertas provenientes del aparato represivo de la dictadura no desmantelado por el gobierno, viene de la mano del avance del partido militar y tiene al gobierno de Alfonsín como a la dirigencia política burguesa como cómplices y responsables de la restricción creciente de la democracia."
```

---

## ⚠️ Detalles a respetar

- `txt_file` es exactamente el `custom_id` del batch (sin extensión duplicada,
  sin slashes), pero no necesariamente aparece dentro del user message: el script lo
  usa como `custom_id` y luego fuerza `nota.nota_id`/`nota.archivo_fuente` al parsear.
- `fecha_iso` es **ISO 8601** (`YYYY-MM-DD`), NO el formato `DD/MM/AAAA` que usa
  el modelo en el output. La fecha de output es DD/MM/AAAA, la del input es YYYY-MM-DD.
- El separador entre la línea `FECHA DE EDICIÓN...` y el texto es `\n\n`.
- La columna `texto` del CSV ya incluye título + cuerpo; no pasar `titulo` por separado.
- `nota.texto_original` de `entrenamiento.jsonl` ya contiene el bloque completo usado como input base.

---

## Mapping con `entrenamiento.jsonl`

El `entrenamiento.jsonl` tiene `nota.nota_id` (= txt_file), `nota.fecha_publicacion`
(en DD/MM/AAAA), `nota.titulo` y `nota.texto_original`. Para Fase 1, el camino
preferido es usar `nota.texto_original` como user content base y proyectar el
assistant al MVS. Si se reconstruye desde CSV, usar `build_user_message(fecha_iso, texto_csv)`.

```python
import json
from datetime import datetime

def ddmmyyyy_to_iso(s: str) -> str:
    if s == 'S/D':
        return s
    return datetime.strptime(s, '%d/%m/%Y').strftime('%Y-%m-%d')

with open(r'entrenamiento.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        nota = row['nota']
        user_msg = nota['texto_original']
        # ... pasar a Qwen como user message en ChatML
```

---

## Nota sobre el modelo

El origen formal de los 350 ejemplos actuales es **GPT-5.4-mini + validación
humana de Nico**. Cualquier mención histórica a `gpt-5.5` en artefactos de
requests/scripts no debe usarse como baseline ni como origen del training set.
Ver `SYSTEM_PROMPT_GPT5_USADO.md` §Pie para el detalle.
