# Ejemplos Codebook corregido

Este archivo convierte el documento original a Markdown y corrige errores evidentes de **forma**, consistencia y algunos problemas de **contenido** detectables a partir del propio texto y del codebook previo [file:3][file:1]. Se conservaron los ejemplos y su lógica general, pero se normalizaron fechas, mayúsculas, acentos, nombres de categorías y varias inconsistencias terminológicas [file:3][file:1].

## Criterios de corrección

- Se normalizó el formato de fechas a `DD/MM/AAAA` cuando era deducible desde la fecha de publicación y las referencias temporales del texto [file:3][file:1].
- Se corrigieron errores ortográficos y tipográficos evidentes, por ejemplo `zona zur` por `zona sur`, `via publica` por `vía pública`, y variaciones como `estado/gobierno` por `Estado/Gobierno` [file:3].
- Se unificó el etiquetado de categorías según el codebook, por ejemplo `Manifestación de baja intensidad`, `Reunión entre las partes litigantes`, `Instituciones públicas` y `Vía pública` [file:1][file:3].
- Cuando una clasificación del ejemplo parecía claramente inconsistente con la definición del codebook, se corrigió de forma conservadora y se dejó asentado en una nota breve [file:1][file:3].

---

## Nota 1

**Fecha de publicación:** 12/04/1998  
**Título:** *A la espera de una decisión política*  
**Bajada:** *Chapadmalal contra el basurero en el sur* [file:3]

Los vecinos del barrio Chapadmalal se manifestaron una vez más contra la instalación de un basurero en la zona sur. La protesta consistió en recaudar firmas y repartir volantes entre los automovilistas. El sitio elegido fue el predio que ocupaba el peaje en la ruta 11, en el sector de La Paloma [file:3].

Los habitantes de Chapadmalal instalaron una carpa en el centro de la ruta 11 y desde allí repartieron volantes y juntaron firmas entre los marplatenses y visitantes que circulaban por el paseo [file:3].

### Registro corregido

- **Acción principal:** Corte parcial de ruta con instalación de carpa.
- **Categoría de acción:** Corte.
- **Acciones complementarias no independientes:** volanteada y juntada de firmas [file:1][file:3].
- **Sujeto:** vecinos del barrio Chapadmalal.
- **Categoría sujeto:** Vecinos.
- **Organización:** Comisión de Defensa y Preservación de Chapadmalal.
- **Categoría tipo de organización:** Vecinal.
- **Demanda:** evitar la instalación de un basurero en la zona sur.
- **Categoría demanda:** Ambiental.
- **Contra quién / dirigido a:** Ejecutivo municipal.
- **Categoría:** Estado/Gobierno.
- **Lugar:** ruta 11, sector La Paloma.
- **Categoría lugar:** Vía pública.
- **Cuándo:** 11/04/1998.
- **Represión:** false.
- **Descripción_represión:** null.
- **Enfrentamiento:** true.
- **Descripción_enfrentamiento:** "Algunos automovilistas impacientes protestaron".
- **Cantidad:** false.
- **Descripción_cantidad:** null.
- **Individuos_nombrados:** true.
- **Lista_individuos_nombrados:** Oscar Alonso; Alberto Vesubio; Gerardo Alanís.
- **Voces_protagonistas:** false.
- **Citas_voces_protagonistas:** null.

**Nota de corrección:** Se eliminó la separación de la volanteada y la juntada de firmas como acciones autónomas porque el codebook indica no desagregar acciones complementarias de la acción principal [file:1][file:3].

---

## Nota 2

**Título:** *Repudian empleados una decisión judicial* [file:3]

Los empleados de la ANSES iniciaron ayer una huelga en repudio a una decisión judicial y al procesamiento de ocho empleados del organismo. Además, miembros de la “intergremial” se reunieron con el titular regional y anunciaron que durante una asamblea decidirían si continuaba la medida de fuerza [file:3].

### Registro corregido

- **Acción 1:** huelga.
- **Categoría acción:** Huelga.
- **Acción 2:** anuncio de asamblea.
- **Categoría acción:** Manifestación de baja intensidad [file:1][file:3].
- **Sujeto:** empleados de la Administración Nacional de Servicio Social local (ANSES).
- **Categoría sujeto:** Asalariados.
- **Organización:** S/D.
- **Categoría tipo de organización:** S/D.
- **Demanda:** repudio a la decisión judicial; respaldo institucional a los empleados procesados; entrevista con el máximo jefe del organismo.
- **Categoría demanda:** Laboral.
- **Contra quién / dirigido a:** ANSES.
- **Categoría:** Estado/Gobierno.
- **Lugar:** sede local de ANSES.
- **Categoría lugar:** Instituciones públicas.
- **Cuándo:** 27/04/1998.
- **Represión:** false.
- **Descripción_represión:** null.
- **Enfrentamiento:** false.
- **Descripción_enfrentamiento:** null.
- **Cantidad:** false.
- **Descripción_cantidad:** null.
- **Individuos_nombrados:** true.
- **Lista_individuos_nombrados:** Silvia Albanese de Andreo; Alfredo Pochat.
- **Voces_protagonistas:** false.
- **Citas_voces_protagonistas:** null.

---

## Nota 3

**Título:** *Huelga en la ANSES ante el sobreseimiento de Albanese* [file:3]

El texto describe varias acciones conectadas: una asamblea, una huelga, un anuncio de continuidad de la medida y la presentación de un petitorio o pliego de reclamos [file:3].

### Registro corregido

- **Acción 1:** asamblea.
- **Categoría acción:** Asamblea.
- **Acción 2:** huelga.
- **Categoría acción:** Huelga.
- **Acción 3:** anuncio de asamblea / continuidad de medida.
- **Categoría acción:** Manifestación de baja intensidad.
- **Acción 4:** petitorio.
- **Categoría acción:** Manifestación de baja intensidad [file:1][file:3].
- **Sujeto:** trabajadores de la seccional local de la ANSES.
- **Categoría sujeto:** Asalariados.
- **Organización:** Asociación de Trabajadores del Estado (ATE); Sindicato de Empleados de la Ex Caja de Subsidios Familiares para el Personal de la Industria (SECASFPI); Asociación de Personal de Organismos de Previsión Social (APOPS).
- **Categoría tipo de organización:** Sindical.
- **Demanda:** repudio al sobreseimiento de Silvia Albanese de Andreo; rechazo al procesamiento de ocho empleados; pedido de desagravio público; exigencia de respaldo institucional hasta sentencia firme.
- **Categoría demanda:** Laboral.
- **Contra quién / dirigido a:** ANSES.
- **Categoría:** Estado/Gobierno.
- **Lugar:** delegación local de ANSES.
- **Categoría lugar:** Instituciones públicas.
- **Cuándo:** 27/04/1998.
- **Represión:** false.
- **Descripción_represión:** null.
- **Enfrentamiento:** false.
- **Descripción_enfrentamiento:** null.
- **Cantidad:** true.
- **Descripción_cantidad:** "Ochenta y siete empleados".
- **Individuos_nombrados:** true.
- **Lista_individuos_nombrados:** Silvia Albanese de Andreo; Alfredo Pochat; Daniel Darío Vázquez; Daniel De Simone; Leonardo Vecchioli; José Ohmialin; Saúl Bouer; Alejandro Bramer Marcovich.
- **Voces_protagonistas:** true.
- **Citas_voces_protagonistas:** conservar las citas del documento original, excluyendo la cita atribuida a “fuentes judiciales” si se desea registrar solo voces de protagonistas [file:3][file:1].

**Nota de corrección:** Se agregó Daniel Darío Vázquez a la lista de individuos nombrados porque aparece expresamente en la nota [file:3].

---

## Nota 4

**Fecha de publicación:** 02/05/1998  
**Título:** *Cortaron Luro y Colón por reclamos sociales* [file:3]

La nota contiene al menos tres acciones delimitables temporal y espacialmente: un corte en Luro al 10500 el jueves, una concentración/corte en Alberti y 204 al día siguiente, y otro corte posterior en Colón y 202 [file:3].

### Acción 1
- **Acción:** corte de calle con quema de cubiertas.
- **Categoría acción:** Corte.
- **Sujeto:** vecinos de los barrios Florentino Ameghino, Jorge Newbery, San Jorge y San Roque.
- **Categoría sujeto:** Vecinos.
- **Organización:** S/D.
- **Categoría tipo de organización:** S/D.
- **Demanda:** obras barriales de prevención ante inundaciones; desagües; asistencia social; unidad sanitaria; materiales para reconstrucción de viviendas.
- **Categoría demanda:** Condiciones de vida.
- **Contra quién / dirigido a:** autoridades municipales, provinciales y nacionales.
- **Categoría:** Estado/Gobierno.
- **Lugar:** Luro al 10500.
- **Categoría lugar:** Vía pública.
- **Cuándo:** 30/04/1998.
- **Represión:** false.
- **Descripción_represión:** null.
- **Enfrentamiento:** true.
- **Descripción_enfrentamiento:** "residentes y comerciantes pidieron a los piqueteros el retiro de ese lugar".
- **Cantidad:** false.
- **Descripción_cantidad:** null.
- **Individuos_nombrados:** false.
- **Lista_individuos_nombrados:** null.
- **Voces_protagonistas:** true.
- **Citas_voces_protagonistas:** conservar las del documento original [file:3].

### Acción 2
- **Acción:** concentración y corte inicial del tránsito.
- **Categoría acción:** Corte.
- **Sujeto:** vecinos de los barrios Florentino Ameghino, Jorge Newbery, San Jorge y San Roque.
- **Categoría sujeto:** Vecinos.
- **Organización:** S/D.
- **Categoría tipo de organización:** S/D.
- **Demanda:** obras barriales de prevención ante inundaciones; entubamiento del arroyo El Cardalito.
- **Categoría demanda:** Condiciones de vida.
- **Contra quién / dirigido a:** autoridades municipales, provinciales y nacionales.
- **Categoría:** Estado/Gobierno.
- **Lugar:** Alberti y 204.
- **Categoría lugar:** Vía pública.
- **Cuándo:** 01/05/1998.
- **Represión:** false.
- **Descripción_represión:** null.
- **Enfrentamiento:** false.
- **Descripción_enfrentamiento:** null.
- **Cantidad:** true.
- **Descripción_cantidad:** "cerca de 40 residentes".
- **Individuos_nombrados:** false.
- **Lista_individuos_nombrados:** null.
- **Voces_protagonistas:** false.
- **Citas_voces_protagonistas:** null.

### Acción 3
- **Acción:** corte de calle con quema de cubiertas.
- **Categoría acción:** Corte.
- **Sujeto:** vecinos de los barrios Florentino Ameghino, Jorge Newbery, San Jorge y San Roque.
- **Categoría sujeto:** Vecinos.
- **Organización:** S/D.
- **Categoría tipo de organización:** S/D.
- **Demanda:** obras barriales de prevención ante inundaciones; entubamiento del arroyo El Cardalito.
- **Categoría demanda:** Condiciones de vida.
- **Contra quién / dirigido a:** autoridades municipales, provinciales y nacionales.
- **Categoría:** Estado/Gobierno.
- **Lugar:** avenida Colón y calle 202.
- **Categoría lugar:** Vía pública.
- **Cuándo:** 01/05/1998.
- **Represión:** false.
- **Descripción_represión:** null.
- **Enfrentamiento:** false.
- **Descripción_enfrentamiento:** null.
- **Cantidad:** true.
- **Descripción_cantidad:** "cerca de 40 residentes".
- **Individuos_nombrados:** false.
- **Lista_individuos_nombrados:** null.
- **Voces_protagonistas:** false.
- **Citas_voces_protagonistas:** null.

**Nota de corrección:** En las acciones 2 y 3 se eliminó la categoría doble `corte; manifestación` porque, según el codebook, si la acción interrumpe el tránsito debe clasificarse como `Corte` y no simultáneamente como `Manifestación` [file:1][file:3].

---

## Nota 5

**Fecha de publicación:** 13/08/1997  
**Título:** *Anuncian un paro los docentes de la UNMdP* [file:3]

La nota describe una convocatoria a paro y una asamblea previa realizada en el aula magna [file:3].

### Registro corregido

#### Acción 1
- **Acción:** convocatoria a paro.
- **Categoría acción:** Manifestación de baja intensidad.
- **Sujeto:** docentes universitarios de Mar del Plata.
- **Categoría sujeto:** Asalariados [file:1][file:3].
- **Organización:** Asociación Docentes Universitarios Marplatenses (ADUM).
- **Categoría tipo de organización:** Sindical.
- **Demanda:** recomposición salarial y mayor presupuesto para educación.
- **Categoría demanda:** Salarial; Política [file:1][file:3].
- **Contra quién / dirigido a:** Gobierno nacional.
- **Categoría:** Estado/Gobierno.
- **Lugar:** aula magna de la Facultad de Ciencias Económicas y Sociales de la UNMdP.
- **Categoría lugar:** Instituciones públicas.
- **Cuándo:** 13/08/1997.
- **Represión:** false.
- **Descripción_represión:** null.
- **Enfrentamiento:** false.
- **Descripción_enfrentamiento:** null.
- **Cantidad:** false.
- **Descripción_cantidad:** null.
- **Individuos_nombrados:** true.
- **Lista_individuos_nombrados:** Pedro Sanllorenti.
- **Voces_protagonistas:** false.
- **Citas_voces_protagonistas:** null.

#### Acción 2
- **Acción:** asamblea.
- **Categoría acción:** Asamblea.
- **Sujeto:** docentes universitarios de Mar del Plata.
- **Categoría sujeto:** Asalariados.
- **Organización:** ADUM.
- **Categoría tipo de organización:** Sindical.
- **Demanda:** recomposición salarial y mayor presupuesto para educación.
- **Categoría demanda:** Salarial; Política.
- **Contra quién / dirigido a:** Gobierno nacional.
- **Categoría:** Estado/Gobierno.
- **Lugar:** aula magna de la Facultad de Ciencias Económicas y Sociales de la UNMdP.
- **Categoría lugar:** Instituciones públicas.
- **Cuándo:** 13/08/1997.
- **Represión:** false.
- **Descripción_represión:** null.
- **Enfrentamiento:** false.
- **Descripción_enfrentamiento:** null.
- **Cantidad:** true.
- **Descripción_cantidad:** "cerca de doscientos docentes".
- **Individuos_nombrados:** true.
- **Lista_individuos_nombrados:** Pedro Sanllorenti.
- **Voces_protagonistas:** false.
- **Citas_voces_protagonistas:** null.

**Nota de corrección:** El sujeto no es `dirigente ADUM / Militantes`, sino docentes universitarios en tanto trabajadores asalariados, de acuerdo con la definición de `Asalariados` del codebook [file:1][file:3].

---

## Nota 6

**Título:** *Paran hoy gremios combativos con incidencia en los servicios* [file:3]

La nota es compleja y mezcla convocatoria, adhesiones sectoriales y anuncio de movilización. Para evitar sobreextensión, se corrigen aquí los dos registros propuestos, manteniendo su estructura general [file:3].

### Registro corregido

#### Acción 1
- **Acción:** anuncio de paro.
- **Categoría acción:** Manifestación de baja intensidad.
- **Sujeto:** gremios y organizaciones convocantes al paro.
- **Categoría sujeto:** Residual [file:1][file:3].
- **Organización:** CTA; MTA; CCC; UOM; y gremios adherentes mencionados en la nota.
- **Categoría tipo de organización:** Sindical; Territorial [file:1][file:3].
- **Demanda:** contra la flexibilización laboral y contra el modelo económico.
- **Categoría demanda:** Política.
- **Contra quién / dirigido a:** Gobierno nacional y CGT nacionales [file:3].
- **Categoría:** Estado/Gobierno; Sindicatos [file:1][file:3].
- **Lugar:** S/D.
- **Categoría lugar:** S/D.
- **Cuándo:** 13/08/1997.
- **Represión:** false.
- **Descripción_represión:** null.
- **Enfrentamiento:** false.
- **Descripción_enfrentamiento:** null.
- **Cantidad:** false.
- **Descripción_cantidad:** null.
- **Individuos_nombrados:** true.
- **Lista_individuos_nombrados:** Daniel Domínguez.
- **Voces_protagonistas:** true.
- **Citas_voces_protagonistas:** "estará garantizada la circulación de un 10% de unidades"; "porque así lo impone la legislación vigente".

#### Acción 2
- **Acción:** anuncio de movilización.
- **Categoría acción:** Manifestación de baja intensidad.
- **Sujeto:** gremios y organizaciones convocantes al paro.
- **Categoría sujeto:** Residual.
- **Organización:** CTA; MTA; CCC; UOM; y gremios adherentes mencionados en la nota.
- **Categoría tipo de organización:** Sindical; Territorial.
- **Demanda:** contra la flexibilización laboral y contra el modelo económico.
- **Categoría demanda:** Política.
- **Contra quién / dirigido a:** Gobierno nacional y CGT nacionales.
- **Categoría:** Estado/Gobierno; Sindicatos.
- **Lugar:** sede de UTA como punto de reunión inicial.
- **Categoría lugar:** Sede sindical [file:3][file:1].
- **Cuándo:** 13/08/1997.
- **Represión:** false.
- **Descripción_represión:** null.
- **Enfrentamiento:** false.
- **Descripción_enfrentamiento:** null.
- **Cantidad:** false.
- **Descripción_cantidad:** null.
- **Individuos_nombrados:** true.
- **Lista_individuos_nombrados:** Daniel Domínguez.
- **Voces_protagonistas:** true.
- **Citas_voces_protagonistas:** "estará garantizada la circulación de un 10% de unidades"; "porque así lo impone la legislación vigente".

**Nota de corrección:** La clasificación original mezclaba múltiples tipos de sujeto incompatibles en un único campo. Se normalizó a una solución conservadora (`Residual`) porque el ejemplo reúne sindicatos, jubilados, actores multisectoriales y agrupaciones políticas en una misma acción [file:1][file:3].

---

## Nota 7

**Título:** *Vecinos de Cerrito Sur elevan severas quejas* [file:3]

### Registro corregido

- **Acción:** reunión de vecinos para hacer público el reclamo.
- **Categoría acción:** Asamblea [file:1][file:3].
- **Sujeto:** vecinos del barrio Cerrito Sur.
- **Categoría sujeto:** Vecinos.
- **Organización:** Sociedad de Fomento Cerrito Sur.
- **Categoría tipo de organización:** Vecinal.
- **Demanda 1:** asfaltado, alumbrado, forestación, desagües e infraestructura barrial.
- **Categoría demanda 1:** Condiciones de vida.
- **Demanda 2:** seguridad.
- **Categoría demanda 2:** Seguridad.
- **Demanda 3:** empleo / inicio del Plan Barrios.
- **Categoría demanda 3:** Económica [file:1][file:3].
- **Contra quién / dirigido a:** Gobierno municipal y Gobierno provincial.
- **Categoría:** Estado/Gobierno.
- **Lugar:** Sociedad de Fomento Cerrito Sur, Marcelo T. de Alvear al 2600.
- **Categoría lugar:** Sede vecinal / no tipificada en el codebook de lugar [file:3][file:1].
- **Cuándo:** 15/09/1997.
- **Represión:** false.
- **Descripción_represión:** null.
- **Enfrentamiento:** false.
- **Descripción_enfrentamiento:** null.
- **Cantidad:** false.
- **Descripción_cantidad:** null.
- **Individuos_nombrados:** true.
- **Lista_individuos_nombrados:** Martín Muñoz; Eva Giménez; José María Conte; José Fiscaletti.
- **Voces_protagonistas:** true.
- **Citas_voces_protagonistas:** conservar las citas del documento original [file:3].

**Nota de corrección:** Se cambió `Reunión` por `Asamblea` porque la acción reúne a una sola de las partes del conflicto, no a las partes litigantes entre sí [file:1][file:3]. También se corrigió `María Conte` por `José María Conte`, tal como figura en el cuerpo de la nota [file:3].

---

## Nota 8

**Título:** *Reclaman el cese de la veda a los buques fresqueros* [file:3]

La nota presenta tres acciones diferenciables: la olla popular ya instalada, la asamblea y la redacción de un documento/petitorio [file:3].

### Registro corregido

#### Acción 1
- **Acción:** olla popular.
- **Categoría acción:** Manifestación [file:1][file:3].
- **Sujeto:** obreros desocupados del Puerto de Mar del Plata.
- **Categoría sujeto:** Asalariados.
- **Organización:** S/D.
- **Categoría tipo de organización:** S/D.
- **Demanda 1:** reactivación del sector.
- **Categoría demanda 1:** Económica.
- **Demanda 2:** eliminación del trabajo en negro.
- **Categoría demanda 2:** Política [file:3][file:1].
- **Demanda 3:** rechazo al Plan Barrios.
- **Categoría demanda 3:** Política.
- **Demanda 4:** defensa de la riqueza ictícola y cese de la veda a los barcos fresqueros.
- **Categoría demanda 4:** Política.
- **Demanda 5:** erradicación de barcos congeladores y factoría.
- **Categoría demanda 5:** Política.
- **Contra quién / dirigido a:** Gobierno nacional.
- **Categoría:** Estado/Gobierno.
- **Lugar:** esquina de Vértiz y Edison.
- **Categoría lugar:** Vía pública.
- **Cuándo:** 22/09/1997.
- **Represión:** false.
- **Descripción_represión:** null.
- **Enfrentamiento:** false.
- **Descripción_enfrentamiento:** null.
- **Cantidad:** true.
- **Descripción_cantidad:** "unas veinte personas todos los días".
- **Individuos_nombrados:** true.
- **Lista_individuos_nombrados:** Ramón González; Oscar Lapalma; Mario Omar Lefiñir; Eduardo Duhalde.
- **Voces_protagonistas:** true.
- **Citas_voces_protagonistas:** conservar las del documento original [file:3].

#### Acción 2
- **Acción:** asamblea.
- **Categoría acción:** Asamblea.
- **Sujeto:** obreros desocupados del Puerto de Mar del Plata.
- **Categoría sujeto:** Asalariados.
- **Organización:** S/D.
- **Categoría tipo de organización:** S/D.
- **Demandas y demás campos:** iguales a la acción 1, con fecha 22/09/1997 [file:3].

#### Acción 3
- **Acción:** redacción de documento / petitorio.
- **Categoría acción:** Manifestación de baja intensidad.
- **Sujeto:** obreros desocupados del Puerto de Mar del Plata.
- **Categoría sujeto:** Asalariados.
- **Organización:** S/D.
- **Categoría tipo de organización:** S/D.
- **Demandas y demás campos:** iguales a la acción 1, con fecha 22/09/1997 [file:3].

**Nota de corrección:** La fecha `1997/06/23` del original es inconsistente con el archivo y con la referencia temporal “ayer” en una nota fechada el 23/09/1997; se corrigió a `22/09/1997` [file:3].

---

## Nota 9

**Título:** *Pescadores movilizados hacia la Municipalidad* [file:3]

La nota permite distinguir concentración, corte de calle, ocupación del hall y reunión con el intendente [file:3].

### Registro corregido

#### Acción 1
- **Acción:** concentración frente a la Municipalidad.
- **Categoría acción:** Manifestación.
- **Sujeto:** trabajadores de las lanchas amarillas de la banquina local.
- **Categoría sujeto:** Asalariados [file:1][file:3].
- **Organización:** Sociedad de Patrones Pescadores.
- **Categoría tipo de organización:** Patronal.
- **Demanda:** suspensión del decreto 701; reducción o exención de aportes previsionales; régimen especial para el sector.
- **Categoría demanda:** Económica [file:1][file:3].
- **Contra quién / dirigido a:** Gobierno nacional.
- **Categoría:** Estado/Gobierno.
- **Lugar:** frente a la Municipalidad.
- **Categoría lugar:** Vía pública.
- **Cuándo:** 27/09/1997.
- **Represión:** false.
- **Descripción_represión:** null.
- **Enfrentamiento:** false.
- **Descripción_enfrentamiento:** null.
- **Cantidad:** true.
- **Descripción_cantidad:** "cerca de 200 trabajadores".
- **Individuos_nombrados:** true.
- **Lista_individuos_nombrados:** Luciano Albano; Elio Aprile; Carlos Ruckauf; Luis Ignoto.
- **Voces_protagonistas:** true.
- **Citas_voces_protagonistas:** conservar las del documento original [file:3].

#### Acción 2
- **Acción:** corte de calle.
- **Categoría acción:** Corte.
- **Sujeto, organización, demanda, destinatario, cantidad e individuos nombrados:** iguales a la acción 1.
- **Lugar:** calle Hipólito Yrigoyen, frente a la Municipalidad.
- **Categoría lugar:** Vía pública.
- **Cuándo:** 27/09/1997.

#### Acción 3
- **Acción:** ocupación del hall de la comuna.
- **Categoría acción:** Ocupación.
- **Sujeto, organización, demanda, destinatario, cantidad e individuos nombrados:** iguales a la acción 1.
- **Lugar:** hall de la Municipalidad.
- **Categoría lugar:** Instituciones públicas.
- **Cuándo:** 27/09/1997.

#### Acción 4
- **Acción:** reunión con el intendente Aprile.
- **Categoría acción:** Reunión entre las partes litigantes.
- **Sujeto, organización, demanda, destinatario, cantidad e individuos nombrados:** iguales a la acción 1.
- **Lugar:** Municipalidad.
- **Categoría lugar:** Instituciones públicas.
- **Cuándo:** 27/09/1997.

**Nota de corrección:** La fecha original `1997/09/28` se corrigió a `27/09/1997` al tratarse de una nota del 27/09/1997 que describe hechos del mismo día, sin indicador de futuro cumplido [file:3]. También se incorporó a Luis Ignoto en la lista de personas nombradas, ya que aparece citado explícitamente [file:3].

---

## Observaciones finales

- En varios ejemplos originales aparecen mezclados `sujeto social`, `organización` y `tipo de organización`, lo que genera clasificaciones híbridas que no siguen del todo las definiciones del codebook [file:1][file:3].
- También hay casos en los que las acciones complementarias fueron separadas como eventos autónomos, algo que el codebook desaconseja explícitamente [file:1][file:3].
- Esta versión corrigió esos problemas solo cuando el texto permitía hacerlo con seguridad razonable, sin reescribir el sentido analítico general de los ejemplos [file:3][file:1].
