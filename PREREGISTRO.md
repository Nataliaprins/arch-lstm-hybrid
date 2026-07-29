# Pre-registro — respecificación de equivalencia ARCH/GARCH ↔ LSTM híbrido

**Fecha:** 2026-07-28
**Rama:** `fix/equivalence-respecification`
**Commit de referencia:** HEAD de esta rama en el momento de este commit (Sections 2–9.7 ya aplicadas y verificadas individualmente; ver `git log`).

Este documento se commitea **antes** de lanzar la corrida de re-estimación final (`make all` desde cero, Sección 12). Su propósito es fijar de antemano los criterios de aceptación de la hipótesis de equivalencia y la configuración exacta que se usará, para que la corrida final no pueda ajustarse post-hoc a lo que convenga al resultado. Después de este commit se ejecuta **una única corrida completa** y se reporta lo que salga, favorable o no a la hipótesis.

## 1. Objeto del estudio

El objetivo no es demostrar que el modelo propuesto (LSTM-SSE-t-Student) pronostica mejor que GARCH(1,1) en MSE/QLIKE. El objetivo es verificar si existe una **equivalencia estructural** entre la recursión GARCH(1,1)-t y una LSTM con pérdida híbrida normalizada, de forma que los hechos estilizados de los retornos (colas pesadas, agrupamiento de volatilidad) queden incorporados explícitamente en la arquitectura, no solo ajustados numéricamente.

## 2. Criterios de aceptación, fijados de antemano

Estos cuatro criterios se evalúan tal cual salgan de la corrida única. No se ajusta ninguno de ellos, ni la tolerancia asociada, después de ver los resultados.

1. **Recuperación de Proposición 2 (Sección 7, Rung 1).** El peldaño 1 de la escalera de ablación (LSTM con compuertas de entrada/olvido constantes-pero-entrenables, `o_t≡1`, `c_0` fijo, MLE Student-t puro desde un punto de partida neutral no circular) recupera α̂ y β̂ del GARCH(1,1)-t de referencia dentro de una tolerancia relativa del **10%**, para cada serie individualmente. Verdicto por serie en `logs/proposition2_check.log`.
2. **Equivalencia TOST (Sección 9.4).** El contraste TOST sobre el diferencial de pérdida QLIKE (modelo propuesto vs. GARCH(1,1), margen δ = 2% de la pérdida QLIKE media de GARCH(1,1), HAC) rechaza la hipótesis nula de no-equivalencia (|d| > δ) en al menos **3 de los 4 mercados**. Columna "TOST p" en la Tabla 9.
3. **Pertenencia al MCS.** El modelo propuesto permanece en el Model Confidence Set al 90% en **4 de 4 mercados** (columna MCS_90 en las Tablas 4–7).
4. **Pendiente de la regresión de sendas (Sección 9.3).** La pendiente *b* de la regresión σ̂²_LSTM = a + b·σ̂²_GARCH incluye 1 en su intervalo de confianza al 95% (HAC), para cada serie o, como mínimo, en la mayoría de las 4. Tabla 13.

**Nota sobre resultados intermedios ya observados con artefactos antiguos (pre-respecificación):** durante el desarrollo de las Secciones 6–9.7 se ejecutó la escalera de ablación una vez, con los modelos econométricos ya estimados (no re-estimados) y el mecanismo de Rung 1 recién construido, como parte de la verificación de que el código funciona. Ese resultado (3 de 4 mercados PASS en el criterio 1: BTC-USD, DJIA, SP500; ETH-USD FAIL en α̂ con 45% de error relativo, β̂ dentro de tolerancia) está commiteado en el historial de la Sección 7 y **no se ha vuelto a tocar ni se recalculará selectivamente** — la corrida final simplemente lo reproduce (u obtiene un resultado distinto, que se reportará igual) bajo la configuración congelada de la sección 3 de este documento. Los criterios 2–4 aún no se han evaluado con los modelos neuronales re-entrenados bajo la especificación corregida (Secciones 2–8), porque ese reentrenamiento es precisamente lo que la corrida final hace.

## 3. Configuración congelada

Los siguientes valores de `config/config.yaml` quedan fijados para la corrida final y **no se modifican** después de ver resultados parciales o finales:

| Clave | Valor | Sección |
|---|---|---|
| `loss.normalize` | `true` | 2 |
| `data.window` | `22` | 5 |
| `data.input_scaling` | `unconditional` | 4 |
| `model.init` | `garch` | 6 |
| `model.nu_mode` | `learned` | 8 |
| `tost.delta_pct` | `0.02` | 9.4 |
| `mcs.level` | `0.90` | — |
| `mcs.n_bootstrap` | `10000` | — |
| `mcs.block_size` | `20` | — |
| `unconditional_var_tolerance` | `0.20` | — |
| `var_confidence_levels` | `[0.99, 0.975]` | — |
| `hyperparameter_search.lstm_units` | `[16, 32, 64, 128]` | — |
| `hyperparameter_search.dropout` | `[0.0, 0.1, 0.2, 0.3]` | — |
| `hyperparameter_search.lambda_values` | `[0.1, 0.3, 0.5, 0.7, 0.9]` | — |
| `hyperparameter_search.batch_size` | `[32, 64, 128]` | — |
| `hyperparameter_search.max_epochs` | `1000` | — |
| `hyperparameter_search.patience` | `20` | — |
| `hyperparameter_search.n_trials` | `50` | — |
| `hyperparameter_search.nu` | `[3, 4, 5, 6, 7, 8]` — **usado únicamente si `model.nu_mode=likelihood_search`; con `nu_mode=learned` (el valor congelado arriba) esta lista no participa de ninguna búsqueda** | 8 |
| `seed` / `n_seeds` | `42` / `10` | — |
| `split.train_frac` / `split.val_frac` | `0.80` / `0.10` | — |
| `returns_scale` | `100` | — |
| Series y rangos de fecha | BTC-USD, ETH-USD, DJIA, SP500; ver `config/config.yaml` | — |

Explícitamente **no** se toca ninguno de estos valores entre el commit de este documento y la publicación de los resultados de la corrida final, incluso si un resultado intermedio (por ejemplo, un mercado que no pasa el criterio 1 o 2) sugiere que un valor distinto "arreglaría" el resultado. Un ajuste hecho en ese momento dejaría de ser una verificación y pasaría a ser un ajuste del modelo a la hipótesis — exactamente lo que este documento existe para prevenir.

## 4. Qué se ejecuta y qué se reporta

- Se ejecuta `make all` una sola vez desde cero (`make clean` primero, o equivalente: sin reutilizar artefactos de entrenamiento neuronal previos a este commit, ya que esos artefactos corresponden a especificaciones pre-Sección-2).
- Se reporta el resultado de los cuatro criterios de la Sección 2 de este documento **tal como salgan**, incluyendo si alguno falla. Un criterio que falla se deja visible en el log y en la tabla correspondiente; no se oculta, no se recalcula con una tolerancia distinta, y no se repite la corrida con otra semilla para ver si "sale mejor".
- Los hallazgos que contradigan la hipótesis de equivalencia (total o parcialmente) se documentan explícitamente en el resumen final para que el equipo humano los revise antes de redactar cualquier respuesta a evaluadores.
- Los logs de verificación técnica ya obligatorios en secciones anteriores (`logs/loss_scales.log`, `logs/garch_init.log`, `logs/proposition2_check.log`, `logs/nu_comparison.log`, `logs/degeneracy.log`, `logs/broken_models.log`) se generan de nuevo en esta corrida y reflejan el estado final, no el estado de desarrollo.

## 5. Antecedentes: hallazgos que ya obligaron a desviarse de la letra literal del brief

Documentados en detalle en los mensajes de commit de cada sección; resumidos aquí porque afectan la interpretación de los resultados finales:

- Sección 2: con las fórmulas de escala tal como se especifican, λ_efectivo a λ_nominal=0.5 sale ~0.98–0.9997 en las cuatro series (el término Student-t domina), no en el rango 0.3–0.7 mencionado como criterio de terminación en la Sección 12.2 del brief original. Se implementó la fórmula tal como se especifica, sin ajustarla para caer en ese rango.
- Sección 4: el código no tenía escalado MinMax (el diagnóstico original lo asumía); el input real de todos los modelos neuronales era ε_t crudo (no ε²_t). Se corrigió a x_t = ε²_t/σ̄²_train, un cambio de arquitectura más profundo que "quitar un MinMax".
- Sección 6: 3 de 4 series tienen GARCH(1,1) en la frontera IGARCH (α̂+β̂≈1), por lo que c_0 = ω̂/(1−α̂−β̂) es indefinido; se usa un fallback a σ̄²_train, documentado y verificado numéricamente (PASS en las cuatro series al 0.67% de error, el presupuesto exacto del factor o_t≈1).
- Sección 9.3: la numeración de tablas del brief ("Table 11" para correspondencia de compuertas) colisiona con tablas ya existentes en el repositorio antes de esta rama; por decisión explícita del usuario se preservó la numeración existente y la tabla nueva quedó como Tabla 13.

## 6. Declaración

Este documento se commitea antes de lanzar la corrida final. La corrida final se ejecuta una única vez. El resultado, sea favorable o desfavorable a la hipótesis de equivalencia estructural, se reporta sin modificar la configuración de la Sección 3 ni los criterios de la Sección 2.
