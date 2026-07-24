# mini — Guía para el Agente de IA

NO LEER NUNCA EL .ENV

## Objetivo del Proyecto

**mini** es un agente de codificación de IA ligero, modular y transparente que se ejecuta en la terminal como un REPL. Su propósito es conectar modelos de lenguaje (vía API compatible con OpenAI) con un entorno bash ejecutándose en subprocesos, permitiendo que el LLM resuelva tareas de programación de forma interactiva.

El objetivo de la IA que opera sobre este código es **escribir, mantener y mejorar mini mismo** — esto es, un agente meticuloso que entiende la arquitectura del sistema, respeta sus patrones de diseño y es capaz de extender cualquier componente sin romper contratos establecidos.

---

## Stack Tecnológico

| Capa | Tecnología |
|------|-----------|
| Lenguaje | Python >= 3.11 |
| API LLM | OpenAI Python SDK >= 1.0 |
| UI Terminal | Rich (formateo) + prompt-toolkit (input interactivo) |
| Tokenización | tiktoken (encoding `cl100k_base`) |
| CLI | Typer |
| Testing | pytest con MagicMock |
| Dependencias | uv (lockfile `uv.lock`) |
| Entry point | `mini = "mini.cli:app"` |

---

## Arquitectura y Flujo

### Diagrama de Componentes

```
CLI (cli.py) ──► Mini (core.py) ──► OpenAIModel (model.py) ──► OpenAI API
                    │                       ▲
                    │                       │
                    ├───────────────────────┼───────────────────┐
                    │                       │                   │
                    ▼                       ▼                   ▼
            LocalEnvironment        ContextWindow         CostTracker
            (environment.py)        (context.py)          (cost_tracker.py)
                    │                       │
                    ▼                       ▼
              subprocess              Display (display.py)
              bash shell         commands.py / exceptions.py
```

### Flujo de Ejecución Principal

1. **Arranque** (`cli.py:main`): Typer parsea args CLI y variables de entorno. Se inyectan `OpenAIModel` y `LocalEnvironment` en `Mini`. Se llama a `mini.start()`.

2. **REPL** (`core.py:Mini.start`): Bucle infinito que:
   - Muestra el prompt `> ` con tab-completion de comandos slash
   - Si el input empieza con `/`, delega en `CommandRegistry.handle()`
   - Si es `exit`/`quit`, rompe el bucle
   - Si es texto normal, lo añade a `self.messages` como rol `user` y llama a `_run_turn()`

3. **Turno LLM** (`core.py:Mini._run_turn`): Bucle interno de tool-use que:
   - Verifica límites (`tracker.check_limits()`)
   - Registra la llamada (`tracker.record_call()`)
   - Prepara contexto vía `context.prepare(messages)` — aquí se aplica compresión si es necesario
   - Streamtea la respuesta del modelo (`model.stream()` → `display.stream_response()`)
   - Si hay tool calls (`actions`), las ejecuta una por una (`env.execute(action)`)
   - Formatea los resultados como mensajes `tool` y los añade al historial
   - Repite hasta que el LLM responda sin tool calls

4. **Compresión de contexto** (`context.py:ContextWindow.prepare`): 3 niveles progresivos:
   - **Umbral**: La compresión se activa cuando los tokens superan el 75% de `max_context_tokens`
   - **Clear**: Trunca outputs grandes de tool (primeras 3 + últimas 3 líneas). No toca errores ni outputs pequeños.
   - **Summarize**: Envía turns antiguos al propio LLM para que genere un resumen en bullet points, inyectado como mensaje `system`.
   - **Aggressive**: Descarta el resumen, conserva solo los últimos `keep_turns` turns completos. Si aún así excede, trunca agresivamente los tool results recientes.

### Contratos y Protocolos

El diseño se basa en **Protocolos de Python** (no ABCs ni herencia), lo que permite duck-typing y testing trivial con mocks:

```python
# _types.py
class Model(Protocol):
    def query(self, messages: list[dict], **kwargs) -> dict: ...
    def stream(self, messages: list[dict], **kwargs) -> Iterator[dict]: ...
    def format_message(self, **kwargs) -> dict: ...
    def format_observation_messages(self, message: dict, outputs: list[dict], template_vars: dict | None = None) -> list[dict]: ...

class Environment(Protocol):
    def execute(self, action: dict, **kwargs) -> dict[str, Any]: ...
```

**Regla de oro**: Cualquier nuevo componente o modificación debe respetar estos protocolos. No añadas dependencias directas a implementaciones concretas dentro del core loop.

---

## Convenciones de Código

### Estilo General
- **Sin comentarios en código**: El código debe ser autoexplicativo. No añadas comentarios a menos que sea estrictamente necesario y el usuario lo pida explícitamente.
- **Type hints obligatorios**: Toda función/método debe tener anotaciones de tipo.
- **Nombres descriptivos**: Preferir nombres largos y claros sobre abreviaturas.

### Estructura de Archivos
- `mini/__init__.py`: Exporta solo la API pública (9 símbolos actualmente).
- Cada módulo tiene una única responsabilidad:
  - `core.py`: Orquestación, nunca lógica de negocio específica
  - `model.py`: Solo comunicación con API de OpenAI
  - `environment.py`: Solo ejecución de comandos
  - `context.py`: Solo compresión y conteo de tokens
  - `cost_tracker.py`: Solo contabilidad de costos
  - `display.py`: Solo I/O terminal
  - `commands.py`: Solo sistema de comandos slash
  - `exceptions.py`: Solo jerarquía de excepciones

### Manejo de Errores
- Usar la jerarquía de `exceptions.py`:
  - `InterruptFlow`: Para interrupciones controladas del flujo
  - `LimitsExceeded`: Cuando se superan límites de pasos o costo
  - `FormatError`: Cuando el LLM devuelve tool calls malformadas o desconocidas
- Capturar `InterruptFlow` en `Mini.start()` para mostrar mensajes al usuario y salir limpiamente.

### Tool Calls
- El **único tool expuesto** es `bash`, definido en `model.py:BASH_TOOL`.
- Los argumentos del tool se parsean con `json.loads()`.
- Los resultados se envuelven en XML: `<returncode>N</returncode>\n<output>...</output>`.
- Cualquier otro tool name debe lanzar `FormatError`.

### Streaming
- El método `model.stream()` es un generador que produce eventos con tipo:
  - `{"type": "reasoning", "delta": str}`
  - `{"type": "content", "delta": str}`
  - `{"type": "done", "message": dict, "actions": list, "usage": dict | None}`
- `display.stream_response()` consume este generador y retorna `(message, actions, usage)`.

---

## Pautas para el Agente de IA

### Cuando trabajes en este códigobase:

1. **Lee antes de escribir**: Siempre lee los archivos existentes y entiende el contexto antes de proponer cambios. Respeta las convenciones existentes.

2. **No rompas los protocolos**: Cualquier nuevo Model o Environment debe cumplir con los protocolos de `_types.py`. No acoples el core loop a implementaciones concretas.

3. **Consistencia en el estilo**: Sin comentarios en código, type hints en todo, nombres descriptivos. Sigue el mismo patrón que los archivos existentes.

4. **Testing**: Ejecuta `uv run pytest` después de cualquier cambio. Los tests usan `MagicMock` para simular Model y Environment — sigue ese patrón para nuevos tests.

5. **No añadas dependencias innecesarias**: El proyecto es liviano por diseño. Revisa `pyproject.toml` antes de agregar cualquier nueva dependencia. Prefiere la stdlib o extender lo existente.

6. **Context Window es sensible**: La compresión de contexto es crítica para sesiones largas. Cualquier cambio en `context.py` debe mantener la compatibilidad con el sistema de 3 niveles (clear → summarize → aggressive) y el interior mode.

7. **Mensajes con estructura `extra`**: Todos los mensajes de assistant llevan un campo `extra` con metadatos (`actions`, `timestamp`, `usage`). Este campo se limpia antes de enviar a la API (`{k: v for k, v in m.items() if k != "extra"}`). No uses `extra` para contenido semántico.

8. **Slash commands**: Añadir un nuevo comando slash requiere:
   - Crear método handler en `Mini` (ej: `_cmd_foo`)
   - Registrar en `_register_commands()` con su `Command`
   - Actualizar la lista legacy `COMMANDS` en `commands.py` (compatibilidad hacia atrás)

9. **Pricing vía env vars**: Los costos se configuran con `INPUT_PRICE`, `OUTPUT_PRICE`, `CACHED_PRICE`. Por defecto son 0. No asumas que el tracking de costos está activo.

10. **No persistas estado**: Todo está en memoria. `/clear` hace reset total (`messages`, `n_tool_calls`, `tracker`, `context`). Cualquier feature de persistencia debe ser explícitamente diseñada y justificada.

---

## Resumen para el Agente

Eres un **ingeniero de software experto** que también entiende en profundidad cómo funcionan los LLMs, los sistemas de agentes, la gestión de contexto, el streaming de tokens y la contabilidad de costos de inferencia. Tu misión es mejorar **mini** siguiendo sus principios fundacionales: modularidad, transparencia, minimalismo y testabilidad. Cada línea que escribas debe honrar estos principios.
