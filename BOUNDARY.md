# Global Topology Constraints

## Process Isolation Law
UI rendering (GUI) and heavy data processing (DataProcessor) MUST NOT share memory space.

- **GUI Process**: Handles all PyQt5/tkinter rendering, user input events, matplotlib canvas updates
- **DataProcessor Process**: Handles all pandas/numpy/scipy computations, file I/O, signal processing
- These two processes run in separate memory spaces with no direct object sharing

## IPC Protocol
The ONLY legal paths for cross-process data flow are:

```python
cmd_queue: multiprocessing.Queue      # GUI → DataProcessor (commands, parameters)
telemetry_queue: multiprocessing.Queue # DataProcessor → GUI (results, status, errors)
```

### Queue Message Contracts

#### cmd_queue Messages (GUI → DataProcessor)
```python
@dataclass
class Command:
    action: str          # "process_batch", "compute_psth", "export_results", "shutdown"
    params: Dict[str, Any]
    request_id: str      # UUID for request tracking
```

#### telemetry_queue Messages (DataProcessor → GUI)
```python
@dataclass
class Telemetry:
    status: str          # "processing", "complete", "error", "progress"
    data: Optional[Dict[str, Any]]
    request_id: str      # Correlates to originating Command
    error: Optional[str]
```

## Dependency Lock
`requirements.txt` is locked after initial setup.

### Approved Dependencies
- numpy>=1.24.0
- pandas>=2.0.0
- scipy>=1.10.0
- matplotlib>=3.7.0
- PyQt5>=5.15.0 (or tkinter if preferred)

### Lock Protocol
1. Any new third-party library requires explicit human approval
2. Document the request and approval in this file under "Approved Additions"
3. Pin exact versions in requirements.txt after approval

### Approved Additions
(None yet)

## Module Boundaries

### DataProcessor (processor.py)
- Pure computation, zero UI dependencies
- Exposes methods callable only via IPC commands
- Returns serializable data only (dicts, lists, primitives, numpy arrays)

### GUI (gui.py)
- Handles all rendering and user interaction
- Never imports or instantiates DataProcessor directly
- Only communicates via queue interfaces

### Visualizer (visualizer.py)
- Stateless matplotlib figure generation
- Receives data dicts, returns figure objects
- No file I/O (export handled by DataProcessor or separate module)
