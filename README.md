# VAP-core

VAP-core is a lightweight tool for deploying a vLLM service, running benchmark workloads, collecting profiler output, and viewing run logs through a simple web UI.

The project includes:

- `main.py`: runs the VAP workflow, including vLLM deployment, benchmark execution, profiling, and TensorBoard startup.
- `server.py`: starts a local configuration and control service.
- `public/index.html`: provides the browser UI for editing configs, validating resources, starting/stopping runs, and viewing logs.
- `example-config.json`: example configuration template.

## Setup

Create or activate the project environment, then install dependencies:

```bash
uv sync
```

Make sure Docker is available and the configured image, model path, devices, and mounts exist on the host.

## Start the UI

Run the local control server:

```bash
python server.py
```

Open the printed local URL in your browser. The UI lets you:

- edit VAP configuration values;
- validate ports, model paths, Docker image, devices, mounts, and config structure;
- start or stop a VAP run;
- view current run logs;
- open TensorBoard after it starts successfully.

## Run from CLI

You can also run VAP directly with a config file:

```bash
python main.py run --config example-config.json
```

Run outputs are written under `logs/`.

## Configuration Notes

The web UI does not overwrite the original config when starting a run. It sends the current form data to the backend, which creates a temporary config file for that run.

The deploy and benchmark `--host` / `--port` values should stay consistent. The UI keeps these fields synchronized automatically.

## Generated Files

Runtime logs and temporary config files are generated locally:

- `logs/`
- `vap-config-*.json`

These files are run artifacts and can be deleted when they are no longer needed.
