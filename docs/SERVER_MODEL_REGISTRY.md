# Server Model Registry

The Langgao deployment keeps model weights on the GPU server. Desktop clients
only browse the registry, request administrative installation, and use enabled
models through the existing inference API.

## User Compatibility

Existing clients do not need source changes to use a newly enabled model. After
the service restarts, the model appears through `GET /v1/models`.

The registry management UI requires a client version containing the `Server
Model Registry` action. Browsing uses the normal inference API key. Installing
or enabling a model additionally requires the model-management credential.

## Server Files

```text
/data/mfl/autolabel/models/
/data/mfl/autolabel/server/configs/langgao-model-registry.yaml
/data/mfl/autolabel/server/configs/langgao-models.yaml
/data/mfl/autolabel/server/configs/auto_labeling/<model_id>.yaml
/data/mfl/autolabel/server/app/models/
```

The registry catalog contains curated ModelScope repository IDs. A request
cannot supply an arbitrary URL or destination path.

## API

```text
GET  /v1/model-registry
POST /v1/model-registry/{model_id}/install
POST /v1/model-registry/{model_id}/enable
```

Every request uses the normal header:

```text
Token: <inference API key>
```

Administrative requests additionally use:

```text
X-Model-Upload-Token: <model-management key>
```

`install` downloads the curated snapshot from ModelScope into a temporary
directory, validates that model weights exist, and atomically moves the package
under the model root. `enable` only updates the enabled-model configuration
after confirming that the package, server implementation, and model
configuration all exist.

Enabling is intentionally not a hot-load operation. Restart the service:

```bash
systemctl --user restart langgao-autolabel.service
curl -fsS http://127.0.0.1:18618/health
```

Then verify the model appears in `GET /v1/models` and complete a real
`POST /v1/predict` request before treating it as usable.
