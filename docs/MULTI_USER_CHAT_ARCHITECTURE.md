# Multi-user AI chat architecture

## Conclusion

Other X-AnyLabeling users can use server-hosted AI chat, including image-aware
models such as `gpt-5.6-luna`. Clients must not connect directly to the internal
CLIProxyAPI service or receive its OAuth credentials or shared upstream API
key.

## Current state

- X-AnyLabeling already has an AI chat UI and OpenAI-compatible provider.
- CLIProxyAPI listens only on server loopback `127.0.0.1:8317`.
- The verified internal API supports OpenAI-compatible model listing,
  Responses, and Chat Completions.
- The annotation API listens on `127.0.0.1:18618` and is reached through SSH
  tunnels.
- Annotation API authentication currently uses one shared service token. This
  is sufficient for a controlled single-user deployment, not a multi-user
  production deployment.

## Recommended architecture

```text
X-AnyLabeling client
  |
  | SSH tunnel or private VPN
  | per-user Bearer token
  v
Langgao API Gateway (18618)
  |-- annotation API
  |-- dataset API
  |-- staged model upload API
  `-- OpenAI-compatible chat gateway
        |
        | server-internal credential
        v
      CLIProxyAPI (127.0.0.1:8317)
        |
        v
      approved upstream models
```

## Chat gateway contract

Expose these routes from the Langgao gateway:

- `GET /v1/chat/models`
- `POST /v1/chat/completions`
- `POST /v1/chat/responses`

The gateway should:

1. Authenticate a per-user token and resolve the user identity.
2. Enforce an allowlist of chat and vision models.
3. Apply per-user concurrency, request, token, and image-size limits.
4. Remove client-supplied upstream credentials and inject the internal
   CLIProxyAPI credential server-side.
5. Record model, latency, token usage, status, and request ID without logging
   image bytes, prompts, OAuth data, or API keys.
6. Stream responses with SSE after the non-streaming path passes acceptance.

## Suggested routing

- Image understanding and annotation explanation: `gpt-5.6-luna`.
- Fast text-only help: `gpt-5.4-mini`.
- Image generation: dedicated image endpoints and approved image models.

The client may choose a model explicitly. Automatic routing should be
deterministic and policy-based, not an unrestricted model-generated decision.

## Data isolation

- Local images are uploaded only for the active request and should not be
  retained by default.
- Server images are referenced by opaque dataset IDs, never arbitrary absolute
  paths.
- Each user receives a separate cache/output namespace.
- A user may read only datasets explicitly granted to that identity.
- Model uploads enter an isolated staging directory and require administrative
  review before activation.

## Required work before multi-user release

- Replace the shared annotation token with a user/token database.
- Add role-based access for datasets and model uploads.
- Implement the OpenAI-compatible chat gateway and quotas.
- Add TLS through a private VPN or reverse proxy if SSH tunnels are not used.
- Run concurrent annotation/chat load tests and verify GPU and upstream limits.
