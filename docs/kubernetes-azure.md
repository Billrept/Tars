# Kubernetes and Azure Guide for Tars

This guide turns Tars into a concrete Kubernetes/Azure talking point for a Software Engineer Intern interview.

## What Was Added

Tars now has a small Kubernetes deployment layer:

- `k8s/00-namespace.yaml` creates the `tars` namespace.
- `k8s/10-configmaps.yaml` stores non-secret runtime settings.
- `k8s/20-redis.yaml` runs Redis for ARQ jobs and WebSocket progress pub/sub.
- `k8s/30-api.yaml` runs the FastAPI service.
- `k8s/40-worker.yaml` runs the ARQ background worker.
- `k8s/50-frontend.yaml` runs the static frontend through nginx.
- `frontend/Dockerfile` packages the static frontend as a container image.
- `frontend/config.js` lets Kubernetes override the frontend API/WebSocket URLs without rebuilding the image.

## Mental Model

The Docker Compose architecture maps directly to Kubernetes:

| Compose service | Kubernetes object | Why it exists |
|---|---|---|
| `redis` | `Deployment/redis` + `Service/redis` | Queue backend and pub/sub bus |
| `api` | `Deployment/tars-api` + `Service/tars-api` | FastAPI HTTP and WebSocket API |
| `worker` | `Deployment/tars-worker` | Runs optimization jobs outside request handling |
| `frontend` | `Deployment/tars-frontend` + `Service/tars-frontend` | Serves the Three.js UI |

Kubernetes concepts used here:

- `Deployment`: keeps the desired number of pods running and replaces failed pods.
- `Service`: gives pods a stable internal DNS name such as `redis` or `tars-api`.
- `ConfigMap`: injects environment variables and frontend runtime config.
- `Probe`: tells Kubernetes when a container has started, is ready for traffic, or should be restarted.
- `Namespace`: keeps all Tars resources grouped under `tars`.

## Local Kubernetes Run

Use this when learning with Docker Desktop Kubernetes, kind, or minikube.

Build the two images:

```bash
docker build -t tars-api:local .
docker build -t tars-frontend:local ./frontend
```

If you use kind:

```bash
kind load docker-image tars-api:local
kind load docker-image tars-frontend:local
```

If you use minikube:

```bash
minikube image load tars-api:local
minikube image load tars-frontend:local
```

Apply the manifests:

```bash
kubectl apply -k k8s/
kubectl -n tars rollout status deployment/redis
kubectl -n tars rollout status deployment/tars-api
kubectl -n tars rollout status deployment/tars-worker
kubectl -n tars rollout status deployment/tars-frontend
```

Expose the app locally:

```bash
kubectl -n tars port-forward svc/tars-api 8000:8000
kubectl -n tars port-forward svc/tars-frontend 3000:80
```

Then open:

- Frontend: `http://localhost:3000`
- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

Smoke test:

```bash
curl http://localhost:8000/health
kubectl -n tars get pods
kubectl -n tars logs deploy/tars-api
kubectl -n tars logs deploy/tars-worker
```

The API can take a minute or two on first boot because it warms the ephemeris cache. That is why `tars-api` uses a `startupProbe`; Kubernetes should wait for startup instead of killing the pod too early.

## Azure AKS Path

Use this as the interview-ready cloud deployment story.

Set variables:

```bash
export RESOURCE_GROUP=tars-rg
export LOCATION=southeastasia
export ACR_NAME=tars$RANDOM$RANDOM
export AKS_NAME=tars-aks
```

Create Azure resources:

```bash
az account show --query id -o tsv
az provider register --namespace Microsoft.ContainerService
az provider register --namespace Microsoft.ContainerRegistry

az group create \
  --name "$RESOURCE_GROUP" \
  --location "$LOCATION"

az acr create \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --name "$ACR_NAME" \
  --sku Basic

az aks create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$AKS_NAME" \
  --location "$LOCATION" \
  --node-count 2 \
  --generate-ssh-keys

az aks update \
  --resource-group "$RESOURCE_GROUP" \
  --name "$AKS_NAME" \
  --attach-acr "$ACR_NAME"

az aks get-credentials \
  --resource-group "$RESOURCE_GROUP" \
  --name "$AKS_NAME" \
  --overwrite-existing
```

Build images in Azure Container Registry:

```bash
az acr build \
  --registry "$ACR_NAME" \
  --image tars-api:v1 \
  --file Dockerfile .

az acr build \
  --registry "$ACR_NAME" \
  --image tars-frontend:v1 \
  --file frontend/Dockerfile frontend
```

Deploy to AKS:

```bash
kubectl apply -k k8s/

kubectl -n tars set image deployment/tars-api \
  api="$ACR_NAME.azurecr.io/tars-api:v1"

kubectl -n tars set image deployment/tars-worker \
  worker="$ACR_NAME.azurecr.io/tars-api:v1"

kubectl -n tars set image deployment/tars-frontend \
  frontend="$ACR_NAME.azurecr.io/tars-frontend:v1"

kubectl -n tars rollout status deployment/tars-api
kubectl -n tars rollout status deployment/tars-worker
kubectl -n tars rollout status deployment/tars-frontend
```

For a quick public demo, expose only the API and frontend:

```bash
kubectl -n tars patch svc tars-api \
  -p '{"spec":{"type":"LoadBalancer"}}'

kubectl -n tars patch svc tars-frontend \
  -p '{"spec":{"type":"LoadBalancer"}}'

kubectl -n tars get svc
```

After Azure assigns public IPs, update the frontend config so the browser calls the public API:

```bash
export API_IP=<tars-api-external-ip>

kubectl -n tars create configmap tars-frontend-config \
  --from-literal=config.js="window.TARS_CONFIG = { API_BASE_URL: 'http://$API_IP:8000', WS_BASE_URL: 'ws://$API_IP:8000' };" \
  --dry-run=client \
  -o yaml | kubectl apply -f -

kubectl -n tars rollout restart deployment/tars-frontend
```

For a stronger production design, use an ingress controller, TLS, and same-origin routing instead of two public load balancers.

## What To Say In An Interview

Short answer:

> I took a Docker Compose application and mapped each runtime concern into Kubernetes. The FastAPI server, ARQ worker, Redis queue, and nginx frontend are separate Deployments. Services provide stable DNS between components, ConfigMaps hold non-secret configuration, and probes protect startup and readiness. On Azure, I would build images in ACR, run them on AKS, attach ACR to AKS for image pulls, and expose the app through an ingress or load balancer.

Deeper answer:

> The API is mostly stateless, so it can scale horizontally after the ephemeris warmup behavior is handled. The worker can also scale horizontally if Redis queue load grows. Redis is currently deployed in-cluster for learning, but for production I would evaluate Azure Cache for Redis to reduce operational burden. I would keep secrets out of ConfigMaps, use managed identities where possible, add CI/CD to build and deploy images, and add metrics/logging before calling it production-grade.

## Interview Questions And Strong Answers

**What is Kubernetes?**  
Kubernetes is a container orchestration platform. You declare the desired state of your app, and the control plane keeps pods, networking, and rollout state aligned with that desired state.

**What is a Pod?**  
A pod is the smallest deployable unit in Kubernetes. It can contain one or more containers that share networking and volumes. In this repo, each Deployment creates pods for Redis, API, worker, or frontend.

**Why use a Deployment instead of a Pod directly?**  
A Deployment manages ReplicaSets and rolling updates. If a pod dies, Kubernetes creates another one. A raw Pod does not give that rollout and self-healing behavior.

**What is a Service?**  
A Service gives changing pods a stable network identity. The API connects to Redis through `redis:6379` rather than a pod IP.

**What is a ConfigMap?**  
A ConfigMap stores non-secret configuration. Tars uses one for backend environment variables and another for `frontend/config.js`.

**What should not go in a ConfigMap?**  
Secrets such as passwords, tokens, API keys, and certificates. Those should use Kubernetes Secrets or, on Azure, a managed secret path such as Azure Key Vault integration.

**Why does this project need a startup probe?**  
The FastAPI app warms ephemeris data before it is truly ready. A startup probe gives it time to initialize before liveness/readiness checks start enforcing restarts or traffic routing.

**What is the difference between liveness and readiness?**  
Liveness decides whether Kubernetes should restart a container. Readiness decides whether a pod should receive traffic.

**How would you scale this app?**  
Scale API replicas for HTTP/WebSocket load, scale worker replicas for optimization queue throughput, and watch Redis as the shared bottleneck. In production, add metrics before autoscaling.

**What is AKS?**  
Azure Kubernetes Service is Microsoft Azure's managed Kubernetes offering. Azure manages much of the control-plane operation while you manage workloads, node pools, networking, security, and cost.

**What is ACR?**  
Azure Container Registry is a private container registry. In this project, ACR stores the `tars-api` and `tars-frontend` images that AKS pulls.

**Why split API and worker?**  
The API should respond quickly and stream progress, while the worker handles expensive optimization jobs. Splitting them avoids long CPU jobs blocking request handling.

**What is one weakness of the current Kubernetes setup?**  
Redis is in-cluster and ephemeral for learning. For production, I would consider Azure Cache for Redis, persistence strategy, backups, and monitoring.

**How would CI/CD work?**  
A pipeline would run tests, build API and frontend images, push tagged images to ACR, then update AKS with the new image tags. Rollout status would be checked before marking deployment successful.

**How would you secure it?**  
Use TLS at ingress, keep Redis private, store secrets outside ConfigMaps, apply least-privilege RBAC, avoid public admin endpoints, scan images, and pin image tags.

## How To Make This More Impressive

Do these in order:

1. Add CI/CD: build images, run tests, push to ACR, deploy to AKS.
2. Add ingress with TLS and WebSocket support.
3. Replace in-cluster Redis with Azure Cache for Redis.
4. Add Prometheus/Grafana or Azure Monitor dashboards for API latency, worker job duration, Redis queue size, and pod restarts.
5. Add Horizontal Pod Autoscalers for API and worker.
6. Add Kubernetes Secrets or Azure Key Vault for any future credentials.
7. Add resource profiling so CPU/memory requests are evidence-based.

## Official References

- Kubernetes probes: https://kubernetes.io/docs/concepts/configuration/liveness-readiness-startup-probes/
- AKS infrastructure flow: https://learn.microsoft.com/en-us/azure/aks/create-aks-infrastructure
- Azure Container Registry builds: https://learn.microsoft.com/en-us/azure/container-registry/container-registry-quickstart-task-cli
- AKS ACR tutorial: https://learn.microsoft.com/en-us/azure/aks/tutorial-kubernetes-prepare-acr
