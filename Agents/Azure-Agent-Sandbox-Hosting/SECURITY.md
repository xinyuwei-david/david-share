# Security

- The tools sign in with Azure CLI (`az login`) and never store a secret. `.env` holds non-secret deployment settings (names, IDs, endpoints and CIDRs) and is git-ignored; so are `aca-state.json`, kubeconfigs and `evidence-local/`.
- The sample sandbox port in path A is opened with anonymous auth and a deny-by-default IP rule; set `ACA_ALLOW_CIDR` to your own address, or remove the port after testing. Path C's workshop LoadBalancer is public unless you restrict `loadBalancerSourceRanges`.
- Evidence under `evidence/runs/` was exported with subscription, tenant, node, pod and endpoint identifiers removed; `tools/validate_docs.py` fails the build if a GUID, IPv4 address, email, token or internal name appears in the delivered files.
- BYO-AKS is a private preview. This repository publishes its deployment steps and our measurements, but not the product group's sample; the sample and preview enablement come from the product group (see [NOTICE.md](NOTICE.md)).

Report a problem to the repository owner privately rather than in a public issue.
