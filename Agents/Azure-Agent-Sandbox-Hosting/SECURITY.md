# Security

- The tools sign in with Azure CLI (`az login`) and never store a secret. `.env` holds non-secret deployment settings (names, IDs, endpoints and CIDRs) and is git-ignored; so are `aca-state.json`, kubeconfigs and `evidence-local/`.
- The sample sandbox port in path A is opened with anonymous auth and a deny-by-default IP rule; set `ACA_ALLOW_CIDR` to your own address, or remove the port after testing. Path C's workshop LoadBalancer is public unless you restrict `loadBalancerSourceRanges`.
- Evidence under `evidence/runs/` was exported with subscription, tenant, node, pod and endpoint identifiers removed; `tools/validate_docs.py` checks delivered text files for non-placeholder GUIDs, non-example IPv4 addresses, emails and credential patterns.
- This public edition contains only A/B/C. Private-preview onboarding, implementation and measurements must remain outside this public tree; the scope guard rejects unpublished measurement paths and metric namespaces.

Report a problem to the repository owner privately rather than in a public issue.
