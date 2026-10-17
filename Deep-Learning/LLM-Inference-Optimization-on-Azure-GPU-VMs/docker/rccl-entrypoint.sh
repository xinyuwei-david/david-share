#!/usr/bin/env bash
set -euo pipefail

keydir=/run/rccl-ssh
port="${RCCL_SSH_PORT:-2222}"
[[ "$port" =~ ^[0-9]+$ ]] && ((port > 0 && port < 65536)) || {
  echo "RCCL_SSH_PORT must be an integer from 1 to 65535" >&2
  exit 2
}
for file in id_ed25519 authorized_keys known_hosts ssh_host_ed25519_key; do
  [ -s "$keydir/$file" ] || { echo "missing SSH input: $file; run prepare-ssh first" >&2; exit 2; }
done
mkdir -p /root/.ssh /run/sshd
chmod 700 /root/.ssh
for file in id_ed25519 authorized_keys known_hosts; do
  install -m600 "$keydir/$file" "/root/.ssh/$file"
done
install -m600 "$keydir/ssh_host_ed25519_key" /run/sshd/ssh_host_ed25519_key
printf '%s\n' \
  'Host *' '  User root' '  IdentityFile /root/.ssh/id_ed25519' \
  '  BatchMode yes' '  StrictHostKeyChecking yes' \
  '  UserKnownHostsFile /root/.ssh/known_hosts' > /root/.ssh/config
chmod 600 /root/.ssh/config
printf '%s\n' \
  "Port $port" 'PermitRootLogin prohibit-password' 'PasswordAuthentication no' \
  'KbdInteractiveAuthentication no' 'PubkeyAuthentication yes' \
  'HostKey /run/sshd/ssh_host_ed25519_key' > /run/sshd/rccl.conf
/usr/sbin/sshd -t -f /run/sshd/rccl.conf
exec /usr/sbin/sshd -D -e -f /run/sshd/rccl.conf
