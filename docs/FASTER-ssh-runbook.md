# FASTER (TAMU HPRC) via Claude Science: SSH runbook

FASTER allows only password + Duo (no public key). Claude Science cannot log in by itself.
It works by reusing **your** SSH ControlMaster. Because the app forces its own socket path
(`/tmp/op-mux-<random>/%C`), you link that path to your master with `cs-faster-link`.

## One-time setup (your machine)

1. `~/.ssh/config`:
   ```
   Host faster-claude
       HostName faster.hprc.tamu.edu
       User hieult
       Port 22
       ControlMaster auto
       ControlPath /home/hieule/.ssh/cm-faster-claude
       ControlPersist 30m
       ServerAliveInterval 60
       ServerAliveCountMax 3
   ```
2. Script: save `cs-faster-link` as `~/bin/cs-faster-link`, then `chmod +x ~/bin/cs-faster-link`.
3. Claude Science → Customize → Compute → Add SSH host:
   pick `faster-claude` from ~/.ssh/config, **Public key**, Advanced empty,
   paste `faster-claude-notes.txt` into "Anything Claude Science should know?", Add.

## Every session

1. Run `cs-faster-link` in your terminal. It logs you in (password, then Duo option `1`) if needed.
2. Expected output:
   ```
   Master running (pid=NNNNN)
   OK: login2  (/tmp/op-mux-XXXXXX/<hash>)
   ```
3. Paste that output to the agent and say what to run. The agent makes one attempt;
   on an auth error it stops and shows you the exact error.

Quick health checks (your terminal):
```
ssh -O check faster-claude              # is my master alive?
ls -la /tmp/op-mux-*/                    # does the app's link exist and point to my master?
ssh -o BatchMode=yes faster-claude hostname   # does non-interactive ssh work? -> login2
```

## Failure cases

| Symptom | Likely cause | What to do |
|---|---|---|
| Agent: `ProviderError ... ssh_password_rejected` | Your master is gone (idle disconnect, laptop sleep, network change, ControlPersist expired) | Run `cs-faster-link` (re-logs in), paste output, say "go". |
| Same error, and `ssh -O check` says `Master running` | App's link missing, or the app is using a new `/tmp/op-mux-*` dir (e.g. after restarting Claude Science) | Run `ls -la /tmp/op-mux-*/`, then `cs-faster-link` to relink all dirs. Say "go". |
| `cs-faster-link`: `No /tmp/op-mux-* dir yet` | App has not tried to connect since it started | Ask the agent to make one attempt (it will fail), then re-run `cs-faster-link`, then "go". |
| `cs-faster-link` hangs or asks for password/Duo | No master; it is logging you in | Enter password, choose `1`, approve the Duo push. |
| `ssh -O check`: `No such file or directory` | No master on that socket | `ssh faster-claude` (or `cs-faster-link`). |
| `ssh -O check`: `No ControlPath specified` | Run in the wrong place (e.g. agent sandbox) or config not loaded | Run it in **your** terminal; check `ssh -G faster-claude \| grep -i control`. |
| `OK:` line missing / BatchMode `Permission denied` | Link points at a dead socket | `ssh -O exit faster-claude; cs-faster-link`. |
| Works, then fails after a while | Server dropped idle session | Keep `ServerAliveInterval 60`; re-run `cs-faster-link` when it happens. |
| Agent error is not auth (e.g. `CommandFailed`, Slurm error) | Connection is fine; command/job problem | Let the agent read the log and fix the job script. |
| Repeated failures with link + master both OK | Claude Science ssh behavior changed | Capture the app's ssh command line (below) and report to the Claude Science team. |

Capture the app's ssh command line (start this, then say "go" to the agent within 60 s):
```
for i in $(seq 120); do ps -eo pid,args | grep -E '[s]sh .*faster'; sleep 0.5; done | sort -u > ~/cs-ssh-argv.txt
```

## Rules for the agent (already in faster-claude-notes.txt)
- Never authenticate: no password/Duo, never runs `ssh faster-claude` or `ssh -O check`.
- One attempt per handshake; on auth error stop and show the exact error.
- Login node only for light commands; all compute via sbatch with `--account=142788516569`.
- GPU: `--partition=gpu --gres=gpu:a100:1` (a40 acceptable); keep cpus/mem/time conservative.

## Cleanup
- Remove the links: `rm /tmp/op-mux-*/*` (your master keeps running).
- Close your master: `ssh -O exit faster-claude`.
