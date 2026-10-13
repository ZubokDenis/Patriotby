import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

from bot import TelegramMiniAppBot

APP_DIR = Path("/content/Patriot_MiniApp") if Path("/content/Patriot_MiniApp").exists() else Path(__file__).resolve().parent
LOG_LHR = APP_DIR / "lhr8000.log"
URL_FILE = APP_DIR / "public_url.txt"
ENV_FILE = APP_DIR / ".env_keys.json"

_uvicorn_proc = None
_tunnel_proc = None


def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def get_env_with_keys() -> dict:
    env = os.environ.copy()
    if ENV_FILE.exists():
        try:
            saved = json.loads(ENV_FILE.read_text(encoding="utf-8"))
            if saved.get("BOT_TOKEN"):
                env["BOT_TOKEN"] = saved["BOT_TOKEN"]
            if saved.get("GEMINI_API_KEY"):
                env["GEMINI_API_KEY"] = saved["GEMINI_API_KEY"]
        except Exception:
            pass
    if URL_FILE.exists():
        env["PUBLIC_WEBAPP_URL"] = URL_FILE.read_text(encoding="utf-8").strip()
    return env


def ensure_ssh_key(force_rotate: bool = False) -> None:
    ssh_dir = Path.home() / ".ssh"
    ssh_key = ssh_dir / "id_ed25519"
    ssh_pub = ssh_dir / "id_ed25519.pub"
    if force_rotate:
        ssh_key.unlink(missing_ok=True)
        ssh_pub.unlink(missing_ok=True)
    if not ssh_key.exists():
        ssh_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(ssh_key)],
            check=False,
        )


def ensure_uvicorn_running() -> None:
    global _uvicorn_proc
    if _uvicorn_proc is not None and _uvicorn_proc.poll() is None:
        return
    if is_port_open(8000):
        return
    print("[Watchdog] Starting uvicorn on port 8000...", flush=True)
    env = get_env_with_keys()
    _uvicorn_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(APP_DIR),
        env=env,
        stdout=open(str(APP_DIR / "uvicorn.log"), "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    time.sleep(2.0)


def is_ssh_tunnel_running() -> bool:
    global _tunnel_proc
    if _tunnel_proc is not None and _tunnel_proc.poll() is None:
        return True
    return False


def is_tunnel_throttled_or_dead() -> bool:
    if not LOG_LHR.exists():
        return False
    try:
        txt = LOG_LHR.read_text(encoding="utf-8", errors="ignore")
        lower = txt.lower()
        if "your connection has been slowed" in lower or "slowed by" in lower:
            return True
        if "connection closed by" in lower or "remote port forwarding failed" in lower:
            return True
    except Exception:
        pass
    return False


def pick_reachable_lhr_host() -> str:
    candidates = ["35.171.254.69"]
    try:
        resolved = list({ai[4][0] for ai in socket.getaddrinfo("localhost.run", 22, socket.AF_INET, socket.SOCK_STREAM)})
        for ip in resolved:
            if ip not in candidates:
                candidates.append(ip)
    except Exception:
        pass
    for ip in candidates:
        try:
            with socket.create_connection((ip, 22), timeout=2.5) as s:
                s.settimeout(2.5)
                banner = s.recv(32)
                if b"SSH-2.0" in banner:
                    return ip
        except Exception:
            continue
    return "localhost.run"


def start_ssh_tunnel(rotate_key: bool = True) -> None:
    global _tunnel_proc
    if _tunnel_proc is not None:
        try:
            _tunnel_proc.terminate()
            _tunnel_proc.wait(timeout=2.0)
        except Exception:
            try:
                _tunnel_proc.kill()
            except Exception:
                pass
    ensure_ssh_key(force_rotate=rotate_key)
    log_f = open(str(LOG_LHR), "w", encoding="utf-8")
    host = pick_reachable_lhr_host()
    print(f"[Watchdog] Starting localhost.run SSH tunnel via {host}...", flush=True)
    _tunnel_proc = subprocess.Popen(
        [
            "ssh",
            "-4",
            "-n",
            "-T",
            "-o",
            "ConnectTimeout=6",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ServerAliveInterval=20",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "ExitOnForwardFailure=yes",
            "-R",
            "80:127.0.0.1:8000",
            f"nokey@{host}",
        ],
        cwd=str(APP_DIR),
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )


def get_latest_lhr_url() -> str:
    if not LOG_LHR.exists():
        return ""
    try:
        txt = LOG_LHR.read_text(encoding="utf-8", errors="ignore")
        matches = re.findall(r"(https://[a-zA-Z0-9-]+\.lhr\.life)", txt)
        return matches[-1] if matches else ""
    except Exception:
        return ""


def run_watchdog_loop() -> None:
    os.chdir(str(APP_DIR))
    URL_FILE.unlink(missing_ok=True)
    last_synced_url = ""
    tunnel_started_at = 0.0

    while True:
        try:
            ensure_uvicorn_running()

            if not is_ssh_tunnel_running() or is_tunnel_throttled_or_dead():
                if is_tunnel_throttled_or_dead():
                    print("[Watchdog] Tunnel throttled/closed by localhost.run, rotating SSH key & reconnecting...", flush=True)
                start_ssh_tunnel(rotate_key=True)
                tunnel_started_at = time.time()
                time.sleep(3.5)

            current_url = get_latest_lhr_url()
            if not current_url and (time.time() - tunnel_started_at) > 12.0:
                print("[Watchdog] SSH tunnel did not yield URL within 12s, retrying...", flush=True)
                start_ssh_tunnel(rotate_key=True)
                tunnel_started_at = time.time()
                time.sleep(3.5)
                current_url = get_latest_lhr_url()

            if current_url and current_url != last_synced_url:
                print(f"[Watchdog] Detected new/rotated tunnel URL: {current_url}", flush=True)
                URL_FILE.write_text(current_url, encoding="utf-8")
                env = get_env_with_keys()
                token = env.get("BOT_TOKEN", "").strip()
                if token:
                    bot = TelegramMiniAppBot(token=token, webapp_url=current_url)
                    bot.sync_url_everywhere()
                last_synced_url = current_url
        except Exception as e:
            print(f"[Watchdog] Loop warning: {e}", flush=True)

        time.sleep(2.5)


if __name__ == "__main__":
    run_watchdog_loop()
