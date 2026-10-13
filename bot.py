import io
import json
import os
import socket
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)


socket.getaddrinfo = _ipv4_getaddrinfo

ACTIVE_CHATS_FILE = Path("active_chats.json")


def load_active_chats() -> Dict[str, int]:
    if ACTIVE_CHATS_FILE.exists():
        try:
            return json.loads(ACTIVE_CHATS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_active_chat(chat_id: int, message_id: int) -> None:
    chats = load_active_chats()
    chats[str(chat_id)] = int(message_id)
    ACTIVE_CHATS_FILE.write_text(json.dumps(chats), encoding="utf-8")


class TelegramMiniAppBot:

    def __init__(self, token: str, webapp_url: str):
        self.token = token.strip()
        self.webapp_url = webapp_url.strip().rstrip("/")
        self.api_base = f"https://api.telegram.org/bot{self.token}"
        self.file_base = f"https://api.telegram.org/file/bot{self.token}"

    def _api_call(self, method: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.api_base}/{method}"
        data = json.dumps(payload or {}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=25) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def build_inline_keyboard(self) -> Dict[str, Any]:
        if not self.webapp_url.startswith("https://"):
            return {}
        fresh_url = f"{self.webapp_url}/?v={int(time.time())}"
        return {
            "inline_keyboard": [
                [
                    {
                        "text": "Открыть приложение",
                        "web_app": {"url": fresh_url},
                    }
                ],
            ]
        }

    def configure_menu_button(self, chat_id: Optional[int] = None) -> None:
        if not self.webapp_url.startswith("https://"):
            return
        try:
            reset_payload: Dict[str, Any] = {"menu_button": {"type": "default"}}
            if chat_id is not None:
                reset_payload["chat_id"] = int(chat_id)
            self._api_call("setChatMenuButton", reset_payload)

            payload: Dict[str, Any] = {
                "menu_button": {
                    "type": "web_app",
                    "text": "Открыть карту",
                    "web_app": {"url": f"{self.webapp_url}/?v={int(time.time())}"},
                }
            }
            if chat_id is not None:
                payload["chat_id"] = int(chat_id)
            self._api_call("setChatMenuButton", payload)
        except Exception as e:
            print(f"[Telegram Bot] MenuButton warning: {e}", flush=True)

    def sync_url_everywhere(self) -> None:
        if not self.webapp_url.startswith("https://"):
            return

        self.configure_menu_button()

        chats = load_active_chats()
        if "980497393" not in chats:
            chats["980497393"] = 0

        kb = self.build_inline_keyboard()
        for chat_id_str, last_msg_id in chats.items():
            cid = int(chat_id_str)
            self.configure_menu_button(chat_id=cid)
            if last_msg_id and kb:
                for mid in range(max(1, int(last_msg_id) - 3), int(last_msg_id) + 1):
                    try:
                        self._api_call(
                            "editMessageReplyMarkup",
                            {
                                "chat_id": cid,
                                "message_id": mid,
                                "reply_markup": kb,
                            },
                        )
                    except Exception:
                        pass
            elif cid == 980497393 and not last_msg_id:
                try:
                    self.send_welcome(cid, "Денис")
                except Exception:
                    pass

        for attempt in range(2):
            try:
                self._api_call(
                    "setWebhook",
                    {"url": f"{self.webapp_url}/api/telegram/webhook", "drop_pending_updates": False},
                )
                break
            except Exception as e:
                if attempt == 1:
                    print(f"[Watchdog] setWebhook warning: {e}", flush=True)
                time.sleep(1.0)

    def send_welcome(self, chat_id: int, first_name: str = "Пользователь") -> None:
        self.configure_menu_button(chat_id=chat_id)

        text = (
            f"Здравствуйте, <b>{first_name}</b>.\n\n"
            "<b>Патриот: Окно в прошлое</b> — интерактивная историческая платформа и экскурсовод.\n\n"
            "<b>Основные возможности:</b>\n"
            "— Интерактивная карта объектов с поддержкой любого города Беларуси по запросу или геолокации\n"
            "— Модуль «Окно в прошлое»: сравнение «До / После» и ИИ-реставрация архивных снимков\n"
            "— Персональные маршруты, контрольные вопросы на местности и диалог с историческим персонажем\n"
            "— Экспорт маршрута в PDF с QR-кодами\n\n"
            "Нажмите кнопку <b>«Открыть приложение»</b> ниже."
        )

        payload: Dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
        }
        kb = self.build_inline_keyboard()
        if kb:
            payload["reply_markup"] = kb

        res = self._api_call("sendMessage", payload)
        msg_id = res.get("result", {}).get("message_id")
        if msg_id:
            save_active_chat(chat_id, msg_id)

    def handle_photo_message(self, message: Dict[str, Any]) -> None:
        chat_id = message["chat"]["id"]
        photos = message.get("photo", [])
        if not photos:
            return

        best_photo = photos[-1]
        file_id = best_photo["file_id"]

        self._api_call(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": "Снимок получен. Выполняется реставрация и колоризация...",
            },
        )

        try:
            file_info = self._api_call("getFile", {"file_id": file_id})
            file_path = file_info["result"]["file_path"]
            download_url = f"{self.file_base}/{file_path}"

            uploads_dir = Path("static/uploads")
            uploads_dir.mkdir(parents=True, exist_ok=True)
            uid = f"{int(time.time())}_{uuid.uuid4().hex[:5]}"
            in_path = uploads_dir / f"tg_in_{uid}.jpg"
            out_path = uploads_dir / f"tg_out_{uid}.jpg"

            from ai_restoration import restoration_pipeline

            urllib.request.urlretrieve(download_url, str(in_path))
            meta = restoration_pipeline.restore_photo(str(in_path), str(out_path))

            caption = (
                f"<b>Реставрация завершена</b>\n"
                f"Модель: <code>{meta['pipeline']}</code>\n"
                f"Время обработки: {meta['elapsed_sec']} с"
            )
            self._send_photo_file(chat_id, out_path.read_bytes(), caption)
        except Exception as e:
            self._api_call(
                "sendMessage",
                {"chat_id": chat_id, "text": f"Не удалось обработать фото: {e}"},
            )

    def _send_photo_file(self, chat_id: int, image_bytes: bytes, caption: str) -> None:
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        url = f"{self.api_base}/sendPhoto"
        body = io.BytesIO()

        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="chat_id"\r\n\r\n')
        body.write(f"{chat_id}\r\n".encode())

        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="parse_mode"\r\n\r\nHTML\r\n')

        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="caption"\r\n\r\n')
        body.write(f"{caption}\r\n".encode("utf-8"))

        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="photo"; filename="restored.jpg"\r\n')
        body.write(b"Content-Type: image/jpeg\r\n\r\n")
        body.write(image_bytes)
        body.write(f"\r\n--{boundary}--\r\n".encode())

        req = urllib.request.Request(
            url,
            data=body.getvalue(),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=20)

    def process_update(self, upd: Dict[str, Any]) -> None:
        msg = upd.get("message")
        if not msg:
            return
        chat_id = msg["chat"]["id"]
        first_name = msg.get("from", {}).get("first_name", "Пользователь")
        print(f"[Telegram Bot] Incoming message from chat_id={chat_id} ({first_name})", flush=True)

        if "photo" in msg:
            self.handle_photo_message(msg)
        else:
            self.send_welcome(chat_id, first_name)
