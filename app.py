import asyncio
import io
import os
import threading
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ai_guide import (
    chat_with_historical_resident,
    generate_locations_for_place,
    generate_smart_route,
)
from ai_restoration import (
    _imread_bgr,
    _save_webp_with_thumb,
    ensure_demo_images,
    find_and_prepare_location_photos,
    find_location_photo_gallery,
    restoration_pipeline,
)
from database import (
    add_crowdsourced_photo,
    create_location,
    get_connection,
    get_crowdsourced_photos,
    get_location_by_id,
    get_locations,
    get_user_progress,
    init_db,
    record_quest_completion,
    resolve_existing_city,
    update_location_gallery,
)
from pdf_exporter import generate_route_pdf_bytes

STATIC_DIR = Path(os.environ.get("PATRIOT_STATIC_DIR", "static"))
UPLOADS_DIR = STATIC_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Патриот: Окно в прошлое — Telegram Mini App API",
    version="1.0.0",
)

app.add_middleware(GZipMiddleware, minimum_size=500)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_static_cache_headers(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path.lower()
    if path.startswith("/static/") and path.endswith((".webp", ".jpg", ".jpeg", ".png", ".css", ".js")):
        response.headers["Cache-Control"] = "public, max-age=86400"
    return response


def _startup_background_tasks() -> None:
    try:
        restoration_pipeline.lazy_init_base()
    except Exception as e:
        print(f"[Startup Warmup] {e}", flush=True)
    try:
        ensure_demo_images(str(STATIC_DIR))
        for loc in get_locations():
            if loc.get("category") == "archival":
                continue
            existing_photos = loc.get("photos", [])
            if len(existing_photos) == 0:
                gallery = find_location_photo_gallery(
                    title=loc["title"],
                    city=loc["city"],
                    lat=loc.get("lat"),
                    lon=loc.get("lon"),
                    static_dir=str(STATIC_DIR),
                    max_photos=6,
                )
                if gallery:
                    update_location_gallery(loc["id"], gallery)
    except Exception as e:
        print(f"[Startup Gallery Sync] {e}", flush=True)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    threading.Thread(target=_startup_background_tasks, daemon=True).start()


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ExplorePlaceRequest(BaseModel):
    place_name: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None
    user_interest: str = ""


class RouteGenerateRequest(BaseModel):
    city: str = "Гомель"
    user_query: str = ""
    duration_minutes: int = 45
    transport_mode: str = "walk"
    categories: List[str] = []
    user_lat: Optional[float] = None
    user_lon: Optional[float] = None


class QuestCheckRequest(BaseModel):
    user_id: str = "demo_user"
    user_name: str = "Исследователь"
    location_id: int
    answer: str


class PersonaChatRequest(BaseModel):
    location_id: int
    message: str
    history: List[Dict[str, str]] = []


class ExportPdfRequest(BaseModel):
    route: Dict[str, Any]
    user_id: str = ""
    base_url: str = ""


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def serve_mini_app() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


def _get_public_url() -> str:
    if Path("public_url.txt").exists():
        u = Path("public_url.txt").read_text(encoding="utf-8", errors="ignore").strip()
        if u:
            return u
    return os.environ.get("PUBLIC_WEBAPP_URL", "").strip()


@app.post("/api/telegram/webhook")
async def telegram_webhook(request: Request) -> Dict[str, Any]:
    from bot import TelegramMiniAppBot

    upd = await request.json()
    bot_token = os.environ.get("BOT_TOKEN", "").strip()
    if not bot_token and Path(".env_keys.json").exists():
        import json
        bot_token = json.loads(Path(".env_keys.json").read_text(encoding="utf-8")).get("BOT_TOKEN", "").strip()
    pub_url = _get_public_url()
    if bot_token and pub_url:
        bot = TelegramMiniAppBot(token=bot_token, webapp_url=pub_url)
        bot.process_update(upd)
    return {"ok": True}


@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "gpu_device": restoration_pipeline.__class__.__module__,
        "public_url": _get_public_url(),
    }


@app.get("/api/locations")
async def api_get_locations(
    city: Optional[str] = None,
    category: Optional[str] = None,
    search: Optional[str] = None,
) -> Dict[str, Any]:
    from ai_guide import _EXPANDED_CITIES_CACHE, _min_expected_city_objects

    effective_city = city
    if not effective_city and not (search and search.strip()):
        effective_city = "Гомель"
    elif effective_city and effective_city.strip().lower() in ("all", "все"):
        effective_city = None

    resolved_city = resolve_existing_city(search) if (search and search.strip()) else None
    locs = await asyncio.to_thread(get_locations, effective_city, category, search)

    target_city_for_expand = resolved_city or (effective_city if not (search and search.strip()) else None)
    if (
        target_city_for_expand
        and target_city_for_expand.strip().lower() not in _EXPANDED_CITIES_CACHE
        and (not category or category.strip().lower() in ("all", "все"))
        and 0 < len(locs) < _min_expected_city_objects(target_city_for_expand)
    ):
        threading.Thread(
            target=generate_locations_for_place,
            args=(target_city_for_expand,),
            daemon=True,
        ).start()

    return {
        "locations": locs,
        "count": len(locs),
        "city": resolved_city or effective_city or (locs[0]["city"] if locs else "Гомель"),
        "resolved_city": resolved_city,
        "is_vector_match": bool(search and search.strip() and len(locs) > 0 and not resolved_city),
    }


@app.post("/api/locations")
async def api_create_location(request: Request) -> Dict[str, Any]:
    data = await request.json()

    def _create_sync() -> Dict[str, Any]:
        gallery = find_location_photo_gallery(
            title=data.get("title", ""),
            city=data.get("city", "Гомель"),
            lat=data.get("lat"),
            lon=data.get("lon"),
            static_dir=str(STATIC_DIR),
            max_photos=6,
        )
        if gallery:
            data["photos"] = gallery
            modern_item = next((p for p in gallery if not p.get("is_archival")), gallery[0])
            archival_item = next((p for p in gallery if p.get("is_archival")), gallery[0])
            data["photo_before_url"] = archival_item["url"]
            data["photo_after_url"] = modern_item["url"]
        return create_location(data)

    return await asyncio.to_thread(_create_sync)


@app.post("/api/locations/{location_id}/restore_gallery_photo")
async def api_restore_gallery_photo(location_id: int, request: Request) -> Any:
    try:
        body = await request.json()
    except Exception:
        body = {}
    loc = await asyncio.to_thread(get_location_by_id, location_id)
    if not loc:
        return JSONResponse({"error": "Location not found"}, status_code=404)

    photos = loc.get("photos", [])
    if not photos:
        return JSONResponse({"error": "No photos in gallery"}, status_code=400)

    photo_index = int(body.get("photo_index", -1))
    if photo_index < 0 or photo_index >= len(photos):
        arch_idx = next((i for i, p in enumerate(photos) if p.get("is_archival")), len(photos) - 1)
        photo_index = arch_idx

    target_item = photos[photo_index]
    orig_url = target_item["url"].split("?")[0]
    if orig_url.startswith(("http://", "https://")):
        from ai_restoration import _download_image_bgr, _save_webp_with_thumb

        real_dir = STATIC_DIR / "real_photos"
        real_dir.mkdir(parents=True, exist_ok=True)
        local_name = f"remote_{location_id}_{photo_index}.webp"
        input_file = real_dir / local_name
        if not input_file.exists():
            img_dl = await asyncio.to_thread(_download_image_bgr, target_item["url"])
            if img_dl is None:
                return JSONResponse({"error": "Failed to download remote source image"}, status_code=404)
            await asyncio.to_thread(_save_webp_with_thumb, img_dl, str(input_file))
        orig_url = f"/static/real_photos/{local_name}"
        target_item["url"] = orig_url
        target_item["thumb_url"] = f"/static/real_photos/remote_{location_id}_{photo_index}_thumb.webp"
    else:
        rel_path = orig_url.lstrip("/").replace("static/", "", 1)
        input_file = STATIC_DIR / rel_path
        if not input_file.exists():
            alt_jpg = input_file.with_suffix(".jpg")
            alt_webp = input_file.with_suffix(".webp")
            if alt_webp.exists():
                input_file = alt_webp
            elif alt_jpg.exists():
                input_file = alt_jpg
            else:
                return JSONResponse({"error": "Source image file not found"}, status_code=404)

    uid = f"gen_{location_id}_{photo_index}_{int(time.time())}"
    out_filename = f"{uid}.webp"
    thumb_filename = f"{uid}_thumb.webp"
    out_file = UPLOADS_DIR / out_filename

    meta = await asyncio.to_thread(
        restoration_pipeline.restore_photo,
        str(input_file),
        str(out_file),
        f"{loc['title']}, {loc['city']}",
    )
    restored_url = f"/static/uploads/{out_filename}"
    restored_thumb_url = f"/static/uploads/{thumb_filename}"
    target_item["restored_url"] = restored_url
    target_item["restored_thumb_url"] = restored_thumb_url
    photos[photo_index] = target_item
    await asyncio.to_thread(update_location_gallery, location_id, photos)

    return {
        "status": "ok",
        "location_id": location_id,
        "photo_index": photo_index,
        "original_url": orig_url,
        "restored_url": restored_url,
        "restored_thumb_url": restored_thumb_url,
        "elapsed_sec": meta["elapsed_sec"],
        "pipeline": meta["pipeline"],
        "photos": photos,
    }


@app.post("/api/locations/{location_id}/find_photo")
async def api_find_location_photo(location_id: int, request: Request) -> Any:
    loc = await asyncio.to_thread(get_location_by_id, location_id)
    if not loc:
        return JSONResponse({"error": "Location not found"}, status_code=404)

    t0 = time.time()
    if loc.get("category") == "archival" and loc.get("photos"):
        return {
            "status": "ok",
            "location_id": location_id,
            "photos": loc["photos"],
            "elapsed_sec": "0.1",
        }

    def _find_sync() -> List[Dict[str, Any]]:
        gal = find_location_photo_gallery(
            title=loc["title"],
            city=loc["city"],
            lat=loc.get("lat"),
            lon=loc.get("lon"),
            static_dir=str(STATIC_DIR),
            max_photos=7,
            force_refresh=True,
        )
        if gal:
            update_location_gallery(location_id, gal)
        return gal

    gallery = await asyncio.to_thread(_find_sync)

    return {
        "status": "ok",
        "location_id": location_id,
        "photos": gallery,
        "elapsed_sec": str(round(time.time() - t0, 2)),
    }


@app.post("/api/search_and_restore_photo")
async def api_search_and_restore_photo(request: Request) -> Any:
    body = await request.json()
    query = str(body.get("query", "")).strip()
    location_id = body.get("location_id")
    update_location = bool(body.get("update_location_photos", True))

    def _search_and_restore_sync() -> Any:
        loc = get_location_by_id(int(location_id)) if (location_id and str(location_id).isdigit()) else None
        search_title = query or (loc["title"] if loc else "Дворец Румянцевых и Паскевичей")
        search_city = "" if query else (loc["city"] if loc else "Гомель")

        archival_item = None
        if loc and not query:
            existing_arch = [p for p in (loc.get("photos") or []) if p.get("is_archival")]
            if existing_arch:
                archival_item = existing_arch[0]

        if not archival_item:
            gallery = find_location_photo_gallery(
                title=search_title,
                city=search_city,
                static_dir=str(STATIC_DIR),
                max_photos=6,
            )
            archival_item = next((p for p in gallery if p.get("is_archival")), None)
            if not archival_item and loc and loc.get("lat") and loc.get("lon"):
                from ai_restoration import _search_pastvu_archival_photos, build_standalone_archival_gallery
                pv_cands = _search_pastvu_archival_photos(float(loc["lat"]), float(loc["lon"]), limit=3)
                if pv_cands:
                    arch_gal = build_standalone_archival_gallery(pv_cands[0][0], pv_cands[0][1], f"studio_arch_{loc['id']}", str(STATIC_DIR))
                    if arch_gal:
                        archival_item = arch_gal[0]
            if not archival_item and gallery:
                archival_item = gallery[0]

        if not archival_item:
            return None

        orig_url = archival_item["url"].split("?")[0]
        if orig_url.startswith(("http://", "https://")):
            from ai_restoration import _download_image_bgr, _save_webp_with_thumb

            real_dir = STATIC_DIR / "real_photos"
            real_dir.mkdir(parents=True, exist_ok=True)
            local_name = f"studio_src_{int(time.time())}_{uuid.uuid4().hex[:4]}.webp"
            input_file = real_dir / local_name
            img_dl = _download_image_bgr(archival_item["url"])
            if img_dl is None:
                return None
            _save_webp_with_thumb(img_dl, str(input_file))
            orig_url = f"/static/real_photos/{local_name}"
        else:
            rel_path = orig_url.lstrip("/").replace("static/", "", 1)
            input_file = STATIC_DIR / rel_path
            if not input_file.exists() and input_file.with_suffix(".jpg").exists():
                input_file = input_file.with_suffix(".jpg")

        uid = f"studio_{int(time.time())}_{uuid.uuid4().hex[:5]}"
        out_filename = f"{uid}.webp"
        thumb_filename = f"{uid}_thumb.webp"
        out_file = UPLOADS_DIR / out_filename

        meta = restoration_pipeline.restore_photo(
            str(input_file),
            str(out_file),
            prompt_hint=f"{search_title} {search_city}".strip(),
        )
        restored_url = f"/static/uploads/{out_filename}"
        restored_thumb_url = f"/static/uploads/{thumb_filename}"

        if loc and update_location:
            loc_photos = loc.get("photos", [])
            loc_photos.insert(
                0,
                {
                    "url": orig_url,
                    "thumb_url": archival_item.get("thumb_url") or orig_url,
                    "restored_url": restored_url,
                    "restored_thumb_url": restored_thumb_url,
                    "is_archival": True,
                    "caption": f"Архив: {search_title}",
                },
            )
            update_location_gallery(loc["id"], loc_photos)

        return {
            "status": "ok",
            "original_photo_url": orig_url,
            "restored_photo_url": restored_url,
            "elapsed_sec": meta["elapsed_sec"],
            "pipeline": meta["pipeline"],
            "location_id": loc["id"] if loc else None,
        }

    res = await asyncio.to_thread(_search_and_restore_sync)
    if res is None:
        return JSONResponse({"error": "No photo found"}, status_code=404)
    return res


@app.post("/api/explore_place")
async def api_explore_place(req: ExplorePlaceRequest) -> Dict[str, Any]:
    geo, locs = await asyncio.to_thread(
        generate_locations_for_place,
        req.place_name,
        req.lat,
        req.lon,
        req.user_interest,
    )
    return {
        "city": geo["name"],
        "display_name": geo["display_name"],
        "center": {"lat": geo["lat"], "lon": geo["lon"]},
        "locations": locs,
    }


@app.post("/api/routes/generate")
async def api_generate_route(req: RouteGenerateRequest) -> Dict[str, Any]:
    route = await asyncio.to_thread(
        generate_smart_route,
        req.user_query,
        req.city,
        req.duration_minutes,
        req.transport_mode,
        req.categories,
        req.user_lat,
        req.user_lon,
    )
    return route


@app.post("/api/routes/export_pdf")
async def api_export_pdf(req: ExportPdfRequest) -> Response:
    def _export_sync() -> Tuple[bytes, str]:
        base_url = req.base_url or _get_public_url() or os.environ.get("PUBLIC_WEBAPP_URL", "https://t.me")
        pdf_b = generate_route_pdf_bytes(req.route, base_webapp_url=base_url)
        pdf_fname = f"Patriot_Route_{int(time.time())}_{uuid.uuid4().hex[:6]}.pdf"
        try:
            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            (UPLOADS_DIR / pdf_fname).write_bytes(pdf_b)
        except Exception as e:
            print(f"[PDF File Save] Non-fatal warning: {e}")

        bot_token = os.environ.get("BOT_TOKEN", "").strip()
        if bot_token and req.user_id and req.user_id.isdigit():
            try:
                _send_pdf_to_telegram_user(
                    bot_token=bot_token,
                    chat_id=req.user_id,
                    pdf_bytes=pdf_b,
                    city=req.route.get("city", "Маршрут"),
                )
            except Exception as e:
                print(f"[Telegram PDF Send] Non-fatal warning: {e}")
        return pdf_b, pdf_fname

    pdf_bytes, saved_fname = await asyncio.to_thread(_export_sync)
    filename = "Patriot_Route.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(pdf_bytes)),
            "X-PDF-URL": f"/static/uploads/{saved_fname}",
            "Cache-Control": "no-store",
        },
    )


@app.post("/api/quest/check")
async def api_check_quest(req: QuestCheckRequest) -> Dict[str, Any]:
    loc = await asyncio.to_thread(get_location_by_id, req.location_id)
    if not loc:
        return {"correct": False, "points": 0}

    expected = (loc.get("quest_answer") or "").strip().lower()
    given = req.answer.strip().lower()

    is_correct = False
    if not expected:
        is_correct = len(given) > 0
    elif expected in given or given in expected:
        is_correct = len(given) >= max(1, len(expected) - 2)
    elif expected == "6" and "шест" in given:
        is_correct = True
    elif expected == "3" and "тр" in given:
        is_correct = True
    elif expected == "англия" and ("великобритан" in given or "британ" in given):
        is_correct = True

    if is_correct:
        prog = await asyncio.to_thread(record_quest_completion, req.user_id, req.user_name, req.location_id)
        return {"correct": True, "points": prog["points"], "completed_quests": prog["completed_quests"]}

    prog = await asyncio.to_thread(get_user_progress, req.user_id)
    return {"correct": False, "points": prog["points"], "completed_quests": prog["completed_quests"]}


@app.get("/api/user_progress")
async def api_get_user_progress(user_id: str = "demo_user") -> Dict[str, Any]:
    return await asyncio.to_thread(get_user_progress, user_id)


@app.post("/api/persona/chat")
async def api_persona_chat(req: PersonaChatRequest) -> Dict[str, Any]:
    return await asyncio.to_thread(
        chat_with_historical_resident,
        req.location_id,
        req.message,
        req.history,
    )


@app.post("/api/restore_photo")
async def api_restore_photo(
    file: UploadFile = File(...),
    title: str = Form("Архивное фото"),
    location_id: Optional[str] = Form(None),
    user_id: str = Form("anonymous"),
    user_name: str = Form("Исследователь"),
    update_location_photos: str = Form("false"),
) -> Dict[str, Any]:
    ext = Path(file.filename or "photo.jpg").suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        ext = ".jpg"

    uid = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
    raw_path = UPLOADS_DIR / f"raw_{uid}{ext}"
    orig_filename = f"orig_{uid}.webp"
    rest_filename = f"restored_{uid}.webp"

    orig_path = UPLOADS_DIR / orig_filename
    rest_path = UPLOADS_DIR / rest_filename

    content = await file.read()

    def _restore_upload_sync() -> Dict[str, Any]:
        nonlocal orig_filename, orig_path
        raw_path.write_bytes(content)
        raw_bgr = _imread_bgr(str(raw_path))
        if raw_bgr is not None:
            _save_webp_with_thumb(raw_bgr, str(orig_path), max_width=960, quality=82, thumb_width=200, thumb_quality=70)
            raw_path.unlink(missing_ok=True)
        else:
            orig_filename = f"orig_{uid}{ext}"
            orig_path = raw_path

        meta = restoration_pipeline.restore_photo(str(orig_path), str(rest_path))
        orig_url = f"/static/uploads/{orig_filename}"
        rest_url = f"/static/uploads/{rest_filename}"

        loc_id_int = int(location_id) if (location_id and location_id.isdigit()) else None
        saved_record = add_crowdsourced_photo(
            {
                "location_id": loc_id_int,
                "title": title,
                "user_id": user_id,
                "user_name": user_name,
                "original_photo_url": orig_url,
                "restored_photo_url": rest_url,
                "update_location_photos": update_location_photos.lower() == "true",
            }
        )
        return {
            "status": "ok",
            "original_photo_url": orig_url,
            "restored_photo_url": rest_url,
            "elapsed_sec": meta["elapsed_sec"],
            "device": meta["device"],
            "pipeline": meta["pipeline"],
            "record": saved_record,
        }

    return await asyncio.to_thread(_restore_upload_sync)


@app.get("/api/crowdsourced_photos")
async def api_get_crowdsourced_photos(location_id: Optional[int] = None) -> Dict[str, Any]:
    return {"photos": await asyncio.to_thread(get_crowdsourced_photos, location_id)}


def _send_pdf_to_telegram_user(bot_token: str, chat_id: str, pdf_bytes: bytes, city: str) -> None:
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"

    body = io.BytesIO()
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="chat_id"\r\n\r\n')
    body.write(f"{chat_id}\r\n".encode())

    caption = f"Туристический PDF-буклет с QR-кодами по маршруту ({city})"
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="caption"\r\n\r\n')
    body.write(f"{caption}\r\n".encode("utf-8"))

    body.write(f"--{boundary}\r\n".encode())
    body.write(
        f'Content-Disposition: form-data; name="document"; filename="Patriot_{city}.pdf"\r\n'.encode("utf-8")
    )
    body.write(b"Content-Type: application/pdf\r\n\r\n")
    body.write(pdf_bytes)
    body.write(f"\r\n--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        url,
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=15)


@app.post("/api/restore_gpu")
async def restore_gpu_endpoint(request: Request) -> JSONResponse:
    body = await request.json()
    if body.get("secret") != "patriot-bridge-2026":
        return JSONResponse({"error": "forbidden"}, status_code=403)

    img_b64 = body.get("image_b64", "")
    prompt = body.get("prompt") or body.get("prompt_hint") or "modern high-resolution color photograph"
    if not img_b64:
        return JSONResponse({"error": "missing image_b64"}, status_code=400)

    def _run_gpu_restore() -> Dict[str, Any]:
        import base64
        import cv2
        import numpy as np

        raw = base64.b64decode(img_b64)
        arr = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return {"error": "invalid image"}

        cleaned = restoration_pipeline._clean_sky_tears_and_defects(bgr)
        if restoration_pipeline.device == "cuda":
            ddc = restoration_pipeline.colorize_seed(cleaned)
            restored = restoration_pipeline.generate_with_controlnet(ddc, scene_description=prompt)
            pipeline_name = "Colab CUDA (Realistic Vision V6.0 + ControlNet Tile + GFPGAN v1.4 + DDColor)"
        else:
            tmp_in = UPLOADS_DIR / f"tmp_gpu_in_{uuid.uuid4().hex[:8]}.jpg"
            tmp_out = UPLOADS_DIR / f"tmp_gpu_out_{uuid.uuid4().hex[:8]}.webp"
            cv2.imwrite(str(tmp_in), cleaned)
            meta = restoration_pipeline.restore_photo(str(tmp_in), str(tmp_out), prompt_hint=prompt)
            restored = _imread_bgr(str(tmp_out))
            tmp_in.unlink(missing_ok=True)
            tmp_out.unlink(missing_ok=True)
            pipeline_name = meta.get("pipeline", "Remote Fallback")

        if restored is None:
            return {"error": "restoration failed"}

        ok, enc = cv2.imencode(".webp", restored, [cv2.IMWRITE_WEBP_QUALITY, 92])
        if not ok:
            return {"error": "encode failed"}
        b64_out = base64.b64encode(enc.tobytes()).decode("ascii")
        return {
            "status": "ok",
            "device": restoration_pipeline.device,
            "pipeline": pipeline_name,
            "image_b64": b64_out,
            "restored_b64": b64_out,
        }

    res = await asyncio.to_thread(_run_gpu_restore)
    return JSONResponse(res)

