import io
import os
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent


def _get_cyrillic_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    if bold:
        candidates = [
            str(BASE_DIR / "weights" / "DejaVuSans-Bold.ttf"),
            "weights/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
            "C:/Windows/Fonts/seguisb.ttf",
        ]
    else:
        candidates = [
            str(BASE_DIR / "weights" / "DejaVuSans.ttf"),
            "weights/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ]

    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                continue

    return ImageFont.load_default()


def _wrap_text_by_pixel_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: Any,
    max_width_px: int,
    max_lines: int = 5,
) -> List[str]:
    words = (text or "").replace("\r", " ").replace("\n", " ").split()
    if not words:
        return []
    lines: List[str] = []
    curr = words[0]
    for w in words[1:]:
        candidate = f"{curr} {w}"
        try:
            width = draw.textlength(candidate, font=font)
        except Exception:
            width = len(candidate) * 11
        if width <= max_width_px:
            curr = candidate
        else:
            lines.append(curr)
            curr = w
            if len(lines) >= max_lines:
                break
    if len(lines) < max_lines and curr:
        lines.append(curr)
    elif len(lines) == max_lines and words:
        last = lines[-1]
        if not last.endswith("..."):
            lines[-1] = last.rstrip(".,;: ") + "..."
    return lines


def _make_qr_image(data_url: str, box_size: int = 190) -> Image.Image:
    try:
        import qrcode

        qr = qrcode.QRCode(version=1, box_size=6, border=2)
        qr.add_data(data_url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="#0F172A", back_color="white").convert("RGB")
        return img.resize((box_size, box_size), Image.Resampling.NEAREST)
    except Exception:
        img = Image.new("RGB", (box_size, box_size), "white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([4, 4, box_size - 5, box_size - 5], outline="#0F172A", width=3)
        for pos in [(14, 14), (box_size - 54, 14), (14, box_size - 54)]:
            draw.rectangle([pos[0], pos[1], pos[0] + 40, pos[1] + 40], outline="#0F172A", width=4)
            draw.rectangle([pos[0] + 10, pos[1] + 10, pos[0] + 30, pos[1] + 30], fill="#0F172A")
        return img


def generate_route_pdf_bytes(
    route_data: Dict[str, Any],
    base_webapp_url: str = "https://t.me",
) -> bytes:
    page_w, page_h = 1240, 1754
    pages: List[Image.Image] = []

    font_h1 = _get_cyrillic_font(36, bold=True)
    font_h2 = _get_cyrillic_font(23, bold=True)
    font_h3 = _get_cyrillic_font(20, bold=True)
    font_body = _get_cyrillic_font(18, bold=False)
    font_small = _get_cyrillic_font(15, bold=False)
    font_small_bold = _get_cyrillic_font(15, bold=True)

    city = route_data.get("city", "Гомель")
    stops: List[Dict[str, Any]] = route_data.get("stops", [])
    narrative = route_data.get("ai_narrative", "")
    total_km = route_data.get("total_distance_km", 0)
    est_min = route_data.get("estimated_minutes", 45)
    transport_label = route_data.get("transport_label", "пешком")
    districts = route_data.get("covered_districts") or []
    cats_label = route_data.get("selected_categories_label") or "Все категории наследия"

    def new_page() -> Tuple[Image.Image, ImageDraw.ImageDraw]:
        img = Image.new("RGB", (page_w, page_h), "#F8FAFC")
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, page_w, 16], fill="#1D4ED8")
        draw.line([(60, page_h - 68), (page_w - 60, page_h - 68)], fill="#CBD5E1", width=2)
        draw.text(
            (60, page_h - 48),
            f"Интерактивная платформа «Патриот: Окно в прошлое»  •  Маршрут по городу {city}",
            fill="#64748B",
            font=font_small,
        )
        return img, draw

    page, draw = new_page()
    y = 44

    draw.rounded_rectangle([60, y, page_w - 60, y + 186], radius=18, fill="#0F172A")
    draw.text((92, y + 24), "ТУРИСТИЧЕСКИЙ И ИСТОРИЧЕСКИЙ ПАСПОРТ МАРШРУТА", fill="#60A5FA", font=font_small_bold)
    draw.text((92, y + 54), f"Город: {city} ({len(stops)} точек маршрута)", fill="#FFFFFF", font=font_h1)
    dist_info = f"   •   Охвачено районов: {len(districts)}" if len(districts) > 1 else ""
    draw.text(
        (92, y + 110),
        f"Формат: {transport_label}   •   Дистанция: ~{total_km} км   •   Время: ~{est_min} мин{dist_info}",
        fill="#E2E8F0",
        font=font_body,
    )
    cat_line = f"Выбранные категории: {cats_label}"
    if len(cat_line) > 88:
        cat_line = cat_line[:85] + "..."
    draw.text((92, y + 144), cat_line, fill="#93C5FD", font=font_small)
    y += 208

    if narrative:
        intro_lines = _wrap_text_by_pixel_width(draw, narrative, font_body, max_width_px=page_w - 176, max_lines=5)
        box_h = 56 + len(intro_lines) * 26
        draw.rounded_rectangle([60, y, page_w - 60, y + box_h], radius=14, fill="#EFF6FF", outline="#BFDBFE", width=2)
        draw.text((86, y + 16), "Напутствие ИИ-экскурсовода:", fill="#1D4ED8", font=font_h3)
        ty = y + 46
        for line in intro_lines:
            draw.text((86, ty), line, fill="#1E293B", font=font_body)
            ty += 25
        y += box_h + 22

    cat_labels = {
        "religious": "Храмы и святыни",
        "architecture": "Архитектура и история",
        "historical": "Архитектура и история",
        "ww2": "Память ВОВ и мемориалы",
        "lost": "Древние и утраченные",
        "nature": "Парки и природа",
        "archival": "Архивный фотоснимок",
    }

    text_col_max_w = page_w - 430

    for idx, stop in enumerate(stops, start=1):
        card_h = 344
        if y + card_h > page_h - 90:
            pages.append(page)
            page, draw = new_page()
            y = 46

        draw.rounded_rectangle(
            [60, y, page_w - 60, y + card_h],
            radius=16,
            fill="#FFFFFF",
            outline="#CBD5E1",
            width=2,
        )

        qr_col_x = page_w - 300
        draw.line([(qr_col_x, y + 20), (qr_col_x, y + card_h - 20)], fill="#E2E8F0", width=2)

        draw.ellipse([84, y + 22, 134, y + 72], fill="#1D4ED8")
        num_str = str(idx)
        try:
            nw = draw.textlength(num_str, font=font_h2)
        except Exception:
            nw = 14 * len(num_str)
        draw.text((109 - nw / 2, y + 33), num_str, fill="#FFFFFF", font=font_h2)

        title_str = stop.get("title", "Исторический объект")
        title_lines = _wrap_text_by_pixel_width(draw, title_str, font_h2, max_width_px=text_col_max_w - 75, max_lines=2)
        ty_title = y + 20
        for tline in title_lines:
            draw.text((150, ty_title), tline, fill="#0F172A", font=font_h2)
            ty_title += 27

        cat_ru = stop.get("category_label") or cat_labels.get(stop.get("category", ""), "Исторический объект")
        district_lbl = stop.get("district_label", "")
        leg_km = stop.get("leg_distance_km", 0)
        leg_min = stop.get("leg_time_min", 0)
        leg_part = f"Переход: {leg_km} км (~{leg_min} мин)" if idx > 1 and leg_km else "Старт маршрута"
        district_part = f"{district_lbl}  •  " if district_lbl else ""
        lat_v = round(float(stop.get("lat", 52.4242)), 5)
        lon_v = round(float(stop.get("lon", 31.0143)), 5)
        meta_line = f"{district_part}{cat_ru}  •  {leg_part}  •  {lat_v}, {lon_v}"
        meta_wrapped = _wrap_text_by_pixel_width(draw, meta_line, font_small_bold, max_width_px=text_col_max_w - 75, max_lines=1)
        if meta_wrapped:
            draw.text((150, max(y + 54, ty_title + 2)), meta_wrapped[0], fill="#2563EB", font=font_small_bold)

        desc = stop.get("full_history") or stop.get("short_description", "")
        desc_y_start = max(y + 88, ty_title + 32)
        desc_lines = _wrap_text_by_pixel_width(draw, desc, font_body, max_width_px=text_col_max_w, max_lines=5)
        ty = desc_y_start
        for line in desc_lines:
            if ty > y + 220:
                break
            draw.text((86, ty), line, fill="#1E293B", font=font_body)
            ty += 24

        quest_q = stop.get("quest_question", "")
        if quest_q:
            draw.rounded_rectangle([84, y + 236, qr_col_x - 20, y + 324], radius=10, fill="#FEF3C7", outline="#F59E0B", width=2)
            draw.text((100, y + 244), "Загадка краеведческого квеста:", fill="#B45309", font=font_small_bold)
            q_lines = _wrap_text_by_pixel_width(draw, quest_q, font_small, max_width_px=(qr_col_x - 135), max_lines=2)
            qy = y + 268
            for ql in q_lines:
                draw.text((100, qy), ql, fill="#78350F", font=font_small)
                qy += 21

        loc_id = stop.get("id", idx)
        if base_webapp_url and base_webapp_url.startswith("http"):
            qr_link = f"{base_webapp_url.rstrip('/')}/?loc={loc_id}&city={urllib.parse.quote(city)}"
        else:
            qr_link = f"https://www.openstreetmap.org/?mlat={lat_v}&mlon={lon_v}#map=17/{lat_v}/{lon_v}"

        qr_img = _make_qr_image(qr_link, box_size=190)
        page.paste(qr_img, (qr_col_x + 25, y + 48))
        draw.text((qr_col_x + 30, y + 250), "Открой «Окно в прошлое»", fill="#0F172A", font=font_small_bold)
        draw.text((qr_col_x + 48, y + 272), "Наведи камеру на QR", fill="#64748B", font=font_small)

        y += card_h + 22

    pages.append(page)

    total_pages = len(pages)
    for p_idx, p_img in enumerate(pages, start=1):
        p_draw = ImageDraw.Draw(p_img)
        p_draw.text(
            (page_w - 220, page_h - 48),
            f"Страница {p_idx} из {total_pages}",
            fill="#475569",
            font=font_small_bold,
        )

    pdf_buffer = io.BytesIO()
    first_page, rest_pages = pages[0], pages[1:]
    first_page.save(
        pdf_buffer,
        format="PDF",
        resolution=150.0,
        save_all=True,
        append_images=rest_pages,
    )
    return pdf_buffer.getvalue()
