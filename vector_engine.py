import hashlib
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

EMBED_DIM = 512
SEMANTIC_DIM = 64
NGRAM_DIM = EMBED_DIM - SEMANTIC_DIM

SEMANTIC_CLUSTERS: List[Tuple[int, float, Tuple[str, ...]]] = [
    (0, 2.2, ("церков", "собор", "храм", "православ", "свято", "спасо", "покров", "успен", "никольск", "троиц", "воскресен", "борисоглеб", "царква", "сабор", "orthodox", "cathedral")),
    (1, 2.4, ("костел", "костёл", "касцёл", "католич", "фарн", "доминикан", "иезуит", "францискан", "бернардин", "воздвижен", "божьего тела", "catholic")),
    (2, 2.2, ("синагог", "иешив", "ешибот", "еврейск", "хедер", "synagogue", "yeshiva")),
    (3, 2.2, ("часовн", "каплиц", "усыпальниц", "склеп", "надгроб", "chapel")),
    (4, 2.2, ("монастыр", "обитель", "кляштар", "лавр", "кельи", "monastery", "convent")),
    (5, 2.3, ("дворец", "палац", "резиденц", "имение", "княз", "граф", "румянцев", "паскевич", "радзивилл", "сапег", "palace")),
    (6, 2.3, ("замок", "замак", "замчищ", "башн", "донжон", "ров", "бастион", "castle", "tower")),
    (7, 2.2, ("усадьб", "сядзіб", "особняк", "флигель", "охотничий домик", "дворянск", "панск", "manor", "mansion")),
    (8, 2.3, ("крепост", "крэпасц", "цитадел", "редюит", "каземат", "форт", "оборонительн", "вал", "детинец", "городищ", "fortress", "citadel")),
    (9, 2.1, ("ратуш", "магистрат", "торговые ряды", "гостиный двор", "рыночн", "площад", "town hall")),
    (10, 2.5, ("мемориал", "вечный огонь", "братская могила", "курган славы", "обелиск", "стела", "монумент", "скорбящ", "павшим", "погибшим", "жертвам", "memorial")),
    (11, 2.5, ("войн", "вов", "великая отечествен", "освобожден", "освободител", "фронт", "партизан", "подпольщ", "героев", "аллея героев", "1941", "1943", "1944", "1945", "багратион")),
    (12, 2.6, ("танк", "т-34", "ис-2", "ис-3", "самоходк", "су-100", "пушк", "артиллерийск", "катюш", "бронепоезд", "бронекатер", "самолет", "штурмовик", "бахаров")),
    (13, 2.4, ("колокол", "хатын", "ола ", "сожжен", "концлагер", "гетто", "узник", "памяти жертв")),
    (14, 2.3, ("вокзал", "железнодорож", "станци", "чыгунач", "вакзал", "паровоз", "локомотив", "депо", "перрон", "railway", "station")),
    (15, 2.1, ("мост", "переправ", "виадук", "путепровод", "паром", "bridge")),
    (16, 2.2, ("парк", "сквер", "сад", "ботаническ", "аллея", "дендрарий", "роща", "дуб", "гинкго", "park", "garden")),
    (17, 2.3, ("река", "набережн", "пруд", "озеро", "лебяжий", "затон", "пристан", "пляж", "остров", "сож", "березин", "двина", "припят", "неман", "днепр", "гомеюк", "пина", "мухавец")),
    (18, 2.2, ("музей", "экспозиц", "краеведческ", "выставк", "галере", "картинн", "museum", "gallery")),
    (19, 2.1, ("театр", "драматическ", "кукол", "филармони", "концертн", "сцена", "амфитеатр", "theater", "theatre")),
    (20, 2.0, ("библиотек", "книгопечатан", "скорин", "просветител", "гимнази", "коллегиум", "академи", "училищ")),
    (21, 2.2, ("памятник", "бюст", "скульптур", "статуя", "памятный знак", "камень", "борисов камень", "валун", "крест", "monument", "statue")),
    (22, 2.0, ("улиц", "проспект", "перекресток", "бульвар", "пешеходн", "брусчатк", "застройк", "квартал", "фасад")),
    (23, 2.0, ("гостиниц", "отель", "савой", "континентал", "доходный дом", "аптек", "почтамт", "банк", "каланч", "водонапорн")),
    (24, 2.1, ("завод", "фабрик", "мануфактур", "мельниц", "бровар", "пивоварн", "промышлен")),
    (25, 2.2, ("архив", "старинн", "историческ", "прошлое", "черно-бел", "снимок", "фотограф", "открытк", "хроник", "утрачен", "исчезнувш")),
    (26, 2.3, ("гомель", "гомел", "гомий", "homel", "gomel")),
    (27, 2.3, ("светлогорск", "светлагорск", "шатилки", "шацілкі", "svietlahorsk")),
    (28, 2.3, ("клецк", "kletsk", "kleck")),
    (29, 2.3, ("бобруйск", "бабруйск", "babruysk", "bobruisk")),
    (30, 2.3, ("баранович", "баранавіч", "baranovichi", "baranavichy")),
    (31, 2.3, ("полоцк", "полацк", "евфросини", "софийск", "polotsk", "polatsk")),
    (32, 2.3, ("мозыр", "мазыр", "mazyr", "mozyr")),
    (33, 2.3, ("минск", "мінск", "minsk")),
    (34, 2.3, ("брест", "брэст", "берестье", "brest")),
    (35, 2.3, ("гродн", "гародн", "каложа", "ложск", "grodno", "hrodna")),
    (36, 2.3, ("витебск", "віцебск", "шагал", "vitebsk")),
    (37, 2.3, ("могилев", "магілёў", "буйнич", "mogilev", "mahiliou")),
    (38, 2.3, ("пинск", "пінск", "бутримович", "полесь", "pinsk")),
    (39, 2.3, ("несвиж", "нясвіж", "фарный", "nesvizh")),
    (40, 2.3, ("мир ", "мирск", "мірск", "mir castle")),
    (41, 2.3, ("лида", "лідск", "гедимин", "lida")),
    (42, 2.3, ("новогруд", "навагруд", "мицкевич", "novogrudok")),
    (43, 2.3, ("орша", "оршанск", "orsha")),
    (44, 2.3, ("речиц", "рэчыц", "rechitsa")),
    (45, 2.3, ("жлобин", "жлобін", "красный берег", "zhlobin")),
    (46, 2.3, ("слуцк", "слуцкие пояса", "slutsk")),
    (47, 2.3, ("борисов", "барысаў", "березина 1812", "borisov")),
    (48, 2.3, ("туров", "тураў", "кирилл туровский", "turov")),
]

STOP_TOKENS = {
    "и", "в", "во", "на", "с", "со", "к", "ко", "по", "из", "за", "у", "от", "до", "о", "об", "для", "при",
    "что", "как", "где", "это", "этот", "эта", "эти", "все", "вся", "весь", "его", "ее", "их", "а", "но", "или",
    "год", "года", "году", "годы", "век", "века", "веке", "гг", "им", "имени", "беларусь", "беларуси",
    "республика", "область", "области", "район", "района", "город", "города", "городе", "вид", "общий", "фото",
}

RUSSIAN_ENDINGS = (
    "иями", "ями", "ами", "ого", "его", "ому", "ему", "ими", "ыми", "ая", "яя", "ое", "ее",
    "ые", "ие", "ой", "ей", "ий", "ый", "ую", "юю", "ам", "ям", "ах", "ях", "ом", "ем",
    "ов", "ев", "ей", "ии", "ия", "ие", "ию", "ья", "ье", "ьи", "а", "я", "о", "е", "ы", "и", "у", "ю", "ь",
)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    t = text.lower().replace("ё", "е").replace("ў", "у").replace("і", "и")
    t = re.sub(r"[^a-zа-я0-9\s-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def stem_word(w: str) -> str:
    if len(w) <= 4:
        return w
    for end in RUSSIAN_ENDINGS:
        if w.endswith(end) and len(w) - len(end) >= 4:
            return w[: -len(end)]
    return w[:6] if len(w) > 6 else w


def _hash_feature(token: str) -> Tuple[int, float]:
    h = hashlib.md5(token.encode("utf-8")).digest()
    idx = int.from_bytes(h[:4], "little") % NGRAM_DIM
    sign = 1.0 if (h[4] & 1) == 0 else -1.0
    return SEMANTIC_DIM + idx, sign


def encode_text(text: str, title_boost: str = "") -> np.ndarray:
    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    norm_full = normalize_text(f"{title_boost} {text}")
    norm_title = normalize_text(title_boost) if title_boost else ""

    if not norm_full:
        return vec

    for dim_idx, weight, keywords in SEMANTIC_CLUSTERS:
        hits = 0
        title_hits = 0
        for kw in keywords:
            if kw in norm_full:
                hits += 1
                if norm_title and kw in norm_title:
                    title_hits += 1
        if hits > 0:
            vec[dim_idx] = weight * (1.0 + 0.35 * min(hits - 1, 3) + 0.65 * min(title_hits, 2))

    def _accumulate_tokens(raw_str: str, multiplier: float) -> None:
        words = [w for w in re.split(r"[\s-]+", raw_str) if len(w) >= 2 and w not in STOP_TOKENS]
        for w in words:
            st = stem_word(w)
            idx, sign = _hash_feature(f"st:{st}")
            vec[idx] += sign * (1.9 * multiplier)

            if len(w) >= 4:
                idx4, sign4 = _hash_feature(f"p4:{w[:4]}")
                vec[idx4] += sign4 * (1.35 * multiplier)
            if len(w) >= 5:
                idx5, sign5 = _hash_feature(f"p5:{w[:5]}")
                vec[idx5] += sign5 * (1.55 * multiplier)

            if len(w) >= 4:
                padded = f"<{w}>"
                for i in range(len(padded) - 2):
                    tri = padded[i : i + 3]
                    idx_g, sign_g = _hash_feature(f"g:{tri}")
                    vec[idx_g] += sign_g * (0.45 * multiplier)

    _accumulate_tokens(norm_full, 1.0)
    if norm_title:
        _accumulate_tokens(norm_title, 3.2)

    sem_norm = float(np.linalg.norm(vec[:SEMANTIC_DIM]))
    lex_norm = float(np.linalg.norm(vec[SEMANTIC_DIM:]))
    if sem_norm > 1e-8 and lex_norm > 1e-8:
        vec[:SEMANTIC_DIM] = (vec[:SEMANTIC_DIM] / sem_norm) * 0.65
        vec[SEMANTIC_DIM:] = (vec[SEMANTIC_DIM:] / lex_norm) * 0.76
    elif norm := float(np.linalg.norm(vec)):
        if norm > 1e-8:
            vec /= norm
    return vec


def encode_location(loc: Dict[str, Any]) -> np.ndarray:
    title = str(loc.get("title", ""))
    city = str(loc.get("city", ""))
    cat = str(loc.get("category", ""))
    epoch = str(loc.get("epoch", ""))
    person = str(loc.get("historical_person", ""))
    short_d = str(loc.get("short_description", ""))
    body = f"{city} {cat} {epoch} {person} {short_d}"
    boost = f"{title} {person}"
    return encode_text(body, title_boost=boost)


def to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob: Optional[bytes]) -> Optional[np.ndarray]:
    if not blob:
        return None
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.size != EMBED_DIM:
        return None
    return arr.copy()


class SQLiteVectorIndex:

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.locations: List[Dict[str, Any]] = []
        self.matrix: np.ndarray = np.zeros((0, EMBED_DIM), dtype=np.float32)
        self.city_names: List[str] = []
        self.city_matrix: np.ndarray = np.zeros((0, EMBED_DIM), dtype=np.float32)
        self._photo_vec_cache: Dict[str, np.ndarray] = {}

    def rebuild(self, locations: List[Dict[str, Any]], embeddings: List[np.ndarray]) -> None:
        with self._lock:
            self.locations = list(locations)
            if embeddings:
                self.matrix = np.vstack(embeddings).astype(np.float32)
            else:
                self.matrix = np.zeros((0, EMBED_DIM), dtype=np.float32)

            unique_cities = sorted(list({str(l.get("city", "")).strip() for l in locations if l.get("city")}))
            self.city_names = unique_cities
            if unique_cities:
                c_vecs = [encode_text(c, title_boost=c) for c in unique_cities]
                self.city_matrix = np.vstack(c_vecs).astype(np.float32)
            else:
                self.city_matrix = np.zeros((0, EMBED_DIM), dtype=np.float32)

    def resolve_city(self, query: str) -> Optional[str]:
        q_clean = normalize_text(query)
        if not q_clean:
            return None
        q_clean = re.sub(r"^(?:г|город|в|во|около|рядом с)\s+", "", q_clean).strip()
        with self._lock:
            if not self.city_names:
                return None
            for c in self.city_names:
                c_norm = normalize_text(c)
                if q_clean == c_norm or (len(q_clean) >= 4 and c_norm.startswith(q_clean[:4]) and abs(len(q_clean) - len(c_norm)) <= 2):
                    return c
            if len(q_clean.split()) <= 2:
                q_vec = encode_text(q_clean, title_boost=q_clean)
                sims = self.city_matrix @ q_vec
                best_idx = int(np.argmax(sims))
                if float(sims[best_idx]) >= 0.58:
                    return self.city_names[best_idx]
        return None

    def search(
        self,
        query: str,
        city: str = "",
        category: str = "",
        top_k: int = 35,
    ) -> Tuple[List[Dict[str, Any]], float]:
        with self._lock:
            if not self.locations or self.matrix.shape[0] == 0:
                return [], 0.0
            locs_snap = list(self.locations)
            mat_snap = self.matrix

        q_vec = encode_text(query, title_boost=query)
        scores = (mat_snap @ q_vec).astype(np.float32)

        q_norm = normalize_text(query)
        q_stems = [stem_word(w) for w in q_norm.split() if len(w) >= 3 and w not in STOP_TOKENS]
        q_prefixes = [w[:5] for w in q_norm.split() if len(w) >= 5 and w not in STOP_TOKENS]
        city_norm = normalize_text(city) if city else ""
        cat_norm = category.strip().lower() if category and category != "all" else ""

        results: List[Tuple[float, Dict[str, Any]]] = []
        max_sim = 0.0

        for i, loc in enumerate(locs_snap):
            if cat_norm and str(loc.get("category", "")).lower() != cat_norm:
                continue
            loc_city_norm = normalize_text(str(loc.get("city", "")))
            if city_norm and loc_city_norm != city_norm:
                continue

            sim = float(scores[i])
            title_norm = normalize_text(str(loc.get("title", "")))
            desc_norm = normalize_text(str(loc.get("short_description", "")))
            person_norm = normalize_text(str(loc.get("historical_person", "")))
            combined_norm = f"{title_norm} {person_norm} {desc_norm} {loc_city_norm}"

            if q_norm and q_norm in title_norm:
                sim += 0.45
            elif q_norm and q_norm in loc_city_norm:
                sim += 0.50
            elif q_stems and all(st in combined_norm for st in q_stems):
                sim += 0.28
            elif q_prefixes and all(pf in f"{title_norm} {person_norm}" for pf in q_prefixes):
                sim += 0.30

            if sim > max_sim:
                max_sim = sim
            if sim >= 0.32:
                results.append((sim, loc))

        cutoff = max(0.32, max_sim * 0.58)
        filtered = [item for item in results if item[0] >= cutoff]
        filtered.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in filtered[:top_k]], max_sim

    def encode_photo_caption(self, caption: str) -> np.ndarray:
        key = caption.strip()[:180]
        cached = self._photo_vec_cache.get(key)
        if cached is not None:
            return cached
        vec = encode_text(key, title_boost=key)
        if len(self._photo_vec_cache) < 4000:
            self._photo_vec_cache[key] = vec
        return vec


vector_index = SQLiteVectorIndex()


def compute_archival_landmark_similarity(
    arch_caption: str,
    landmark_title: str,
    landmark_desc: str = "",
    city: str = "",
    dist_km: Optional[float] = None,
) -> float:
    if not arch_caption or not landmark_title:
        return 0.0

    clean_arch = re.sub(r"^архивный\s+(?:снимок|кадр|исторический\s+снимок)\s*(?:\([^)]*\))?\s*:?\s*", "", arch_caption.strip(), flags=re.IGNORECASE)
    arch_low = normalize_text(clean_arch)
    title_low = normalize_text(landmark_title)
    city_low = normalize_text(city)
    if city_low and len(city_low) >= 4:
        c_stem = city_low[:5]
        arch_low = " ".join(w for w in arch_low.split() if not w.startswith(c_stem))
        title_low = " ".join(w for w in title_low.split() if not w.startswith(c_stem))

    if "костел" in title_low and ("собор" in arch_low or "церков" in arch_low) and "костел" not in arch_low and "фарн" not in arch_low:
        return 0.0
    if ("собор" in title_low or "церков" in title_low or "храм" in title_low) and "костел" in arch_low and "бывш" not in title_low and "воскресен" not in title_low:
        return 0.0
    if any(w in arch_low for w in ("школа", "дом пионеров", "дом физкультуры", "горисполком", "райисполком", "универмаг", "аптека", "кинотеатр", "общежитие", "автобус", "троллейбус")):
        if not any(w in title_low for w in ("школа", "пионеров", "физкультуры", "исполком", "универмаг", "аптека", "кинотеатр", "автобус", "троллейбус")):
            return 0.0

    v_arch = vector_index.encode_photo_caption(arch_low)
    v_land = encode_text(f"{title_low} {normalize_text(landmark_desc)}", title_boost=title_low)
    cos_sim = float(np.dot(v_arch, v_land))

    if dist_km is not None:
        if dist_km <= 0.18:
            cos_sim += 0.10
        elif dist_km > 1.8:
            cos_sim -= 0.18

    return float(np.clip(cos_sim, 0.0, 1.0))
