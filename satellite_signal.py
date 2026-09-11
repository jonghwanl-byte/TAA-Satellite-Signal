#!/usr/bin/env python3
"""
TAA Satellite 시그널 — 국내 상장 ETF 동적 배분

코어(QQQ / TLT / GLD)와 별도 계좌로 독립 운용하는 위성 포트폴리오.
상태 판정 규칙은 코어와 완전히 동일하고, 유니버스와 비중만 다르다.

[상태 판정] 자산별로 20/120/200일선 각각 ON/OFF 상태를 추적한다.
  OFF -> ON : 종가 > MA * 1.015
  ON -> OFF : 종가 < MA * 0.975
  그 외      : 전일 상태 유지

  진입 문턱(+1.5%)과 이탈 문턱(-2.5%)이 달라 그 사이 구간에서는
  상태가 바뀌지 않는다(히스테리시스). 횡보장 휩소를 억제한다.

[포지션 비중] ON 개수로 자산별 기본 비중을 스케일링
  3개 -> 100%   2개 -> 75%   1개 -> 50%   0개 -> 0%
  기본 비중 BASE_WEIGHT 에 위 배율을 곱한다.

  자산별 상한만 적용하고 총합 상한은 두지 않는다. 각 자산은 다른
  자산의 상태와 무관하게 자기 점수로만 비중이 정해지므로, 유니버스에
  종목을 추가해도 규칙이 그대로다. 다만 다수가 동시에 ON이면 합계가
  100%를 넘을 수 있어 리포트에 초과분과 환산 비중을 함께 표시한다.

[주의] 히스테리시스는 경로 의존적이다. 짧은 기간만 받으면 상태가 0에서
  출발해 실제와 다른 신호가 나오고, 매일 시작점이 밀려 어제와 오늘의
  결과가 뒤집힌다. 조회 기간은 반드시 충분히 길게 유지할 것.

[집행] 당일 종가로 산출하고 다음 거래일에 집행한다(lag = 1).

[데이터 소스] 2025-12-27 KRX 정보데이터시스템이 회원제로 전환되어
  pykrx 의 ETF 전용 엔드포인트는 KRX_ID/KRX_PW 없이는 동작하지 않는다.
  일반 시세 엔드포인트(get_market_ohlcv_by_date)는 인증 없이 살아 있어
  이쪽을 1순위로 쓴다. 단 응답이 3000행에서 잘리므로 기간을 나눠 받는다.

환경변수:
  TELEGRAM_BOT_TOKEN  (필수, 구 TELEGRAM_TOKEN 도 인식)
  TELEGRAM_CHAT_ID    (필수, 구 TELEGRAM_TO 도 인식)
  PORTFOLIO_VALUE     (선택) 평가액(KRW). 지정 시 목표 수량(주)까지 계산
  ALWAYS_SEND         (선택) "false"면 비중 변동이 있을 때만 전송. 기본 true
  PRICE_SOURCE        (선택) 소스 순서. 기본 "pykrx,naver,history,download"
  KRX_ID / KRX_PW     (선택) 있으면 pykrx ETF 엔드포인트도 사용
"""

import io
import os
import sys
import time
import contextlib
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests
import yfinance as yf

# ===== 설정 ============================================================
TICKERS = {
    "XLE": "(10%) 미국 에너지",
    "SOX": "(5%) 필라델피아 반도체", 
    "069500.KS": "(10%) 한국 주식",
    "283580.KS": "중국 주식",
    "241180.KS": "일본 주식",
    "453810.KS": "인도 주식",
    "385560.KS": "채권 30년",
    "148070.KS": "채권 10년",
    "396500.KS": "(2%) TIGER 반도체",
    "0091P0.KS": "(1%) 원자력",
    "463250.KS": "(1%) 방산",
    "466920.KS": "(1%) 조선",
}
BASE_WEIGHT = 0.10                                   # 자산별 상한(=기본 비중)
BASE_WEIGHTS = {t: BASE_WEIGHT for t
                in TICKERS}

MA_PERIODS = [20, 120, 200]

BAND_UP = 1.015         # 매수(ON) 문턱  MA +1.5%
BAND_DN = 0.975         # 매도(OFF) 문턱 MA -2.5%

SCALAR_MAP = {3: 1.00, 2: 0.75, 1: 0.50, 0: 0.00}

LOOKBACK = "max"        # 히스테리시스 상태 수렴을 위해 전체 히스토리 사용
DEFAULT_START = "2005-01-01"   # period="max" 가 무시될 때 쓰는 명시적 시작일
KRX_START = "20050101"         # pykrx 조회 시작일
KRX_CHUNK_YEARS = 6            # 3000행 응답 제한 회피용 분할 단위

# 가격 소스 우선순위. 국내 상장 종목은 KRX 원본(pykrx)이 1순위.
# PRICE_SOURCE=naver,history 처럼 환경변수로 덮어쓸 수 있다.
#
# 주의: os.environ.get(key, default) 는 키가 '빈 문자열'로 존재하면 default 를
# 쓰지 않는다. GitHub Actions 에서 미설정 vars 는 빈 문자열로 주입되므로
# 반드시 `or` 로 한 번 더 걸러야 한다.
DEFAULT_SOURCES = "pykrx,naver,history,download"
WARMUP_EXTRA = 250      # 최소 요구 길이 = max(MA) + 이 값. 신규 상장 종목은 제외됨
RETRIES = 4
FETCH_GAP = 0.7         # 종목 간 요청 간격(초). Yahoo 레이트리밋 회피
STALE_DAYS = 5          # 최신 데이터가 이보다 오래되면 경고
TG_MAX_LEN = 3800
KST = timezone(timedelta(hours=9))
# =======================================================================


def env(key: str, default: str = "") -> str:
    """빈 문자열로 주입된 환경변수도 미설정으로 취급한다.

    GitHub Actions 는 정의되지 않은 vars.* 를 빈 문자열로 넘기므로
    os.environ.get(key, default) 만으로는 기본값이 적용되지 않는다.
    """
    return (os.environ.get(key) or default).strip()


HAS_KRX_AUTH = bool(env("KRX_ID") and env("KRX_PW"))


def esc(s) -> str:
    """텔레그램 HTML 파싱 오류 방지."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def fmt(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.2f}"


def pct(w: float) -> str:
    """정수로 떨어지면 정수, 아니면 소수 1자리."""
    return f"{w:.0%}" if abs(w * 100 - round(w * 100)) < 1e-9 else f"{w:.1%}"


# ---------- 데이터 수집 -------------------------------------------------
def _naive(ts):
    """tz 유무와 무관하게 tz-naive Timestamp 로 통일."""
    ts = pd.Timestamp(ts)
    return ts.tz_convert(None) if ts.tzinfo is not None else ts


def _clean(s) -> pd.Series | None:
    """Close 시리즈 정규화. 비어 있으면 None."""
    if s is None or len(s) == 0:
        return None
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return None
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s.sort_index()


def _krx_windows(start: str, end: str):
    """[start, end] 를 KRX_CHUNK_YEARS 단위로 잘라 (from, to) 목록 생성.

    pykrx 응답이 3000행에서 잘리므로(약 12년) 한 번에 받으면 오래된
    구간이 조용히 사라진다. 6년 단위로 나누면 각 구간이 1500행 안쪽이다.
    """
    s = datetime.strptime(start, "%Y%m%d")
    e = datetime.strptime(end, "%Y%m%d")
    out = []
    while s <= e:
        nxt = min(s.replace(year=s.year + KRX_CHUNK_YEARS), e)
        out.append((s.strftime("%Y%m%d"), nxt.strftime("%Y%m%d")))
        if nxt >= e:
            break
        s = nxt + timedelta(days=1)
    return out


def _via_pykrx(ticker: str):
    """KRX 원본 데이터 경로. 국내 상장 종목의 1순위 소스.

    Yahoo 가 .KS 종목에 대해 잘린 히스토리(약 1개월)를 돌려주는 사례가 있어,
    KRX 에서 직접 받는 경로를 먼저 시도한다.

    KRX 회원제 전환 이후 ETF 전용 엔드포인트는 인증이 필요하므로,
    KRX_ID/KRX_PW 가 없으면 일반 시세 엔드포인트만 쓴다.
    """
    from pykrx import stock                              # 지연 임포트

    code = ticker.split(".")[0]
    today = datetime.now(KST).strftime("%Y%m%d")

    getters = [stock.get_market_ohlcv_by_date]
    if HAS_KRX_AUTH:
        getters.insert(0, stock.get_etf_ohlcv_by_date)

    for getter in getters:
        frames = []
        for f, t in _krx_windows(KRX_START, today):
            try:
                # pykrx 는 내부 오류를 stdout 으로 직접 출력한다. 로그를 삼킨다.
                with contextlib.redirect_stdout(io.StringIO()):
                    df = getter(f, t, code)
            except Exception:                            # noqa: BLE001
                continue
            if df is not None and len(df) and "종가" in df.columns:
                frames.append(df)

        if not frames:
            continue

        df = pd.concat(frames)
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        s = _clean(df["종가"])
        if s is not None and len(s):
            return s[s > 0]                              # 휴장일 0 제거

    return None


def _via_naver(ticker: str):
    """네이버 금융 siseJson. 로그인 불필요한 국내 시세 경로."""
    import json
    import re

    code = ticker.split(".")[0]
    today = datetime.now(KST).strftime("%Y%m%d")
    url = (
        "https://api.finance.naver.com/siseJson.naver"
        f"?symbol={code}&requestType=1"
        f"&startTime={KRX_START}&endTime={today}&timeframe=day"
    )
    r = requests.get(url, timeout=30, headers={
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://finance.naver.com/",
    })
    r.raise_for_status()

    # 응답이 JS 리터럴(작은따옴표, 트레일링 콤마) 형태라 정규화가 필요하다.
    txt = re.sub(r",\s*]", "]", r.text.strip().replace("'", '"'))
    rows = json.loads(txt)
    if len(rows) < 2:
        return None

    df = pd.DataFrame(rows[1:], columns=rows[0])
    df.index = pd.to_datetime(df["날짜"].astype(str), format="%Y%m%d")
    s = _clean(df["종가"])
    return s[s > 0] if s is not None else None


def _via_history(ticker: str):
    """yf.Ticker().history() 경로."""
    df = yf.Ticker(ticker).history(
        period=LOOKBACK, interval="1d", auto_adjust=True,
    )
    return _clean(df["Close"]) if df is not None and "Close" in df else None


def _via_download(ticker: str):
    """명시적 start 를 준 yf.download() 경로.

    period="max" 가 무시되고 기본값(1mo)으로 떨어지는 사례가 있어,
    기간을 날짜로 직접 지정한다.
    """
    df = yf.download(
        ticker, start=DEFAULT_START, interval="1d",
        auto_adjust=True, progress=False, threads=False,
    )
    if df is None or len(df) == 0:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.get_level_values(0):
            return None
        return _clean(df["Close"])
    return _clean(df["Close"]) if "Close" in df.columns else None


SOURCE_FNS = {
    "pykrx": _via_pykrx,
    "naver": _via_naver,
    "history": _via_history,
    "download": _via_download,
}


def resolve_sources() -> list:
    """PRICE_SOURCE 파싱. 비었거나 전부 잘못된 값이면 기본 순서로 되돌린다."""
    raw = env("PRICE_SOURCE", DEFAULT_SOURCES)
    names = [s.strip().lower() for s in raw.split(",") if s.strip()]

    unknown = [n for n in names if n not in SOURCE_FNS]
    if unknown:
        print(f"[WARN] 알 수 없는 PRICE_SOURCE 항목 무시: {unknown}", file=sys.stderr)
    names = [n for n in names if n in SOURCE_FNS]

    if not names:
        print(f"[WARN] PRICE_SOURCE 가 비어 기본값 사용: {DEFAULT_SOURCES}",
              file=sys.stderr)
        names = DEFAULT_SOURCES.split(",")
    return names


SOURCES = resolve_sources()
NEED = max(MA_PERIODS) + WARMUP_EXTRA


def fetch_one(ticker: str):
    """(Series, 채택소스, 오류메시지) 반환.

    PRICE_SOURCE 순서대로 시도하고 가장 긴 히스토리를 채택한다. 충분한
    길이를 확보하면 남은 경로는 건너뛴다. 한 종목의 실패가 다른 종목을
    오염시키지 않도록 반드시 종목별로 요청한다.
    """
    best, best_src, errs = None, None, []

    for label in SOURCES:
        fn = SOURCE_FNS[label]

        for attempt in range(1, RETRIES + 1):
            try:
                s = fn(ticker)
                if s is None or len(s) == 0:
                    print(f"  {ticker} [{label}] 빈 응답")
                else:
                    print(f"  {ticker} [{label}] {len(s)}일"
                          f"{'' if len(s) >= NEED else ' — 부족'}")
                    if best is None or len(s) > len(best):
                        best, best_src = s, label
                break
            except ImportError:
                errs.append(f"{label}: 미설치")
                break
            except Exception as e:                      # noqa: BLE001
                errs.append(f"{label}: {e}")
                msg = str(e).lower()
                if any(k in msg for k in ("no data", "not found", "delisted",
                                          "invalid", "no price data")):
                    break                               # 종목 자체 문제 — 재시도 무의미
                if attempt < RETRIES:
                    wait = 3 * attempt
                    print(f"  {ticker} [{label}] 실패({e}) — {wait}초 후 재시도 "
                          f"{attempt}/{RETRIES - 1}", file=sys.stderr)
                    time.sleep(wait)

        if best is not None and len(best) >= NEED:
            break                                       # 충분하면 다음 경로 생략

    if best is None:
        return None, None, ("; ".join(errs) if errs else "빈 응답")
    return best, best_src, ("; ".join(errs) if errs else None)


def fetch_prices():
    """{ticker: Series}, {ticker: 오류메시지} 반환."""
    out, errs = {}, {}
    auth = "KRX 인증 있음" if HAS_KRX_AUTH else "KRX 인증 없음 — 일반 시세 사용"
    print(f"yfinance {getattr(yf, '__version__', '?')} | 소스 순서 {SOURCES} "
          f"| {len(TICKERS)}종목 | {auth}")

    for ticker in TICKERS:
        s, src, err = fetch_one(ticker)
        if s is None:
            errs[ticker] = err
            print(f"  -> {ticker}: 실패 ({err})", file=sys.stderr)
        else:
            out[ticker] = s
            print(f"  -> {ticker}: {src} 채택, {len(s)}일, 최종 {s.index[-1].date()}")
        time.sleep(FETCH_GAP)

    return out, errs


# ---------- 상태 머신 ---------------------------------------------------
def compute_states(close: pd.Series):
    """(정보 dict, 오류메시지) 반환.

    백테스트와 동일한 규칙:
        price > ma * BAND_UP  -> ON
        price < ma * BAND_DN  -> OFF
        그 외                 -> 유지
    """
    if len(close) < NEED:
        return None, f"데이터 부족 ({len(close)}일 / 최소 {NEED}일)"

    mas = {n: close.rolling(n).mean() for n in MA_PERIODS}
    state = {n: 0 for n in MA_PERIODS}
    prev_snapshot = None

    for i in range(max(MA_PERIODS) - 1, len(close)):
        price = float(close.iloc[i])

        nxt = {}
        for n in MA_PERIODS:
            ma = mas[n].iloc[i]
            if pd.isna(ma):
                nxt[n] = 0
                continue
            ma = float(ma)
            s = state[n]
            if price > ma * BAND_UP:
                s = 1
            elif price < ma * BAND_DN:
                s = 0
            nxt[n] = s

        if i == len(close) - 1:
            prev_snapshot = dict(state)     # 마지막 봉 직전 상태
        state = nxt

    if prev_snapshot is None:
        return None, "상태 계산 실패"

    last = float(close.iloc[-1])
    before = float(close.iloc[-2])
    return {
        "today": dict(state),
        "yesterday": prev_snapshot,
        "ma": {n: float(mas[n].iloc[-1]) for n in MA_PERIODS},
        "price": last,
        "pct": (last / before - 1) * 100 if before else 0.0,
        "date": close.index[-1],
    }, None


# ---------- 리포트 ------------------------------------------------------
def build_report():
    prices, fetch_errs = fetch_prices()

    now = datetime.now(KST).strftime("%Y-%m-%d")
    raw_cap = env("PORTFOLIO_VALUE")
    try:
        capital = float(raw_cap.replace(",", "")) if raw_cap else None
    except ValueError:
        print(f"[WARN] PORTFOLIO_VALUE 파싱 실패: {raw_cap!r} — 무시", file=sys.stderr)
        capital = None

    rows, changes, failed, warns = [], [], [], []
    base_date = None
    t_total = y_total = 0.0
    live = {}                       # 환산 비중 표시용 {name: weight}

    for ticker, name in TICKERS.items():
        close = prices.get(ticker)
        if close is None or close.empty:
            failed.append(f"{name} ({ticker}) — {fetch_errs.get(ticker, '데이터 없음')}")
            continue

        info, err = compute_states(close)
        if info is None:
            failed.append(f"{name} ({ticker}) — {err}")
            continue

        if base_date is None:
            base_date = _naive(info["date"])
            stale = (_naive(pd.Timestamp.now(tz="UTC")) - base_date).days
            if stale > STALE_DAYS:
                warns.append(f"데이터가 낡음 — 최종 {base_date.date()} ({stale}일 전)")

        t_on = sum(info["today"].values())
        y_on = sum(info["yesterday"].values())
        t_w = BASE_WEIGHTS[ticker] * SCALAR_MAP[t_on]
        y_w = BASE_WEIGHTS[ticker] * SCALAR_MAP[y_on]
        t_total += t_w
        y_total += y_w
        if t_w > 0:
            live[name] = t_w

        if abs(t_w - y_w) > 1e-9:
            flips = []
            for n in MA_PERIODS:
                if info["today"][n] > info["yesterday"][n]:
                    flips.append(f"{n}일↑")
                elif info["today"][n] < info["yesterday"][n]:
                    flips.append(f"{n}일↓")
            mark = "🔴" if t_w > y_w else "🔵"
            changes.append(
                f"{mark} <b>{esc(name)}</b>  {pct(y_w)} → <b>{pct(t_w)}</b>"
                f"  ({', '.join(flips)})"
            )

        dots = "".join("●" if info["today"][n] else "○" for n in MA_PERIODS)
        arrow = "▲" if info["pct"] > 0 else ("▼" if info["pct"] < 0 else "―")

        line = (f"{dots} <b>{pct(t_w)}</b>  {esc(name)}"
                f"  <code>{esc(ticker)}</code>\n"
                f"     ₩{fmt(info['price'])} {arrow}{abs(info['pct']):.1f}%")
        if capital:
            qty = capital * t_w / info["price"]
            line += f"  ·  {qty:,.0f}주"
        line += "\n     " + "  ".join(
            f"{n}일 {info['price'] / info['ma'][n] - 1:+.1%}" for n in MA_PERIODS
        )
        rows.append(line)

    t_cash, y_cash = 1.0 - t_total, 1.0 - y_total
    if rows and abs(t_cash - y_cash) > 1e-9:
        mark = "🔵" if t_cash > y_cash else "🔴"
        changes.append(
            f"{mark} <b>현금</b>  {pct(max(y_cash, 0))} → <b>{pct(max(t_cash, 0))}</b>"
        )

    ma_label = "/".join(str(n) for n in MA_PERIODS)
    lines = [f"<b>🛰 TAA Satellite — 국내 {len(TICKERS)}자산</b>", f"<i>{now} KST</i>"]
    if base_date is not None:
        lines.append(f"<i>기준: {base_date.strftime('%Y-%m-%d')} 마감 · 익일 집행</i>")
    lines.append("")

    if changes:
        lines.append(f"<b>■ 리밸런싱 필요 — {len(changes)}건</b>")
        lines += changes
    else:
        lines.append("<b>■ 리밸런싱 불필요</b>")
    lines.append("")

    lines.append("<b>■ 목표 비중</b>")
    lines.append(f"<i>● = {ma_label}일선 ON · 자산별 상한 {pct(BASE_WEIGHT)}</i>")
    lines += rows

    if rows:
        cash_line = f"　　 <b>{pct(max(t_cash, 0))}</b>  현금"
        if capital and t_cash > 0:
            cash_line += f"  ·  ₩{fmt(capital * t_cash)}"
        lines.append(cash_line)
        lines.append(f"　　 <i>위험자산 합계 {pct(t_total)}</i>")

    if t_total > 1.0 + 1e-9:
        lines += ["", f"<b>⚠️ 합계가 100%를 {pct(t_total - 1.0)}p 초과</b>",
                  "<i>100% 기준 환산 비중</i>"]
        for name, w in live.items():
            scaled = w / t_total
            entry = f"· {esc(name)}  <b>{scaled:.1%}</b>"
            if capital:
                entry += f"  ·  ₩{fmt(capital * scaled)}"
            lines.append(entry)

    if failed:
        lines += ["", "<b>⚠️ 처리 실패 — 아래 비중은 불완전</b>"]
        lines += [f"· {esc(f)}" for f in failed]
        lines.append("<i>실패 자산은 0%가 아니라 '판정 불가'입니다. "
                     "현금 비중 증가로 읽지 마세요.</i>")
        lines.append(f"<i>yfinance {esc(getattr(yf, '__version__', '?'))}</i>")
    if warns:
        lines += ["", "<b>⚠️ 경고</b>"] + [f"· {esc(w)}" for w in warns]

    return "\n".join(lines), len(changes), len(rows)


# ---------- 텔레그램 ----------------------------------------------------
def send_telegram(text: str) -> bool:
    token = env("TELEGRAM_BOT_TOKEN") or env("TELEGRAM_TOKEN")
    chat_id = env("TELEGRAM_CHAT_ID") or env("TELEGRAM_TO")
    if not token or not chat_id:
        print("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 미설정 — 전송 생략.",
              file=sys.stderr)
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    chunks, buf = [], ""
    for line in text.split("\n"):
        if len(buf) + len(line) + 1 > TG_MAX_LEN:
            chunks.append(buf)
            buf = ""
        buf += line + "\n"
    if buf.strip():
        chunks.append(buf)

    ok = True
    for chunk in chunks:
        try:
            r = requests.post(
                url,
                json={"chat_id": chat_id, "text": chunk,
                      "parse_mode": "HTML", "disable_web_page_preview": True},
                timeout=20,
            )
            if r.status_code != 200:
                print(f"텔레그램 전송 실패 {r.status_code}: {r.text}",
                      file=sys.stderr)
                ok = False
        except Exception as e:                          # noqa: BLE001
            print(f"텔레그램 전송 오류: {e}", file=sys.stderr)
            ok = False
        time.sleep(0.5)
    return ok


# ---------- 진입점 ------------------------------------------------------
def main():
    report, n_changes, n_ok = build_report()

    plain = report
    for tag in ("<b>", "</b>", "<i>", "</i>", "<code>", "</code>"):
        plain = plain.replace(tag, "")
    print(plain.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">"))

    if n_ok == 0:
        print("\n전 종목 처리 실패.", file=sys.stderr)
        send_telegram(report)
        sys.exit(1)

    always = env("ALWAYS_SEND", "true").lower() != "false"
    if n_changes == 0 and not always:
        print("\n변동이 없어 전송하지 않았습니다 (ALWAYS_SEND=false).")
        return

    if not send_telegram(report):
        sys.exit(1)
    print("\n텔레그램 전송 완료.")


if __name__ == "__main__":
    main()
