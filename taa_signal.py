#!/usr/bin/env python3
"""
TAA Satellite Signal — 국내 상장 해외/국가 대표지수 ETF 범용 추세 신호

[규칙]
  일봉 이평선 20/60/120/200일 각각 ON/OFF 판정 (밴드 ±2%, 히스테리시스)
    OFF -> ON : 종가 > MA x 1.02
    ON  -> OFF: 종가 < MA x 0.98
    그 사이   : 전일 상태 유지
  목표 비중 = (켜진 일봉 이평선 수 / 4) x 자산별 한도(20%)
  주봉(20/60주)은 참고 표시 전용 (비중 계산에 쓰지 않음)

[집행] 당일 종가로 신호를 계산하고 다음 거래일 종가 무렵(14시경)에 매매한다.

[환경변수]
  TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID  (필수)
  PORTFOLIO_VALUE  (선택) 평가액(KRW). 있으면 목표 금액/수량도 표시
  ALWAYS_SEND      (선택) "false"면 비중 변동·전환 임박·오류가 있을 때만 전송 (기본 true)
  PRICE_SOURCE     (선택) 소스 순서. 기본 "pykrx,naver,history"
  CSV_DIR          (선택) 지정 시 {코드}.csv(Date,Close)를 읽는다. 로컬 테스트용
  DRY_RUN          (선택) "true"면 텔레그램 전송 없이 출력만
"""
import contextlib
import html
import io
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

# ===== 설정 ============================================================
ASSETS = {                                  # 코드: 이름
    "069500": "KODEX 200",
    "283580": "KODEX 중국본토CSI300",
    "241180": "TIGER 니케이225",
    "453810": "KODEX 인도Nifty50",
    "385560": "KBSTAR KIS국고채30년",
}
DAILY_MA = (20, 60, 120, 200)
WEEKLY_MA = (20, 60)
BAND = 0.02                 # 진입 +2% / 이탈 -2%
CAP = 0.20                  # 자산별 한도
NEAR = 0.005                # 전환선까지 0.5%p 이내면 '전환 임박'
STALE_DAYS = 6              # 연휴 감안: 마지막 데이터가 이보다 오래되면 경고
START = "20050101"
RETRIES = 3
TG_MAX_LEN = 3800
KST = timezone(timedelta(hours=9))


def env(key, default=""):
    return (os.environ.get(key) or default).strip()


# ---------- 데이터 수집 -------------------------------------------------
def _clean(s):
    if s is None or len(s) == 0:
        return None
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[s > 0]
    if len(s) == 0:
        return None
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s[~s.index.duplicated(keep="last")].sort_index()


def _via_csv(code):
    d = pd.read_csv(os.path.join(env("CSV_DIR"), f"{code}.csv"),
                    parse_dates=["Date"]).set_index("Date")
    return _clean(d["Close"])


def _via_pykrx(code):
    """KRX 일반 시세 엔드포인트. 응답이 3000행에서 잘려 6년씩 나눠 받는다."""
    from pykrx import stock
    today = datetime.now(KST)
    s = datetime.strptime(START, "%Y%m%d")
    frames = []
    while s <= today:
        e = min(s.replace(year=s.year + 6), today)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                df = stock.get_market_ohlcv_by_date(
                    s.strftime("%Y%m%d"), e.strftime("%Y%m%d"), code)
            if df is not None and len(df) and "종가" in df.columns:
                frames.append(df)
        except Exception:                                   # noqa: BLE001
            pass
        s = e + timedelta(days=1)
    if not frames:
        return None
    df = pd.concat(frames)
    df.index = pd.to_datetime(df.index)
    return _clean(df["종가"])


def _via_naver(code):
    today = datetime.now(KST).strftime("%Y%m%d")
    r = requests.get(
        "https://api.finance.naver.com/siseJson.naver"
        f"?symbol={code}&requestType=1&startTime={START}&endTime={today}&timeframe=day",
        timeout=30, headers={"User-Agent": "Mozilla/5.0",
                             "Referer": "https://finance.naver.com/"})
    r.raise_for_status()
    txt = re.sub(r",\s*]", "]", r.text.strip().replace("'", '"'))
    rows = json.loads(txt)
    if len(rows) < 2:
        return None
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df.index = pd.to_datetime(df["날짜"].astype(str), format="%Y%m%d")
    return _clean(df["종가"])


def _via_history(code):
    import yfinance as yf
    df = yf.Ticker(f"{code}.KS").history(period="max", interval="1d", auto_adjust=True)
    return _clean(df["Close"]) if df is not None and "Close" in df else None


SOURCES = {"pykrx": _via_pykrx, "naver": _via_naver, "history": _via_history}


def fetch_one(code):
    """(Series, 소스, 오류) — 소스 순서대로 시도, 가장 긴 히스토리 채택."""
    need = max(DAILY_MA) + 60
    if env("CSV_DIR"):
        return _via_csv(code), "csv", None
    order = [x.strip().lower() for x in env("PRICE_SOURCE", "pykrx,naver,history").split(",")]
    order = [x for x in order if x in SOURCES] or ["pykrx", "naver", "history"]
    best, src, errs = None, None, []
    for name in order:
        for k in range(1, RETRIES + 1):
            try:
                s = SOURCES[name](code)
                if s is not None and len(s) and (best is None or len(s) > len(best)):
                    best, src = s, name
                break
            except Exception as e:                          # noqa: BLE001
                errs.append(f"{name}: {e}")
                if k < RETRIES:
                    time.sleep(3 * k)
        if best is not None and len(best) >= need:
            break
    return best, src, ("; ".join(errs) if errs else None)


# ---------- 신호 계산 ---------------------------------------------------
def states(close, ma):
    """히스테리시스 상태열. 첫 유효일은 종가>MA 로 초기화한다."""
    s, out = np.nan, np.full(len(close), np.nan)
    for i, (x, m) in enumerate(zip(close, ma)):
        if np.isnan(m):
            continue
        if x > m * (1 + BAND):
            s = 1.0
        elif x < m * (1 - BAND):
            s = 0.0
        elif np.isnan(s):
            s = 1.0 if x > m else 0.0
        out[i] = s
    return out


def analyze(c, wins):
    rows = []
    for n in wins:
        ma = c.rolling(n).mean()
        st = states(c.values, ma.values)
        ok = len(c) >= n
        x = c.iloc[-1]
        rows.append(dict(
            n=n, st=st[-1], prev=st[-2] if len(st) > 1 else np.nan,
            gap=x / ma.iloc[-1] - 1 if ok else np.nan,
            dd=x / c.iloc[-n:].max() - 1 if ok else np.nan,
            up=x / c.iloc[-n:].min() - 1 if ok else np.nan))
    return rows


def near_flip(a):
    if np.isnan(a["gap"]) or np.isnan(a["st"]):
        return False
    dist = a["gap"] + BAND if a["st"] == 1 else BAND - a["gap"]
    return dist <= NEAR


# ---------- 리포트 ------------------------------------------------------
dot = lambda s: "-" if s != s else ("●" if s == 1 else "○")
fmt = lambda v: "  n/a" if v != v else f"{v * 100:+.1f}%"


def build_report():
    top = [f"<b>📊 TAA Satellite 신호</b> (밴드 ±{BAND * 100:g}%)", ""]
    bot = ["<b>📋 상세</b>", "<pre>"]
    notable, n_ok = False, 0
    pv = float(env("PORTFOLIO_VALUE", "0") or 0)
    today = datetime.now(KST).date()

    for code, name in ASSETS.items():
        c, src, err = fetch_one(code)
        if c is None or len(c) < max(DAILY_MA) + 2:
            top += [f"<b>{html.escape(name)}</b> ({code})",
                    f"⚠ 데이터 오류: {html.escape(str(err or '데이터 부족'))[:120]}", ""]
            notable = True
            continue
        n_ok += 1
        w = c.resample("W-FRI").last().dropna()
        D, W = analyze(c, DAILY_MA), analyze(w, WEEKLY_MA)
        on = int(sum(a["st"] == 1 for a in D))
        pon = int(sum(a["prev"] == 1 for a in D))
        tw, pw = on / 4 * CAP, pon / 4 * CAP
        d1 = c.iloc[-1] / c.iloc[-2] - 1
        arrow = "▲" if d1 > 0 else "▼" if d1 < 0 else ""
        last = c.index[-1].date()

        top.append(f"<b>{html.escape(name)}</b> ({code}) {last:%m/%d}")
        top.append(f"일 {''.join(dot(a['st']) for a in D)}  "
                   f"주 {''.join(dot(a['st']) for a in W)}  "
                   f"전일대비 {arrow}{abs(d1) * 100:.1f}%")
        if on != pon:
            verb = "비중확대" if on > pon else "비중축소"
            top.append(f"목표 {tw * 100:.0f}% (한도 {CAP * 100:.0f}% 중)  "
                       f"{'▲' if on > pon else '▼'} {verb} {pw * 100:.0f}%→{tw * 100:.0f}%")
            notable = True
        else:
            top.append(f"목표 {tw * 100:.0f}% (한도 {CAP * 100:.0f}% 중)  변동없음")
        if pv > 0:
            amt = pv * tw
            top.append(f"   ≈ {amt:,.0f}원 / {int(amt // c.iloc[-1])}주")
        near = [a for a in D if near_flip(a)]
        if near:
            top.append("⚠ 전환 임박: " + ", ".join(f"일{a['n']}" for a in near))
            notable = True
        if (today - last).days >= STALE_DAYS:
            top.append(f"⚠ 데이터가 오래됨 (최종 {last})")
            notable = True
        top.append("")

        bot.append(f"{name} ({code})")
        bot.append("      고점대비  저점대비  MA대비")
        for tag, rows in (("일", D), ("주", W)):
            for a in rows:
                bot.append(f"{tag}{a['n']:<3}{fmt(a['dd']):>8}{fmt(a['up']):>9}{fmt(a['gap']):>8}")
        bot.append("")

    top.append("● ON  ○ OFF  - 데이터부족 (20/60/120/200)")
    top.append("목표=켜진 일봉 수÷4×한도 · 주봉은 참고")
    bot.append("</pre>")
    return "\n".join(top) + "\n━━━━━━━━━━\n" + "\n".join(bot), notable, n_ok


# ---------- 텔레그램 ----------------------------------------------------
def send_telegram(text):
    token, chat = env("TELEGRAM_BOT_TOKEN"), env("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 미설정 — 전송 생략", file=sys.stderr)
        return False
    chunks, buf, in_pre = [], "", False
    for line in text.split("\n"):
        if len(buf) + len(line) + 8 > TG_MAX_LEN:
            chunks.append(buf + ("</pre>" if in_pre else ""))
            buf = "<pre>" if in_pre else ""
        if "<pre>" in line:
            in_pre = True
        if "</pre>" in line:
            in_pre = False
        buf += line + "\n"
    if buf.strip():
        chunks.append(buf)
    ok = True
    for ch in chunks:
        try:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              json={"chat_id": chat, "text": ch, "parse_mode": "HTML",
                                    "disable_web_page_preview": True}, timeout=20)
            if r.status_code != 200:
                print(f"텔레그램 실패 {r.status_code}: {r.text}", file=sys.stderr)
                ok = False
        except Exception as e:                              # noqa: BLE001
            print(f"텔레그램 오류: {e}", file=sys.stderr)
            ok = False
        time.sleep(0.5)
    return ok


def main():
    report, notable, n_ok = build_report()
    plain = re.sub(r"</?(b|i|pre|code)>", "", report)
    print(html.unescape(plain))
    if n_ok == 0:
        print("\n전 종목 처리 실패", file=sys.stderr)
        if env("DRY_RUN").lower() != "true":
            send_telegram(report)
        sys.exit(1)
    if env("DRY_RUN").lower() == "true":
        print("\n[DRY_RUN] 전송 생략")
        return
    if env("ALWAYS_SEND", "true").lower() == "false" and not notable:
        print("\n알릴 변동이 없어 전송하지 않음 (ALWAYS_SEND=false)")
        return
    if not send_telegram(report):
        sys.exit(1)
    print("\n텔레그램 전송 완료")


if __name__ == "__main__":
    main()
