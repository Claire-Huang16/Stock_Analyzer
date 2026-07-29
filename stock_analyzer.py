# -*- coding: utf-8 -*-
"""
技術分析全攻略 · 個股評分系統（Streamlit 版）
依據朱家泓《技術分析全攻略》方法論：趨勢／K線／均線／成交量 四維度評分，
回後買上漲 8 條件核對，以及 10 種進場型態確認（6種底部型態＋ABC切線＋
上升軌道＋大量黑K＋回後買上漲），並可選擇串接 OpenAI API 做 AI 智能綜合分析。

執行方式：
    pip install streamlit requests pandas numpy plotly --break-system-packages
    streamlit run stock_analyzer_batch.py
"""

import time
import json
import math
from datetime import datetime, timedelta

import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ══════════════════════════════════════════════════════════════════
# 基礎設定
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="技術分析全攻略 · 個股評分系統", layout="wide")

CSS = """
<style>
.badge-pass{background:rgba(0,200,100,.15);color:#00c864;border-radius:4px;padding:2px 8px;font-size:12px;font-weight:700}
.badge-fail{background:rgba(255,60,60,.1);color:#ff5555;border-radius:4px;padding:2px 8px;font-size:12px;font-weight:700}
.badge-obs{background:rgba(240,165,0,.15);color:#f0a500;border-radius:4px;padding:2px 8px;font-size:12px;font-weight:700}
.tag-bull{background:rgba(0,200,100,.12);color:#00c864;border:1px solid rgba(0,200,100,.3);border-radius:20px;padding:3px 10px;font-size:12px;margin:2px;display:inline-block}
.tag-bear{background:rgba(255,60,60,.12);color:#ff5555;border:1px solid rgba(255,60,60,.3);border-radius:20px;padding:3px 10px;font-size:12px;margin:2px;display:inline-block}
.tag-neu{background:rgba(180,180,180,.08);color:#aaa;border:1px solid rgba(180,180,180,.2);border-radius:20px;padding:3px 10px;font-size:12px;margin:2px;display:inline-block}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# FinMind API
# ══════════════════════════════════════════════════════════════════
def get_date_range(days):
    end = datetime.now()
    start = end - timedelta(days=days)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def api_fetch(url):
    """嘗試多個 endpoint（與 HTML 版相同邏輯）"""
    endpoints = [url, url.replace("api.finmindtrade.com/api/v4", "api.finmind.tw/api/latest")]
    last_err = ""
    for ep in endpoints:
        try:
            res = requests.get(ep, timeout=20)
            if res.status_code != 200:
                last_err = "HTTP " + str(res.status_code)
                continue
            j = res.json()
            if j.get("status") == 200:
                return j
            last_err = j.get("msg") or ("status=" + str(j.get("status")))
        except Exception as e:
            last_err = str(e)
    raise RuntimeError(last_err or "fetch failed")


def fetch_price_data(stock_id, token, days):
    start, end = get_date_range(days)
    url = (f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice"
           f"&data_id={stock_id}&start_date={start}&end_date={end}&token={token}")
    j = api_fetch(url)
    data = j.get("data") or []
    if not data:
        raise RuntimeError("無資料（可能代號錯誤）")
    return data


def fetch_name(token, stock_id):
    try:
        url = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInfo&token={token}"
        j = api_fetch(url)
        for row in j.get("data") or []:
            if row.get("stock_id") == stock_id:
                return row.get("stock_name") or stock_id
    except Exception:
        pass
    return stock_id


# ══════════════════════════════════════════════════════════════════
# 技術指標（沿用 HTML 版邏輯，逐根K棒計算，data 為 list[dict]）
# ══════════════════════════════════════════════════════════════════
def calc_ma(data, p):
    out = []
    for i in range(len(data)):
        if i < p - 1:
            out.append(None)
        else:
            out.append(sum(d["close"] for d in data[i - p + 1:i + 1]) / p)
    return out


def calc_ema(data, p):
    k = 2 / (p + 1)
    out = []
    e = data[0]["close"]
    for i in range(len(data)):
        e = data[i]["close"] if i == 0 else data[i]["close"] * k + e * (1 - k)
        out.append(e)
    return out


def calc_macd(data):
    e12, e26 = calc_ema(data, 12), calc_ema(data, 26)
    dif = [e12[i] - e26[i] for i in range(len(data))]
    k = 2 / 10
    sig, s = [], dif[0]
    for i in range(len(dif)):
        s = dif[i] if i == 0 else dif[i] * k + s * (1 - k)
        sig.append(s)
    hist = [dif[i] - sig[i] for i in range(len(dif))]
    return {"dif": dif, "sig": sig, "hist": hist}


def calc_rsi(data, p=14):
    closes = [d["close"] for d in data]
    res = [None] * len(closes)
    if len(closes) <= p:
        return res
    ag = al = 0.0
    for i in range(1, p + 1):
        d = closes[i] - closes[i - 1]
        if d > 0:
            ag += d
        else:
            al += abs(d)
    ag /= p
    al /= p
    res[p] = 100 - 100 / (1 + ag / (al or 0.001))
    for i in range(p + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        ag = (ag * (p - 1) + (d if d > 0 else 0)) / p
        al = (al * (p - 1) + (abs(d) if d < 0 else 0)) / p
        res[i] = 100 - 100 / (1 + ag / (al or 0.001))
    return res


def calc_bb(data, p=20, std=2):
    mid = calc_ma(data, p)
    out = []
    for i in range(len(data)):
        if i < p - 1:
            out.append({"u": None, "l": None})
            continue
        m = mid[i]
        s = sum((data[j]["close"] - m) ** 2 for j in range(i - p + 1, i + 1))
        sigma = math.sqrt(s / p)
        out.append({"u": m + std * sigma, "l": m - std * sigma})
    return out


def calc_volma(data, p):
    out = []
    for i in range(len(data)):
        if i < p - 1:
            out.append(None)
        else:
            out.append(sum(d["volume"] for d in data[i - p + 1:i + 1]) / p)
    return out


def calc_kd(data, period=9):
    k_arr, d_arr = [], []
    prev_k, prev_d = 50.0, 50.0
    for i in range(len(data)):
        if i < period - 1:
            k_arr.append(None)
            d_arr.append(None)
            continue
        seg = data[i - period + 1:i + 1]
        lowest = min(d["low"] for d in seg)
        highest = max(d["high"] for d in seg)
        rsv = 50.0 if highest == lowest else (data[i]["close"] - lowest) / (highest - lowest) * 100
        k = prev_k * 2 / 3 + rsv * 1 / 3
        d_ = prev_d * 2 / 3 + k * 1 / 3
        k_arr.append(k)
        d_arr.append(d_)
        prev_k, prev_d = k, d_
    return {"k": k_arr, "d": d_arr}


def pivots(data, w=5):
    highs, lows = [], []
    n = len(data)
    for i in range(w, n - w):
        h, l = data[i]["high"], data[i]["low"]
        is_h = is_l = True
        for j in range(i - w, i + w + 1):
            if data[j]["high"] > h:
                is_h = False
            if data[j]["low"] < l:
                is_l = False
        if is_h:
            highs.append(i)
        if is_l:
            lows.append(i)
    return {"highs": highs, "lows": lows}


def enrich(raw):
    """raw: list[dict] with date/open/high/low/close/volume, ascending by date"""
    m5, m10, m20, m60 = calc_ma(raw, 5), calc_ma(raw, 10), calc_ma(raw, 20), calc_ma(raw, 60)
    md = calc_macd(raw)
    rs = calc_rsi(raw)
    bbs = calc_bb(raw)
    vm5, vm20 = calc_volma(raw, 5), calc_volma(raw, 20)
    kd = calc_kd(raw, 9)
    out = []
    for i, d in enumerate(raw):
        out.append({
            "date": d["date"], "open": d["open"], "high": d["high"], "low": d["low"],
            "close": d["close"], "volume": d["volume"],
            "ma5": m5[i], "ma10": m10[i], "ma20": m20[i], "ma60": m60[i],
            "macd": md["dif"][i], "macdSig": md["sig"][i], "macdHist": md["hist"][i],
            "rsi": rs[i], "bbU": bbs[i]["u"], "bbL": bbs[i]["l"],
            "vm5": vm5[i], "vm20": vm20[i],
            "kdK": kd["k"][i], "kdD": kd["d"][i],
        })
    return out


# ══════════════════════════════════════════════════════════════════
# 評分（趨勢／K線／均線／成交量，各25分）
# ══════════════════════════════════════════════════════════════════
def score_trend(data):
    score, sigs, tdir = 0, [], "盤整"
    last = data[-1]
    pv = pivots(data)
    if len(pv["highs"]) >= 2 and len(pv["lows"]) >= 2:
        rh = [data[pv["highs"][-2]]["high"], data[pv["highs"][-1]]["high"]]
        rl = [data[pv["lows"][-2]]["low"], data[pv["lows"][-1]]["low"]]
        if rh[1] > rh[0] and rl[1] > rl[0]:
            tdir, score = "多頭", score + 10
            sigs.append(("多頭趨勢確立（高高低低）", "bull"))
        elif rh[1] < rh[0] and rl[1] < rl[0]:
            tdir = "空頭"
            sigs.append(("空頭趨勢確立（低高低低）", "bear"))
        else:
            score += 2
            sigs.append(("盤整區間", "neu"))
    else:
        score += 2
    if last["ma20"] is not None:
        if last["close"] > last["ma20"]:
            score += 5
            sigs.append(("收盤站上20MA", "bull"))
        else:
            sigs.append(("收盤跌破20MA", "bear"))
    if last["ma60"] is not None:
        if last["close"] > last["ma60"]:
            score += 5
            sigs.append(("收盤站上60MA", "bull"))
        else:
            sigs.append(("收盤跌破60MA", "bear"))
    p10 = data[-10] if len(data) >= 10 else None
    if last["ma60"] is not None and p10 is not None and p10["ma60"] is not None:
        sl = (last["ma60"] - p10["ma60"]) / p10["ma60"] * 100
        if sl > 0.5:
            score += 5
            sigs.append((f"季線向上 +{sl:.1f}%", "bull"))
        elif sl < -0.5:
            sigs.append((f"季線向下 {sl:.1f}%", "bear"))
        else:
            score += 2
            sigs.append(("季線走平", "neu"))
    return {"score": min(score, 25), "max": 25, "sigs": sigs, "tdir": tdir}


def score_kline(data):
    score, sigs = 0, []
    c = data[-1]
    p1 = data[-2] if len(data) >= 2 else c
    p2 = data[-3] if len(data) >= 3 else c
    body = abs(c["close"] - c["open"])
    total = (c["high"] - c["low"]) or 0.01
    up_sh = c["high"] - max(c["close"], c["open"])
    dn_sh = min(c["close"], c["open"]) - c["low"]
    is_bull = c["close"] > c["open"]
    if is_bull:
        score += 5
        if body / total > 0.7:
            score += 3
            sigs.append(("實體長紅棒", "bull"))
        else:
            sigs.append(("紅K棒", "bull"))
    else:
        if body / total > 0.7:
            sigs.append(("實體長黑棒", "bear"))
        else:
            sigs.append(("黑K棒", "bear"))
    slice20 = data[-20:]
    r_low = min(d["close"] for d in slice20)
    r_high = max(d["close"] for d in slice20)
    pos = (c["close"] - r_low) / ((r_high - r_low) or 0.01)
    if pos < 0.3:
        if dn_sh > body * 1.5:
            score += 8
            sigs.append(("低檔長下影線（變盤訊號）", "bull"))
        if is_bull and c["close"] > p1["high"]:
            score += 5
            sigs.append(("低檔紅K突破前高", "bull"))
    elif pos > 0.7:
        if up_sh > body * 1.5:
            sigs.append(("高檔長上影線（變盤訊號）", "bear"))
    three_bull = p2["close"] < p2["open"] and p1["close"] < p1["open"] and is_bull and c["close"] > p1["high"]
    if three_bull and pos < 0.4:
        score += 6
        sigs.append(("三K底部反轉組合", "bull"))
    three_bear = p2["close"] > p2["open"] and p1["close"] > p1["open"] and not is_bull and c["close"] < p1["low"]
    if three_bear and pos > 0.6:
        sigs.append(("三K頂部反轉組合", "bear"))
    half = (c["high"] + c["low"]) / 2
    if is_bull and c["close"] > half:
        score += 3
        sigs.append((f"收盤超過1/2價位 {half:.1f}", "bull"))
    elif not is_bull and c["close"] < half:
        sigs.append((f"收盤低於1/2價位 {half:.1f}", "bear"))
    return {"score": min(score, 25), "max": 25, "sigs": sigs}


def score_ma(data):
    score, sigs = 0, []
    last = data[-1]
    prev = data[-2] if len(data) >= 2 else last
    if None not in (last["ma5"], last["ma10"], last["ma20"], last["ma60"]):
        if last["ma5"] > last["ma10"] > last["ma20"] > last["ma60"]:
            score += 10
            sigs.append(("均線多頭排列", "bull"))
        elif last["ma5"] < last["ma10"] < last["ma20"] < last["ma60"]:
            sigs.append(("均線空頭排列", "bear"))
        else:
            score += 2
            sigs.append(("均線糾結", "neu"))
    if len(data) >= 3:
        p1 = data[-2]
        if p1["ma5"] is not None and p1["ma5"] < p1["ma20"] and last["ma5"] > last["ma20"]:
            score += 8
            sigs.append(("MA5 黃金交叉 MA20", "bull"))
        elif p1["ma5"] is not None and p1["ma5"] > p1["ma20"] and last["ma5"] < last["ma20"]:
            sigs.append(("MA5 死亡交叉 MA20", "bear"))
    if last["ma20"] is not None:
        sl20 = [d for d in data[-10:] if d["ma20"] is not None]
        slope = (sl20[-1]["ma20"] - sl20[0]["ma20"]) / sl20[0]["ma20"] * 100 if len(sl20) >= 2 else 0
        if slope > 0 and last["close"] > last["ma20"] and prev["close"] < prev["ma20"]:
            score += 5
            sigs.append(("葛蘭畢買點1（突破均線）", "bull"))
        elif slope > 0 and last["close"] > last["ma20"]:
            diff = (last["close"] - last["ma20"]) / last["ma20"] * 100
            if 0 < diff < 3:
                score += 4
                sigs.append(("葛蘭畢買點2（均線支撐）", "bull"))
            elif diff >= 3:
                score += 2
                sigs.append(("均線上揚股價強勢", "bull"))
    if last["ma60"] is not None and last["close"] > last["ma60"]:
        score += 2
        sigs.append(("股價位於季線上方", "bull"))
    return {"score": min(score, 25), "max": 25, "sigs": sigs}


def score_vol(data):
    score, sigs = 0, []
    last = data[-1]
    p1 = data[-2] if len(data) >= 2 else last
    vr = last["volume"] / last["vm20"] if last["vm20"] else 1
    v5r = (last["vm5"] / last["vm20"]) if (last["vm5"] and last["vm20"]) else 1
    is_up = last["close"] > p1["close"]
    if is_up:
        if vr >= 1.5:
            score += 10
            sigs.append((f"上漲爆量 {vr:.1f}倍（多頭確認）", "bull"))
        elif vr >= 1.0:
            score += 6
            sigs.append((f"上漲放量 {vr:.1f}倍", "bull"))
        else:
            score += 2
            sigs.append(("上漲縮量（動能不足）", "neu"))
    else:
        if vr >= 1.5:
            sigs.append((f"下跌爆量 {vr:.1f}倍（賣壓沉重）", "bear"))
        elif vr >= 1.0:
            sigs.append(("下跌放量", "bear"))
        else:
            score += 5
            sigs.append(("下跌縮量（賣壓減輕）", "neu"))
    low20 = min(d["low"] for d in data[-20:])
    if last["close"] < low20 * 1.15 and is_up and vr >= 1.3:
        score += 8
        sigs.append(("底部放量起漲訊號", "bull"))
    if v5r > 1.2:
        score += 5
        sigs.append((f"近5日均量擴增 {v5r:.1f}x", "bull"))
    elif v5r < 0.8:
        score += 1
        sigs.append(("近5日均量萎縮", "neu"))
    score += 2
    return {"score": min(score, 25), "max": 25, "sigs": sigs}


# ══════════════════════════════════════════════════════════════════
# 回後買上漲：8 條件核對
# ══════════════════════════════════════════════════════════════════
def check_pullback_buy(data):
    last = data[-1]
    prev = data[-2] if len(data) >= 2 else last

    results = []
    all_pass = True

    # ①趨勢多頭（高高低低）
    pv = pivots(data)
    c1 = False
    if len(pv["highs"]) >= 2 and len(pv["lows"]) >= 2:
        rh = [data[pv["highs"][-2]]["high"], data[pv["highs"][-1]]["high"]]
        rl = [data[pv["lows"][-2]]["low"], data[pv["lows"][-1]]["low"]]
        c1 = rh[1] > rh[0] and rl[1] > rl[0]
    results.append({"label": "①趨勢多頭（高高低低）", "pass": c1, "required": True})
    all_pass = all_pass and c1

    # ②位置回後上漲
    pullback_days = data[-6:-1] if len(data) >= 6 else data[:-1]
    had_pullback = False
    for i, d in enumerate(pullback_days):
        prev_close = pullback_days[i - 1]["close"] if i > 0 else d["close"]
        if d["close"] < prev_close or d["close"] < d["open"]:
            had_pullback = True
            break
    c2 = had_pullback and last["close"] > prev["close"]
    results.append({"label": "②位置回後上漲（近期有回檔，今轉上）", "pass": c2, "required": True})
    all_pass = all_pass and c2

    # ③站上5MA
    c3 = last["ma5"] is not None and last["close"] > last["ma5"]
    results.append({"label": "③收盤站上5MA（平價不算）", "pass": c3, "required": True,
                     "detail": (f"5MA={last['ma5']:.2f}  收盤={last['close']:.2f}" if last["ma5"] is not None else "")})
    all_pass = all_pass and c3

    # ④突破前一日高點
    c4 = last["high"] > prev["high"]
    results.append({"label": "④突破前一日高點（含上影線）", "pass": c4, "required": True,
                     "detail": f"今高={last['high']:.2f}  昨高={prev['high']:.2f}"})
    all_pass = all_pass and c4

    # ⑤漲幅2%以上
    chg_pct = (last["close"] - prev["close"]) / prev["close"] * 100 if prev["close"] > 0 else 0
    c5 = chg_pct >= 2.0
    results.append({"label": "⑤漲幅2%以上", "pass": c5, "required": True, "detail": f"漲幅={chg_pct:.2f}%"})
    all_pass = all_pass and c5

    # ⑥實體紅K，影線不大於實體
    body = last["close"] - last["open"]
    up_sh = last["high"] - last["close"]
    dn_sh = last["open"] - last["low"]
    max_sh = max(up_sh, dn_sh)
    c6 = body > 0 and max_sh <= body
    results.append({"label": "⑥實體紅K，影線不大於實體", "pass": c6, "required": True,
                     "detail": f"實體={body:.2f}  最大影線={max_sh:.2f}"})
    all_pass = all_pass and c6

    # ⑦成交量增（加分）
    vol_ratio = last["volume"] / last["vm20"] if last["vm20"] else 1
    c7 = vol_ratio >= 1.0
    results.append({"label": "⑦成交量增（加分項）", "pass": c7, "required": False,
                     "detail": f"量比MA20={vol_ratio:.2f}x"})

    # ⑧KD指標確認（K值向上）
    prev_kd = data[-2] if len(data) >= 2 else None
    c8 = False
    if last["kdK"] is not None and prev_kd is not None and prev_kd["kdK"] is not None:
        c8 = last["kdK"] > prev_kd["kdK"]
    results.append({"label": "⑧指標確認（K值向上）", "pass": c8, "required": True,
                     "detail": (f"K={last['kdK']:.1f}  昨K={prev_kd['kdK']:.1f}"
                                if last["kdK"] is not None and prev_kd and prev_kd["kdK"] is not None
                                else "KD資料不足")})
    all_pass = all_pass and c8

    required_total = sum(1 for r in results if r["required"])
    required_passed = sum(1 for r in results if r["required"] and r["pass"])
    bonus_passed = sum(1 for r in results if not r["required"] and r["pass"])

    return {"results": results, "allPass": all_pass, "requiredPassed": required_passed,
            "requiredTotal": required_total, "bonusPassed": bonus_passed}


# ══════════════════════════════════════════════════════════════════
# 型態確認：10 種進場型態
# (1)頭肩底 (2)複式頭肩底 (3)N字底 (4)三重底 (5)圓弧底 (6)一字底(均線糾結)
# (7)突破ABC修正下降切線 (8)突破上升軌道線 (9)突破飆股大量黑K最高點 (10)回後買上漲
# ══════════════════════════════════════════════════════════════════
def _tolerant(a, b, pct):
    base = max(abs(a), abs(b), 1e-6)
    return abs(a - b) / base <= pct


def detect_patterns(data, pb):
    last = len(data) - 1
    last_close = data[last]["close"]
    last_vol = data[last]["volume"]
    vm20 = data[last]["vm20"]
    vol_confirm = (last_vol / vm20 >= 1.3) if vm20 else False

    pv = pivots(data, 4)
    floor = max(0, len(data) - 90)
    lows = [i for i in pv["lows"] if floor <= i < last]
    highs = [i for i in pv["highs"] if floor <= i < last]

    def breakout_check(resistance):
        if resistance is None:
            return {"confirmed": False, "detail": ""}
        confirmed = last_close > resistance
        detail = f"頸線/壓力＝{resistance:.2f}　現價＝{last_close:.2f}" + ("　(帶量突破)" if vol_confirm else "")
        return {"confirmed": confirmed, "detail": detail}

    def ma_slope_up(key, n):
        p = data[max(0, last - n)]
        c = data[last]
        if c[key] is None or p[key] is None:
            return False
        return c[key] > p[key]

    results = []

    # (1) 頭肩底
    def pat_hs():
        name = "頭肩底"
        if len(lows) >= 3:
            l3 = lows[-3:]
            L1, L2, L3 = data[l3[0]]["low"], data[l3[1]]["low"], data[l3[2]]["low"]
            shoulders_similar = _tolerant(L1, L3, 0.06)
            head_lower = L2 < L1 * 0.985 and L2 < L3 * 0.985
            if shoulders_similar and head_lower:
                h_between = [i for i in highs if l3[0] < i < l3[2]]
                neck = max((data[i]["high"] for i in h_between), default=None)
                bo = breakout_check(neck)
                return {"id": "hs", "name": name, "formed": True, "breakout": bo["confirmed"],
                        "detail": bo["detail"] or "型態成形，等待突破頸線",
                        "desc": "左右肩低點相近，頭部最低，突破頸線為買點"}
        return {"id": "hs", "name": name, "formed": False, "breakout": False, "detail": "尚未偵測到符合結構",
                "desc": "左右肩低點相近，頭部最低，突破頸線為買點"}
    results.append(pat_hs())

    # (2) 複式頭肩底
    def pat_chs():
        name = "複式頭肩底"
        if len(lows) >= 4:
            last_lows = lows[-5:]
            low_vals = [data[i]["low"] for i in last_lows]
            min_val = min(low_vals)
            head_pos = low_vals.index(min_val)
            has_left = head_pos > 0
            has_right = head_pos < len(last_lows) - 1
            shoulder_vals = [v for idx, v in enumerate(low_vals) if idx != head_pos]
            shoulders_ok = has_left and has_right and shoulder_vals and all(
                _tolerant(v, shoulder_vals[0], 0.08) and v > min_val * 1.02 for v in shoulder_vals)
            if shoulders_ok:
                h_between = [i for i in highs if last_lows[0] < i < last_lows[-1]]
                neck = max((data[i]["high"] for i in h_between), default=None)
                bo = breakout_check(neck)
                return {"id": "chs", "name": name, "formed": True, "breakout": bo["confirmed"],
                        "detail": bo["detail"] or "型態成形，等待突破頸線",
                        "desc": "多重肩部低點環繞單一最低頭部，突破頸線為買點"}
        return {"id": "chs", "name": name, "formed": False, "breakout": False, "detail": "尚未偵測到符合結構",
                "desc": "多重肩部低點環繞單一最低頭部，突破頸線為買點"}
    results.append(pat_chs())

    # (3) N字底
    def pat_nb():
        name = "N字底"
        if len(lows) >= 2 and len(highs) >= 1:
            A, C = lows[-2], lows[-1]
            b_cands = [i for i in highs if A < i < C]
            if b_cands:
                B = b_cands[-1]
                low_a, low_c, high_b = data[A]["low"], data[C]["low"], data[B]["high"]
                if low_c > low_a:
                    bo = breakout_check(high_b)
                    return {"id": "nb", "name": name, "formed": True, "breakout": bo["confirmed"],
                            "detail": bo["detail"] or "拉回未破前低，等待突破反彈高點",
                            "desc": "低點反彈後拉回不破前低，再突破反彈高點為買點"}
        return {"id": "nb", "name": name, "formed": False, "breakout": False, "detail": "尚未偵測到符合結構",
                "desc": "低點反彈後拉回不破前低，再突破反彈高點為買點"}
    results.append(pat_nb())

    # (4) 三重底
    def pat_tb():
        name = "三重底"
        if len(lows) >= 3:
            l3 = lows[-3:]
            vals = [data[i]["low"] for i in l3]
            all_similar = (_tolerant(vals[0], vals[1], 0.08) and _tolerant(vals[1], vals[2], 0.08)
                           and _tolerant(vals[0], vals[2], 0.08))
            if all_similar:
                h_between = [i for i in highs if l3[0] < i < l3[2]]
                res = max((data[i]["high"] for i in h_between), default=None)
                bo = breakout_check(res)
                return {"id": "tb", "name": name, "formed": True, "breakout": bo["confirmed"],
                        "detail": bo["detail"] or "型態成形，等待突破壓力",
                        "desc": "三個低點高度相近，突破期間高點為買點"}
        return {"id": "tb", "name": name, "formed": False, "breakout": False, "detail": "尚未偵測到符合結構",
                "desc": "三個低點高度相近，突破期間高點為買點"}
    results.append(pat_tb())

    # (5) 圓弧底
    def pat_rb():
        name = "圓弧底"
        win = data[-40:]
        if len(win) >= 30:
            seg = len(win) // 3
            first, mid, tail = win[:seg], win[seg:len(win) - seg], win[len(win) - seg:]

            def avg(arr, key):
                return sum(d[key] for d in arr) / len(arr)

            def slope(arr):
                n = len(arr)
                sx = sy = sxy = sxx = 0.0
                for i, d in enumerate(arr):
                    sx += i; sy += d["close"]; sxy += i * d["close"]; sxx += i * i
                denom = (n * sxx - sx * sx) or 1
                return (n * sxy - sx * sy) / denom

            slope_first, slope_tail = slope(first), slope(tail)
            mid_low = min(d["low"] for d in mid)
            is_convex = avg(first, "close") > mid_low * 1.01 and avg(tail, "close") > mid_low * 1.01
            shape_ok = slope_first < 0 and slope_tail > 0 and is_convex
            avg_range = sum((d["high"] - d["low"]) / d["close"] for d in win) / len(win)
            low_vol = avg_range < 0.05
            if shape_ok and low_vol:
                resistance = max(d["high"] for d in first)
                bo = breakout_check(resistance)
                return {"id": "rb", "name": name, "formed": True, "breakout": bo["confirmed"],
                        "detail": bo["detail"] or "弧形築底中，等待突破起跌壓力",
                        "desc": "價格緩跌後緩升成U型，突破起跌點高點為買點"}
        return {"id": "rb", "name": name, "formed": False, "breakout": False, "detail": "尚未偵測到符合結構",
                "desc": "價格緩跌後緩升成U型，突破起跌點高點為買點"}
    results.append(pat_rb())

    # (6) 一字底（均線糾結）
    def pat_fb():
        name = "一字底(均線糾結)"
        N = 10
        win = data[-N - 1:-1] if len(data) >= N + 1 else []
        ok = len(win) == N and all(
            d["ma5"] is not None and d["ma10"] is not None and d["ma20"] is not None and d["ma60"] is not None
            for d in win)
        if ok:
            def tangled_check(d):
                vals = [d["ma5"], d["ma10"], d["ma20"], d["ma60"]]
                mx, mn = max(vals), min(vals)
                return (mx - mn) / mn <= 0.035
            tangled = all(tangled_check(d) for d in win)
            narrow_range = all((d["high"] - d["low"]) / d["close"] <= 0.045 for d in win)
            if tangled and narrow_range:
                resistance = max(d["high"] for d in win)
                last4 = [data[last]["ma5"], data[last]["ma10"], data[last]["ma20"], data[last]["ma60"]]
                above_all_ma = all(v is not None and last_close > v for v in last4)
                breakout = last_close > resistance and above_all_ma and vol_confirm
                detail = (f"整理區間高點＝{resistance:.2f}　現價＝{last_close:.2f}"
                          + ("　(帶量突破)" if vol_confirm else "　(尚未帶量)"))
                return {"id": "fb", "name": name, "formed": True, "breakout": breakout, "detail": detail,
                        "desc": "均線糾結、價格窄幅整理，帶量突破整理區間為買點"}
        return {"id": "fb", "name": name, "formed": False, "breakout": False, "detail": "尚未偵測到符合結構",
                "desc": "均線糾結、價格窄幅整理，帶量突破整理區間為買點"}
    results.append(pat_fb())

    # (7) 突破ABC修正下降切線
    def pat_abc():
        name = "突破ABC修正下降切線"
        win_floor = max(0, len(data) - 60)
        highs_in_win = [i for i in highs if i >= win_floor]
        if highs_in_win:
            start_idx = highs_in_win[0]
            corr_highs = [i for i in highs_in_win if start_idx < i <= start_idx + 20]
            if len(corr_highs) >= 2:
                h1, h2 = corr_highs[0], corr_highs[-1]
                y1, y2 = data[h1]["high"], data[h2]["high"]
                if y2 < y1 and h2 > h1:
                    slope = (y2 - y1) / (h2 - h1)
                    line_at_last = y1 + slope * (last - h1)
                    ma20up = ma_slope_up("ma20", 10)
                    is_red = data[last]["close"] > data[last]["open"]
                    breakout = last_close > line_at_last and ma20up and is_red
                    detail = (f"下降切線位置≈{line_at_last:.2f}　現價＝{last_close:.2f}"
                              + ("　MA20上揚" if ma20up else "　MA20未上揚"))
                    return {"id": "abc", "name": name, "formed": True, "breakout": breakout, "detail": detail,
                            "desc": "多頭回檔呈ABC下跌，反彈高點畫下降切線，MA20上揚下帶量紅K突破切線為買點"}
        return {"id": "abc", "name": name, "formed": False, "breakout": False, "detail": "尚未偵測到符合結構",
                "desc": "多頭回檔呈ABC下跌，反彈高點畫下降切線，MA20上揚下帶量紅K突破切線為買點"}
    results.append(pat_abc())

    # (8) 突破上升軌道線
    def pat_channel():
        name = "突破上升軌道線"
        floor2 = max(0, len(data) - 60)
        lows_in = [i for i in lows if i >= floor2]
        highs_in = [i for i in highs if i >= floor2]
        if len(lows_in) >= 2:
            l1, l2 = lows_in[-2], lows_in[-1]
            ly1, ly2 = data[l1]["low"], data[l2]["low"]
            if ly2 > ly1 and l2 > l1:
                slope2 = (ly2 - ly1) / (l2 - l1)
                offsets = [data[i]["high"] - (ly1 + slope2 * (i - l1)) for i in highs_in if i > l1]
                offset = max(offsets) if offsets else None
                if offset is not None and offset > 0:
                    upper_at_last = ly1 + slope2 * (last - l1) + offset
                    ma20up2 = ma_slope_up("ma20", 10)
                    is_red2 = data[last]["close"] > data[last]["open"]
                    breakout2 = last_close > upper_at_last and ma20up2 and is_red2 and vol_confirm
                    detail = (f"軌道上緣≈{upper_at_last:.2f}　現價＝{last_close:.2f}"
                              + ("　帶量" if vol_confirm else "　量未放大"))
                    return {"id": "channel", "name": name, "formed": True, "breakout": breakout2, "detail": detail,
                            "desc": "股價沿上升軌道緩步上漲，MA20上揚下帶量長紅收盤突破軌道上緣為買點"}
        return {"id": "channel", "name": name, "formed": False, "breakout": False, "detail": "尚未偵測到符合結構",
                "desc": "股價沿上升軌道緩步上漲，MA20上揚下帶量長紅收盤突破軌道上緣為買點"}
    results.append(pat_channel())

    # (9) 突破飆股大量黑K最高點
    def pat_blackk():
        name = "突破飆股大量黑K最高點"
        lookback, confirm_window = 10, 3
        floor3 = max(0, len(data) - 1 - lookback)
        candidates = []
        for i in range(floor3, last):
            d = data[i]
            if d["close"] < d["open"] and d["vm20"] and d["volume"] / d["vm20"] >= 1.6:
                candidates.append(i)
        if candidates:
            bk_idx = candidates[-1]
            bk_high = data[bk_idx]["high"]
            within_window = 0 <= (last - bk_idx) <= confirm_window
            if within_window:
                is_red3 = data[last]["close"] > data[last]["open"]
                ma20up3 = ma_slope_up("ma20", 10)
                breakout3 = last_close > bk_high and is_red3 and vol_confirm and ma20up3
                detail = (f"大量黑K高點＝{bk_high:.2f}　現價＝{last_close:.2f}"
                          + ("　帶量" if vol_confirm else "　量未放大"))
                return {"id": "blackk", "name": name, "formed": True, "breakout": breakout3, "detail": detail,
                        "desc": "飆股急漲後出現大量黑K回檔，3日內帶量長紅突破其最高點為買點"}
        return {"id": "blackk", "name": name, "formed": False, "breakout": False, "detail": "尚未偵測到符合結構",
                "desc": "飆股急漲後出現大量黑K回檔，3日內帶量長紅突破其最高點為買點"}
    results.append(pat_blackk())

    # (10) 回後買上漲（沿用 checkPullbackBuy 結果）
    if pb:
        pb_formed = pb["allPass"] or pb["requiredPassed"] >= pb["requiredTotal"] - 1
        results.append({
            "id": "pbup", "name": "回後買上漲",
            "formed": pb_formed, "breakout": pb["allPass"],
            "detail": f"必要條件 {pb['requiredPassed']}/{pb['requiredTotal']} 通過" + ("　+成交量增加分" if pb["bonusPassed"] else ""),
            "desc": "趨勢多頭，回檔量縮價穩後，今日紅K放量突破前高為買點",
        })

    any_breakout = any(r["breakout"] for r in results)
    any_formed = any(r["formed"] for r in results)
    return {"results": results, "anyBreakout": any_breakout, "anyFormed": any_formed}


# ══════════════════════════════════════════════════════════════════
# OpenAI API：AI 智能綜合分析
# ══════════════════════════════════════════════════════════════════
def build_analysis_prompt(stock_id, name, data, tr, kl, ma_, vl, pb, pt, total):
    last = data[-1]
    prev = data[-2] if len(data) >= 2 else last
    chg = last["close"] - prev["close"]
    chgp = (chg / prev["close"] * 100) if prev["close"] else 0

    lines = []
    lines.append(f"股票：{stock_id} {name}")
    lines.append(f"資料日期：{last['date']}　收盤：{last['close']}　漲跌：{chg:+.2f} ({chgp:+.2f}%)")
    vr = (last["volume"] / last["vm20"]) if last["vm20"] else None
    lines.append(f"成交量：{last['volume']}　量比(vs MA20量)：{(f'{vr:.2f}x' if vr is not None else 'N/A')}")
    lines.append("")
    lines.append(f"【朱家泓四維度評分】總分 {total}/100")
    lines.append(f"趨勢 {tr['score']}/25、K線 {kl['score']}/25、均線 {ma_['score']}/25、成交量 {vl['score']}/25")
    all_sigs = tr["sigs"] + kl["sigs"] + ma_["sigs"] + vl["sigs"]
    bull_sigs = [s[0] for s in all_sigs if s[1] == "bull"]
    bear_sigs = [s[0] for s in all_sigs if s[1] == "bear"]
    if bull_sigs:
        lines.append("多頭訊號：" + "、".join(bull_sigs))
    if bear_sigs:
        lines.append("空頭訊號：" + "、".join(bear_sigs))
    lines.append("")
    lines.append(f"【回後買上漲 8條件核對】必要條件通過 {pb['requiredPassed']}/{pb['requiredTotal']}"
                 + ("（全數通過）" if pb["allPass"] else ""))
    lines.append("")
    lines.append("【型態確認，10種進場型態】")
    for p in pt["results"]:
        status = "✅已突破" if p["breakout"] else ("🕒成形中未突破" if p["formed"] else "－未偵測到")
        lines.append(f"・{p['name']}：{status}" + (f"（{p['detail']}）" if p.get("detail") else ""))

    return "\n".join(lines)


def run_ai_analysis(api_key, model, prompt_text):
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "temperature": 0.4,
            "max_tokens": 900,
            "messages": [
                {"role": "system", "content": (
                    "你是一位精通台股技術分析的資深操盤手，熟悉朱家泓《技術分析全攻略》方法論"
                    "（趨勢轉折波、K線、均線、成交量四維度評分，以及回後買上漲、頭肩底等進場型態）。"
                    "請根據使用者提供的個股技術數據摘要，用繁體中文給出：1) 整體技術面研判（3-4句）"
                    "2) 進場時機與風險提示 3) 綜合建議（積極做多／可考慮／觀望／不建議）。"
                    "語氣專業、精簡、避免空泛用詞，並提醒這僅為技術面參考，非投資建議。"
                )},
                {"role": "user", "content": prompt_text},
            ],
        },
        timeout=60,
    )
    j = resp.json()
    if resp.status_code != 200:
        msg = (j.get("error") or {}).get("message") or f"HTTP {resp.status_code}"
        raise RuntimeError(msg)
    return j["choices"][0]["message"]["content"]


# ══════════════════════════════════════════════════════════════════
# 圖表
# ══════════════════════════════════════════════════════════════════
def draw_chart(data, name):
    tail = data[-120:]
    dates = [d["date"] for d in tail]
    vol_colors = ["#ef5350" if d["close"] >= d["open"] else "#26a69a" for d in tail]
    hist_colors = ["#ef5350" if (d["macdHist"] or 0) >= 0 else "#26a69a" for d in tail]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                         row_heights=[0.58, 0.20, 0.22])

    fig.add_trace(go.Candlestick(
        x=dates, open=[d["open"] for d in tail], high=[d["high"] for d in tail],
        low=[d["low"] for d in tail], close=[d["close"] for d in tail], name="K線",
        increasing_line_color="#ef5350", increasing_fillcolor="#ef5350",
        decreasing_line_color="#26a69a", decreasing_fillcolor="#26a69a"), row=1, col=1)

    for key, label, color in [("ma5", "MA5", "#ffeb3b"), ("ma10", "MA10", "#ff9800"),
                               ("ma20", "MA20", "#2196f3"), ("ma60", "MA60", "#9c27b0")]:
        fig.add_trace(go.Scatter(x=dates, y=[d[key] for d in tail], name=label,
                                  line=dict(color=color, width=1.2)), row=1, col=1)

    fig.add_trace(go.Scatter(x=dates, y=[d["bbU"] for d in tail], name="BB上軌",
                              line=dict(color="rgba(100,200,255,.4)", width=1, dash="dot"),
                              showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=[d["bbL"] for d in tail], name="BB下軌",
                              line=dict(color="rgba(100,200,255,.4)", width=1, dash="dot"),
                              fill="tonexty", fillcolor="rgba(100,200,255,.04)",
                              showlegend=False), row=1, col=1)

    fig.add_trace(go.Bar(x=dates, y=[d["volume"] for d in tail], marker_color=vol_colors,
                          name="成交量", opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=[d["vm20"] for d in tail], name="量MA20",
                              line=dict(color="#ff9800", width=1.5)), row=2, col=1)

    fig.add_trace(go.Bar(x=dates, y=[d["macdHist"] for d in tail], marker_color=hist_colors,
                          name="MACD柱", opacity=0.8), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=[d["macd"] for d in tail], name="MACD",
                              line=dict(color="#2196f3", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=[d["macdSig"] for d in tail], name="Signal",
                              line=dict(color="#ff9800", width=1.5)), row=3, col=1)

    fig.update_layout(
        title=f"{name} 技術分析圖", template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,20,35,1)",
        height=720, margin=dict(l=50, r=15, t=45, b=25),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        showlegend=True,
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# UI ── Sidebar
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 📊 技術分析全攻略")
    st.caption("朱家泓方法論 · 個股評分系統")
    st.divider()

    token = st.text_input("FinMind API Token", type="password", placeholder="輸入您的Token")
    stocks_raw = st.text_area("批次股票代號", value="2330\n2454\n2317", height=140,
                              placeholder="每行一個，或逗號分隔\n例：\n2330\n2454\n2317\n0050")
    days = st.slider("分析天數", min_value=90, max_value=365, value=180, step=30)

    st.divider()
    openai_key = st.text_input("OpenAI API Key（選填）", type="password", placeholder="sk-...")
    st.caption("用於「🤖 AI 智能綜合分析」，金鑰只會從您本機直接呼叫 OpenAI，不會被儲存或上傳。")
    openai_model = st.selectbox("AI 模型", ["gpt-4o-mini", "gpt-4o"],
                                 format_func=lambda x: x + ("（快速／經濟）" if x == "gpt-4o-mini" else "（進階／較貴）"))

    run_btn = st.button("🔍 批次分析", type="primary", use_container_width=True)

    st.divider()
    st.markdown("**評分維度各25分**")
    st.caption("📈 趨勢分析（轉折波）")
    st.caption("🕯️ K線型態分析")
    st.caption("📊 均線系統分析")
    st.caption("📦 成交量分析")
    st.markdown("**進場判斷**")
    st.caption("🟢 80+ 積極做多　🔵 65-79 可考慮進場")
    st.caption("🟡 50-64 觀望　🔴 <50 不適合進場")


# ══════════════════════════════════════════════════════════════════
# UI ── Main：批次分析
# ══════════════════════════════════════════════════════════════════
st.markdown("## 📊 技術分析全攻略 · 個股評分分析系統")
st.caption("依據朱家泓《技術分析全攻略》課程方法論，從趨勢、K線、均線、成交量四大維度評分")

if "batch_results" not in st.session_state:
    st.session_state.batch_results = []

if run_btn:
    if not token:
        st.error("請輸入 FinMind API Token")
    else:
        # 測試連線
        with st.spinner("🔌 測試 API 連線中..."):
            try:
                test_json = api_fetch(f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInfo&token={token}")
                if test_json.get("status") != 200:
                    st.error(f"Token 無效或 API 錯誤：{test_json.get('msg', 'status=' + str(test_json.get('status')))}")
                    st.stop()
            except Exception as e:
                st.error(f"❌ 無法連線至 FinMind API\n\n錯誤：{e}\n\n請確認 Token 與網路連線是否正常。")
                st.stop()

        import re
        stock_list = [s.strip() for s in re.split(r"[\n,，、\s]+", stocks_raw) if s.strip()]
        if not stock_list:
            st.error("請輸入至少一個股票代號")
            st.stop()

        results = []
        prog = st.progress(0, text=f"批次分析中：0 / {len(stock_list)}")
        log_box = st.empty()
        log_lines = []

        for i, sid in enumerate(stock_list):
            log_lines.append(f"📡 分析 {sid}...")
            log_box.write("\n\n".join(log_lines))
            try:
                rows = fetch_price_data(sid, token, days)
                raw_data = sorted(
                    [{"date": d["date"], "open": float(d["open"]), "high": float(d["max"]),
                      "low": float(d["min"]), "close": float(d["close"]),
                      "volume": float(d["Trading_Volume"])} for d in rows],
                    key=lambda d: d["date"])
                name = fetch_name(token, sid)
                data = enrich(raw_data)
                tr, kl, ma_, vl = score_trend(data), score_kline(data), score_ma(data), score_vol(data)
                pb = check_pullback_buy(data)
                pt = detect_patterns(data, pb)
                total = tr["score"] + kl["score"] + ma_["score"] + vl["score"]
                results.append({"stockId": sid, "name": name, "data": data, "tr": tr, "kl": kl,
                                 "ma": ma_, "vl": vl, "pb": pb, "pt": pt, "total": total})
                log_lines[-1] = f"✅ {sid} {name}　得分：{total}"
            except Exception as ex:
                log_lines[-1] = f"❌ {sid} 失敗：{ex}"
            log_box.write("\n\n".join(log_lines))
            prog.progress((i + 1) / len(stock_list), text=f"批次分析中：{i + 1} / {len(stock_list)}")
            if i < len(stock_list) - 1:
                time.sleep(0.6)  # 避免 API 速率限制

        prog.empty()
        if not results:
            st.error("所有股票均無法取得資料")
        else:
            results.sort(key=lambda r: r["total"], reverse=True)
            st.session_state.batch_results = results


# ══════════════════════════════════════════════════════════════════
# UI ── 批次摘要表 ＋ 篩選
# ══════════════════════════════════════════════════════════════════
results = st.session_state.batch_results

if not results:
    st.info("請在左側輸入 FinMind API Token 與股票代號，點擊「開始分析」即可開始。")
else:
    st.markdown("### 📋 批次分析摘要")

    fc1, fc2, fc3, fc4 = st.columns([1, 1, 1, 1.4])
    with fc1:
        pb_filter = st.radio("進場條件", ["全部", "✅ 符合進場", "❌ 不符合"], horizontal=False, key="pb_filter")
    with fc2:
        score_filter = st.radio("評分", ["全部", "80+", "65-79", "50-64", "<50"], horizontal=False, key="score_filter")
    with fc3:
        pt_filter = st.radio("型態確認", ["全部", "✅ 已突破", "🕒 成形中", "－ 無"], horizontal=False, key="pt_filter")
    with fc4:
        kw = st.text_input("搜尋代號/名稱", key="kw_filter")

    def row_visible(r):
        ok_pb = pb_filter == "全部" or (pb_filter == "✅ 符合進場" and r["pb"]["allPass"]) or \
                (pb_filter == "❌ 不符合" and not r["pb"]["allPass"])
        ok_score = (score_filter == "全部"
                    or (score_filter == "80+" and r["total"] >= 80)
                    or (score_filter == "65-79" and 65 <= r["total"] < 80)
                    or (score_filter == "50-64" and 50 <= r["total"] < 65)
                    or (score_filter == "<50" and r["total"] < 50))
        ok_kw = not kw or kw in r["stockId"] or kw in r["name"]
        ok_pt = (pt_filter == "全部"
                 or (pt_filter == "✅ 已突破" and r["pt"]["anyBreakout"])
                 or (pt_filter == "🕒 成形中" and r["pt"]["anyFormed"] and not r["pt"]["anyBreakout"])
                 or (pt_filter == "－ 無" and not r["pt"]["anyFormed"]))
        return ok_pb and ok_score and ok_kw and ok_pt

    rows = []
    for r in results:
        if not row_visible(r):
            continue
        last = r["data"][-1]
        prev = r["data"][-2] if len(r["data"]) >= 2 else last
        chgp = (last["close"] - prev["close"]) / prev["close"] * 100 if prev["close"] else 0
        pb_txt = "符合進場" if r["pb"]["allPass"] else f"{r['pb']['requiredPassed']}/{r['pb']['requiredTotal']} 通過"
        pt_names = [p["name"] for p in r["pt"]["results"]
                    if (p["breakout"] if r["pt"]["anyBreakout"] else p["formed"])]
        rows.append({
            "股票": f"{r['stockId']} {r['name']}",
            "總分": r["total"],
            "趨勢": r["tr"]["score"], "K線": r["kl"]["score"], "均線": r["ma"]["score"], "成交量": r["vl"]["score"],
            "漲跌幅": f"{chgp:+.2f}%", "收盤": round(last["close"], 1),
            "回後買進場": ("✅ " if r["pb"]["allPass"] else "❌ ") + pb_txt,
            "型態確認": ("✅ " if r["pt"]["anyBreakout"] else ("🕒 " if r["pt"]["anyFormed"] else "－ "))
                        + ("、".join(pt_names) if pt_names else "無"),
        })

    st.caption(f"顯示 {len(rows)} / {len(results)} 檔")
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════
    # 個股詳細分析
    # ══════════════════════════════════════════════════════════════
    st.divider()
    labels = [f"{r['stockId']} {r['name']}（{r['total']}分）" for r in results]
    sel = st.selectbox("選擇個股查看詳細分析", options=list(range(len(results))),
                        format_func=lambda i: labels[i])
    r = results[sel]
    data, tr, kl, ma_, vl, pb, pt, total = r["data"], r["tr"], r["kl"], r["ma"], r["vl"], r["pb"], r["pt"], r["total"]
    name_full = f"{r['stockId']} {r['name']}"
    last = data[-1]
    prev = data[-2] if len(data) >= 2 else last
    chg = last["close"] - prev["close"]
    chgp = (chg / prev["close"] * 100) if prev["close"] else 0
    vr = (last["volume"] / last["vm20"]) if last["vm20"] else 1

    st.markdown(f"## 🏷️ {name_full}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("最新收盤", f"{last['close']:.2f}", f"{chg:+.2f} ({chgp:+.2f}%)")
    m2.metric("當日成交量", f"{last['volume']:,.0f}")
    m3.metric("量比 vs MA20", f"{vr:.2f}x", "放量" if vr > 1.2 else ("縮量" if vr < 0.8 else "正常"))
    m4.metric("資料日期", last["date"])

    st.divider()
    st.markdown("### 📊 綜合評分")
    if total >= 80:
        vc, vt = "🟢", "強力買進訊號"
    elif total >= 65:
        vc, vt = "🔵", "可考慮進場"
    elif total >= 50:
        vc, vt = "🟡", "觀望為主"
    else:
        vc, vt = "🔴", "不建議進場"
    sc1, sc2 = st.columns([1, 3])
    with sc1:
        st.metric("綜合評分 / 100", total)
        st.markdown(f"**{vc} {vt}**")
    with sc2:
        dims = [("📈 趨勢分析", tr), ("🕯️ K線型態", kl), ("📊 均線系統", ma_), ("📦 成交量", vl)]
        for dt, dr in dims:
            pct = dr["score"] / dr["max"]
            st.write(f"{dt}　{dr['score']}/{dr['max']}")
            st.progress(pct)

    st.divider()
    st.markdown("### 🔔 技術訊號")
    sig_cols = st.columns(4)
    for col, (t, sigs) in zip(sig_cols, [("趨勢訊號", tr["sigs"]), ("K線訊號", kl["sigs"]),
                                          ("均線訊號", ma_["sigs"]), ("成交量訊號", vl["sigs"])]):
        with col:
            st.write(f"**{t}**")
            if sigs:
                for s, kind in sigs:
                    icon = "▲" if kind == "bull" else ("▼" if kind == "bear" else "─")
                    st.caption(f"{icon} {s}")
            else:
                st.caption("無明顯訊號")

    st.divider()
    st.markdown("### 💡 操作建議")
    ma20v = last["ma20"] or last["close"]
    bb_up = last["bbU"] or last["close"] * 1.1
    if total >= 80:
        act = "🟢 積極做多"
        adv = [f"**{name_full}** 綜合評分 {total} 分，技術面強勢，建議積極做多。"]
        if tr["tdir"] == "多頭":
            adv.append("趨勢確立多頭，順勢操作，逢低分批佈局。")
        sl = f"停損設於 **{ma20v*0.97:.1f}** 元（20MA下方3%）"
        tgt = f"目標參考 **{last['close']*((bb_up-last['close'])/last['close']+1):.1f}** 元（布林上軌）"
    elif total >= 65:
        act = "🔵 可考慮進場"
        adv = [f"**{name_full}** 評分 {total} 分，技術面偏多，可考慮分批進場。", "建議等待回測均線後再進場，降低風險。"]
        sl = f"停損建議 **{ma20v*0.98:.1f}** 元（20MA下方2%）"
        tgt = f"短線目標 **{last['close']*1.08:.1f}** 元（+8%）"
    elif total >= 50:
        act = "🟡 觀望為主"
        adv = [f"**{name_full}** 評分 {total} 分，技術面訊號混雜，建議觀望。", "等待均線整理完畢或趨勢明確後再行動。"]
        sl = "暫不建議進場"
        tgt = "等待更佳時機"
    else:
        act = "🔴 不適合進場"
        adv = [f"**{name_full}** 評分 {total} 分，技術面偏空，不建議進場。"]
        adv.append("目前空頭趨勢，切忌逆勢做多，等待趨勢反轉。" if tr["tdir"] == "空頭" else "技術指標偏弱，應持現金等待機會。")
        sl = "持倉者建議設停損出場"
        tgt = "等待多頭訊號出現"

    st.markdown(f"**{act}**")
    for line in adv:
        st.markdown(line)
    st.markdown(f"🛑 **停損：** {sl}")
    st.markdown(f"🎯 **目標：** {tgt}")
    hints = [s for s, k in (tr["sigs"] + kl["sigs"] + ma_["sigs"] + vl["sigs"]) if k == "bull"][:5]
    if hints:
        st.markdown("✅ **多頭訊號：** " + " · ".join(hints))

    # ── AI 智能綜合分析 ──
    st.write("")
    if st.button("🤖 AI 智能綜合分析", key=f"ai_btn_{sel}"):
        if not openai_key:
            st.warning("請先在左側輸入 OpenAI API Key，才能使用 AI 智能綜合分析。")
        else:
            with st.spinner("正在請 AI 綜合研判技術面數據，請稍候…"):
                try:
                    prompt = build_analysis_prompt(r["stockId"], r["name"], data, tr, kl, ma_, vl, pb, pt, total)
                    content = run_ai_analysis(openai_key, openai_model, prompt)
                    st.session_state[f"ai_result_{sel}"] = content
                except Exception as e:
                    st.session_state[f"ai_result_{sel}"] = f"❌ AI 分析失敗：{e}\n請確認 API Key 是否正確、額度是否足夠，或稍後再試。"
    if st.session_state.get(f"ai_result_{sel}"):
        st.markdown("#### 🤖 AI 分析結果")
        st.markdown(st.session_state[f"ai_result_{sel}"])

    st.divider()
    st.markdown("### 🎯 回後買上漲 · 進場條件核對")
    for cond in pb["results"]:
        icon = "✅" if cond["pass"] else ("❌" if cond["required"] else "➖")
        extra = f"　({cond['detail']})" if cond.get("detail") else ""
        st.write(f"{icon} {cond['label']}{extra}")
    st.caption(f"必要條件通過：{pb['requiredPassed']}/{pb['requiredTotal']}"
               + ("　✅ 全數通過，符合進場條件！" if pb["allPass"] else ""))

    st.divider()
    st.markdown("### 🔍 型態確認（10種進場型態）")
    for p in pt["results"]:
        icon = "✅" if p["breakout"] else ("🕒" if p["formed"] else "－")
        st.write(f"{icon} **{p['name']}** — {p['desc']}")
        if p.get("detail"):
            st.caption(p["detail"])

    st.divider()
    st.markdown("### 📐 指標對照")
    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        rv = last["rsi"]
        st.write("**RSI 指標**")
        if rv is not None:
            st.metric("RSI(14)", f"{rv:.1f}")
            st.caption("⚠️ 超買區（>70），注意回調" if rv > 70 else ("💚 超賣區（<30），留意反彈" if rv < 30 else "位於正常區間（30-70）"))
    with ic2:
        st.write("**KD 指標**")
        if last["kdK"] is not None and last["kdD"] is not None:
            kd_up = last["kdK"] > last["kdD"]
            st.markdown(f"K=**:{'green' if kd_up else 'red'}[{last['kdK']:.1f}]**　D=**{last['kdD']:.1f}**")
            st.caption("K>D 偏多" if kd_up else "K<D 偏空")
        else:
            st.caption("資料不足")
    with ic3:
        st.write("**均線對照**")
        ma_rows = []
        for label, key in [("MA5", "ma5"), ("MA10", "ma10"), ("MA20", "ma20"), ("MA60", "ma60")]:
            if last[key] is not None:
                diff = (last["close"] - last[key]) / last[key] * 100
                ma_rows.append({"均線": label, "數值": round(last[key], 2),
                                 "股價偏離": f"{diff:+.2f}% ({'上方' if diff > 0 else '下方'})"})
        st.dataframe(pd.DataFrame(ma_rows), hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("### 📉 技術分析圖表")
    st.plotly_chart(draw_chart(data, name_full), use_container_width=True)

    with st.expander("📋 原始資料（最近20筆）"):
        raw_rows = []
        for d in list(reversed(data))[:20]:
            raw_rows.append({
                "日期": d["date"], "開盤": d["open"], "最高": d["high"], "最低": d["low"], "收盤": d["close"],
                "成交量": d["volume"],
                "MA5": round(d["ma5"], 2) if d["ma5"] is not None else None,
                "MA20": round(d["ma20"], 2) if d["ma20"] is not None else None,
                "MA60": round(d["ma60"], 2) if d["ma60"] is not None else None,
                "RSI": round(d["rsi"], 2) if d["rsi"] is not None else None,
                "KD-K": round(d["kdK"], 2) if d["kdK"] is not None else None,
                "KD-D": round(d["kdD"], 2) if d["kdD"] is not None else None,
            })
        st.dataframe(pd.DataFrame(raw_rows), hide_index=True, use_container_width=True)
