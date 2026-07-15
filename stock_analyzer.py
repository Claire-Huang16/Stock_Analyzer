"""
技術分析全攻略 · 批次個股評分分析系統
朱家泓方法論：趨勢、K線、均線、成交量 四大維度評分
回後買上漲 8條件進場核對
資料來源：FinMind API v4 (api.finmindtrade.com)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="技術分析全攻略 · 批次分析",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans TC', sans-serif; }

.score-card {
    background: linear-gradient(135deg,#1a1a2e,#16213e);
    border-radius:16px; padding:24px 16px; text-align:center;
    border:1px solid rgba(255,255,255,.1); margin-bottom:8px;
}
.score-num { font-size:68px; font-weight:700; line-height:1; }
.score-lbl { font-size:12px; color:#888; margin-top:4px; letter-spacing:2px; }
.score-vdt { font-size:15px; font-weight:700; margin-top:10px; padding:6px 14px;
             border-radius:8px; display:inline-block; }
.dim-card {
    background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08);
    border-radius:10px; padding:10px 14px; margin-bottom:8px;
}
.adv-box {
    background:linear-gradient(135deg,#0d2137,#0a1a2e);
    border-left:4px solid #f0a500; border-radius:0 12px 12px 0;
    padding:18px 22px;
}
.pb-pass { background:rgba(0,200,100,.12); border:1px solid rgba(0,200,100,.35);
           border-radius:10px; padding:10px 16px; color:#00c864;
           font-size:14px; font-weight:700; margin-bottom:10px; }
.pb-fail { background:rgba(255,60,60,.1); border:1px solid rgba(255,60,60,.3);
           border-radius:10px; padding:10px 16px; color:#ff5555;
           font-size:14px; font-weight:700; margin-bottom:10px; }
.pb-row  { padding:7px 0; border-bottom:1px solid rgba(255,255,255,.05); font-size:13px; }
.sig-bull { background:rgba(0,200,100,.12); color:#00c864; border:1px solid rgba(0,200,100,.3);
            border-radius:20px; padding:3px 9px; font-size:11px; font-weight:600;
            display:inline-block; margin:2px; }
.sig-bear { background:rgba(255,60,60,.12); color:#ff5555; border:1px solid rgba(255,60,60,.3);
            border-radius:20px; padding:3px 9px; font-size:11px; font-weight:600;
            display:inline-block; margin:2px; }
.sig-neu  { background:rgba(180,180,180,.08); color:#aaa; border:1px solid rgba(180,180,180,.2);
            border-radius:20px; padding:3px 9px; font-size:11px; font-weight:600;
            display:inline-block; margin:2px; }
</style>
""", unsafe_allow_html=True)

# ── FinMind API ───────────────────────────────────────────────────────────────
BASE = "https://api.finmindtrade.com/api/v4/data"

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_price(stock_id: str, token: str, days: int = 200) -> pd.DataFrame:
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        r = requests.get(BASE, params={
            "dataset": "TaiwanStockPrice", "data_id": stock_id,
            "start_date": start, "end_date": end, "token": token
        }, timeout=20)
        j = r.json()
        if j.get("status") != 200 or not j.get("data"):
            return pd.DataFrame()
        df = pd.DataFrame(j["data"])
        df["date"]   = pd.to_datetime(df["date"])
        df["open"]   = pd.to_numeric(df["open"],  errors="coerce")
        df["high"]   = pd.to_numeric(df["max"],   errors="coerce")
        df["low"]    = pd.to_numeric(df["min"],   errors="coerce")
        df["close"]  = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["Trading_Volume"], errors="coerce")
        return df[["date","open","high","low","close","volume"]].sort_values("date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_name(stock_id: str, token: str) -> str:
    try:
        r = requests.get(BASE, params={"dataset": "TaiwanStockInfo", "token": token}, timeout=10)
        j = r.json()
        if j.get("status") == 200:
            df = pd.DataFrame(j["data"])
            row = df[df["stock_id"] == stock_id]
            if not row.empty:
                return row.iloc[0].get("stock_name", stock_id)
    except Exception:
        pass
    return stock_id

# ── Technical Indicators ──────────────────────────────────────────────────────
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    for p in [5, 10, 20, 60]:
        df[f"ma{p}"] = c.rolling(p).mean()
    e12 = c.ewm(span=12).mean(); e26 = c.ewm(span=26).mean()
    df["macd"]      = e12 - e26
    df["macd_sig"]  = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_sig"]
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    mid = c.rolling(20).mean(); std = c.rolling(20).std()
    df["bb_upper"] = mid + 2 * std; df["bb_lower"] = mid - 2 * std
    df["vol_ma5"]  = df["volume"].rolling(5).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    lo9  = df["low"].rolling(9).min(); hi9 = df["high"].rolling(9).max()
    rsv  = ((c - lo9) / (hi9 - lo9).replace(0, np.nan) * 100).fillna(50)
    K, D = [50.0], [50.0]
    for v in rsv.iloc[1:]:
        K.append(K[-1] * 2/3 + v * 1/3)
        D.append(D[-1] * 2/3 + K[-1] * 1/3)
    df["kd_k"] = K; df["kd_d"] = D
    return df

def pivot_waves(df: pd.DataFrame, window: int = 5):
    highs, lows = [], []
    for i in range(window, len(df) - window):
        sl_h = df["high"].iloc[i-window:i+window+1]
        sl_l = df["low"].iloc[i-window:i+window+1]
        if df["high"].iloc[i] == sl_h.max(): highs.append(i)
        if df["low"].iloc[i]  == sl_l.min(): lows.append(i)
    return highs, lows

# ── Scoring ───────────────────────────────────────────────────────────────────
def score_trend(df):
    score, sigs = 0, []; last = df.iloc[-1]; tdir = "盤整"
    highs, lows = pivot_waves(df)
    if len(highs) >= 2 and len(lows) >= 2:
        rh = [df["high"].iloc[highs[-2]], df["high"].iloc[highs[-1]]]
        rl = [df["low"].iloc[lows[-2]],  df["low"].iloc[lows[-1]]]
        if rh[1] > rh[0] and rl[1] > rl[0]:   tdir="多頭"; score+=10; sigs.append(("多頭趨勢確立","bull"))
        elif rh[1] < rh[0] and rl[1] < rl[0]: tdir="空頭"; sigs.append(("空頭趨勢確立","bear"))
        else: score+=2; sigs.append(("盤整區間","neu"))
    else: score+=2
    if pd.notna(last.get("ma20")):
        if last.close>last.ma20: score+=5; sigs.append(("站上20MA","bull"))
        else: sigs.append(("跌破20MA","bear"))
    if pd.notna(last.get("ma60")):
        if last.close>last.ma60: score+=5; sigs.append(("站上60MA","bull"))
        else: sigs.append(("跌破60MA","bear"))
    if len(df)>=10 and pd.notna(last.get("ma60")) and pd.notna(df.iloc[-10].get("ma60")):
        slope = (last.ma60-df.iloc[-10].ma60)/df.iloc[-10].ma60*100
        if slope>0.5:   score+=5; sigs.append((f"季線向上+{slope:.1f}%","bull"))
        elif slope<-0.5: sigs.append((f"季線向下{slope:.1f}%","bear"))
        else:            score+=2; sigs.append(("季線走平","neu"))
    return {"score":min(score,25),"max":25,"sigs":sigs,"tdir":tdir}

def score_kline(df):
    score,sigs=0,[]; c=df.iloc[-1]; p1=df.iloc[-2] if len(df)>1 else c; p2=df.iloc[-3] if len(df)>2 else c
    body=abs(c.close-c.open); total=(c.high-c.low) or 0.01
    upSh=c.high-max(c.close,c.open); dnSh=min(c.close,c.open)-c.low; isBull=c.close>c.open
    if isBull:
        score+=5
        if body/total>0.7: score+=3; sigs.append(("實體長紅棒","bull"))
        else: sigs.append(("紅K棒","bull"))
    else:
        if body/total>0.7: sigs.append(("實體長黑棒","bear"))
        else: sigs.append(("黑K棒","bear"))
    s20=df["close"].iloc[-20:]; rLow=s20.min(); rHigh=s20.max()
    pos=(c.close-rLow)/((rHigh-rLow) or 0.01)
    if pos<0.3:
        if dnSh>body*1.5: score+=8; sigs.append(("低檔長下影線","bull"))
        if isBull and c.close>p1.high: score+=5; sigs.append(("低檔突破前高","bull"))
    elif pos>0.7:
        if upSh>body*1.5: sigs.append(("高檔長上影線","bear"))
    if p2.close<p2.open and p1.close<p1.open and isBull and c.close>p1.high and pos<0.4:
        score+=6; sigs.append(("三K底部反轉","bull"))
    if p2.close>p2.open and p1.close>p1.open and not isBull and c.close<p1.low and pos>0.6:
        sigs.append(("三K頂部反轉","bear"))
    half=(c.high+c.low)/2
    if isBull and c.close>half: score+=3; sigs.append((f"超過1/2價{half:.1f}","bull"))
    elif not isBull and c.close<half: sigs.append((f"低於1/2價{half:.1f}","bear"))
    return {"score":min(score,25),"max":25,"sigs":sigs}

def score_ma(df):
    score,sigs=0,[]; last=df.iloc[-1]; prev=df.iloc[-2] if len(df)>1 else last
    if all(pd.notna(last.get(f"ma{p}")) for p in [5,10,20,60]):
        if last.ma5>last.ma10>last.ma20>last.ma60: score+=10; sigs.append(("均線多頭排列","bull"))
        elif last.ma5<last.ma10<last.ma20<last.ma60: sigs.append(("均線空頭排列","bear"))
        else: score+=2; sigs.append(("均線糾結","neu"))
    if len(df)>=3:
        p1=df.iloc[-2]
        if pd.notna(p1.get("ma5")):
            if p1.ma5<p1.ma20 and last.ma5>last.ma20: score+=8; sigs.append(("MA5黃金交叉","bull"))
            elif p1.ma5>p1.ma20 and last.ma5<last.ma20: sigs.append(("MA5死亡交叉","bear"))
    if pd.notna(last.get("ma20")):
        sl20=df["ma20"].dropna().iloc[-10:]
        slope=(sl20.iloc[-1]-sl20.iloc[0])/sl20.iloc[0]*100 if len(sl20)>=2 else 0
        if slope>0 and last.close>last.ma20 and prev.close<prev.ma20: score+=5; sigs.append(("葛蘭畢買1","bull"))
        elif slope>0 and last.close>last.ma20:
            diff=(last.close-last.ma20)/last.ma20*100
            if 0<diff<3: score+=4; sigs.append(("葛蘭畢買2","bull"))
            elif diff>=3: score+=2; sigs.append(("均線上揚強勢","bull"))
    if pd.notna(last.get("ma60")) and last.close>last.ma60: score+=2; sigs.append(("位於季線上方","bull"))
    return {"score":min(score,25),"max":25,"sigs":sigs}

def score_vol(df):
    score,sigs=0,[]; last=df.iloc[-1]; p1=df.iloc[-2] if len(df)>1 else last
    vr=last.volume/last.vol_ma20 if pd.notna(last.get("vol_ma20")) and last.vol_ma20>0 else 1
    v5r=last.vol_ma5/last.vol_ma20 if pd.notna(last.get("vol_ma5")) and pd.notna(last.get("vol_ma20")) and last.vol_ma20>0 else 1
    isUp=last.close>p1.close
    if isUp:
        if vr>=1.5: score+=10; sigs.append((f"上漲爆量{vr:.1f}x","bull"))
        elif vr>=1.0: score+=6; sigs.append((f"上漲放量{vr:.1f}x","bull"))
        else: score+=2; sigs.append(("上漲縮量","neu"))
    else:
        if vr>=1.5: sigs.append((f"下跌爆量{vr:.1f}x","bear"))
        elif vr>=1.0: sigs.append(("下跌放量","bear"))
        else: score+=5; sigs.append(("下跌縮量","neu"))
    low20=df["low"].iloc[-20:].min()
    if last.close<low20*1.15 and isUp and vr>=1.3: score+=8; sigs.append(("底部放量起漲","bull"))
    if v5r>1.2: score+=5; sigs.append((f"均量擴增{v5r:.1f}x","bull"))
    elif v5r<0.8: score+=1; sigs.append(("均量萎縮","neu"))
    score+=2
    return {"score":min(score,25),"max":25,"sigs":sigs}

# ── 回後買上漲 8條件 ──────────────────────────────────────────────────────────
def check_pullback_buy(df):
    last=df.iloc[-1]; prev=df.iloc[-2] if len(df)>1 else last
    results=[]; all_pass=True

    highs,lows=pivot_waves(df); c1=False
    if len(highs)>=2 and len(lows)>=2:
        rh=[df["high"].iloc[highs[-2]],df["high"].iloc[highs[-1]]]
        rl=[df["low"].iloc[lows[-2]], df["low"].iloc[lows[-1]]]
        c1=rh[1]>rh[0] and rl[1]>rl[0]
    results.append({"label":"① 趨勢多頭（高高低低）","pass":c1,"required":True,"detail":""})
    if not c1: all_pass=False

    pb5=df.iloc[-6:-1]
    had_pb=any(pb5["close"].iloc[i]<pb5["close"].iloc[i-1] or pb5["close"].iloc[i]<pb5["open"].iloc[i]
               for i in range(1,len(pb5)))
    c2=had_pb and last.close>prev.close
    results.append({"label":"② 位置回後上漲","pass":c2,"required":True,"detail":""})
    if not c2: all_pass=False

    c3=pd.notna(last.get("ma5")) and last.close>last.ma5
    det=f"5MA={last.ma5:.2f}  收盤={last.close:.2f}" if pd.notna(last.get("ma5")) else ""
    results.append({"label":"③ 收盤站上5MA（平價不算）","pass":c3,"required":True,"detail":det})
    if not c3: all_pass=False

    c4=last.high>prev.high
    results.append({"label":"④ 突破前一日高點","pass":c4,"required":True,
                    "detail":f"今高={last.high:.2f}  昨高={prev.high:.2f}"})
    if not c4: all_pass=False

    chg_pct=(last.close-prev.close)/prev.close*100 if prev.close>0 else 0
    c5=chg_pct>=2.0
    results.append({"label":"⑤ 漲幅2%以上","pass":c5,"required":True,
                    "detail":f"漲幅={chg_pct:.2f}%"})
    if not c5: all_pass=False

    body=last.close-last.open; upSh=last.high-last.close; dnSh=last.open-last.low
    maxSh=max(upSh,dnSh); c6=body>0 and maxSh<=body
    results.append({"label":"⑥ 實體紅K，影線≤實體","pass":c6,"required":True,
                    "detail":f"實體={body:.2f}  最大影線={maxSh:.2f}"})
    if not c6: all_pass=False

    vr=last.volume/last.vol_ma20 if pd.notna(last.get("vol_ma20")) and last.vol_ma20>0 else 1
    c7=vr>=1.0
    results.append({"label":"⑦ 成交量增","pass":c7,"required":False,
                    "detail":f"量比MA20={vr:.2f}x"})

    c8=False; det="KD資料不足"
    if pd.notna(last.get("kd_k")) and pd.notna(prev.get("kd_k")):
        c8=last.kd_k>prev.kd_k; det=f"K={last.kd_k:.1f}  昨K={prev.kd_k:.1f}"
    results.append({"label":"⑧ KD K值向上","pass":c8,"required":True,"detail":det})
    if not c8: all_pass=False

    req_total=sum(1 for r in results if r["required"])
    req_passed=sum(1 for r in results if r["required"] and r["pass"])
    bonus=sum(1 for r in results if not r["required"] and r["pass"])
    return {"results":results,"all_pass":all_pass,"req_passed":req_passed,"req_total":req_total,"bonus":bonus}

# ── Chart ─────────────────────────────────────────────────────────────────────
def draw_chart(df, name):
    tail=df.tail(120)
    vol_colors=["#ef5350" if r.close>=r.open else "#26a69a" for _,r in tail.iterrows()]
    hist_colors=["#ef5350" if (r.macd_hist or 0)>=0 else "#26a69a" for _,r in tail.iterrows()]
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,row_heights=[0.6,0.2,0.2],vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=tail["date"],open=tail["open"],high=tail["high"],
        low=tail["low"],close=tail["close"],name="K線",
        increasing=dict(line=dict(color="#ef5350"),fillcolor="#ef5350"),
        decreasing=dict(line=dict(color="#26a69a"),fillcolor="#26a69a")),row=1,col=1)
    for ma_col,color,nm in [("ma5","#ffeb3b","MA5"),("ma10","#ff9800","MA10"),
                              ("ma20","#2196f3","MA20"),("ma60","#9c27b0","MA60")]:
        fig.add_trace(go.Scatter(x=tail["date"],y=tail[ma_col],name=nm,
                                  line=dict(color=color,width=1.2)),row=1,col=1)
    fig.add_trace(go.Scatter(x=tail["date"],y=tail["bb_upper"],name="BB上軌",
        line=dict(color="rgba(100,200,255,.4)",width=1,dash="dot"),showlegend=False),row=1,col=1)
    fig.add_trace(go.Scatter(x=tail["date"],y=tail["bb_lower"],name="BB下軌",
        line=dict(color="rgba(100,200,255,.4)",width=1,dash="dot"),
        fill="tonexty",fillcolor="rgba(100,200,255,.04)",showlegend=False),row=1,col=1)
    fig.add_trace(go.Bar(x=tail["date"],y=tail["volume"],marker_color=vol_colors,
                          name="成交量",opacity=0.7),row=2,col=1)
    fig.add_trace(go.Scatter(x=tail["date"],y=tail["vol_ma20"],name="量MA20",
                              line=dict(color="#ff9800",width=1.5)),row=2,col=1)
    fig.add_trace(go.Bar(x=tail["date"],y=tail["macd_hist"],marker_color=hist_colors,
                          name="MACD柱",opacity=0.8),row=3,col=1)
    fig.add_trace(go.Scatter(x=tail["date"],y=tail["macd"],name="MACD",
                              line=dict(color="#2196f3",width=1.5)),row=3,col=1)
    fig.add_trace(go.Scatter(x=tail["date"],y=tail["macd_sig"],name="Signal",
                              line=dict(color="#ff9800",width=1.5)),row=3,col=1)
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,20,35,1)",height=650,margin=dict(l=10,r=10,t=40,b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h",y=1.04,bgcolor="rgba(0,0,0,0)",font=dict(size=11)),
        title=dict(text=name,font=dict(color="#fff",size=14)))
    fig.update_yaxes(gridcolor="rgba(255,255,255,.05)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,.05)")
    return fig

# ── UI Helpers ────────────────────────────────────────────────────────────────
def sig_html(sigs):
    parts=[]
    for txt,t in sigs:
        cls={"bull":"sig-bull","bear":"sig-bear","neu":"sig-neu"}.get(t,"sig-neu")
        icon="▲" if t=="bull" else ("▼" if t=="bear" else "─")
        parts.append(f'<span class="{cls}">{icon} {txt}</span>')
    return " ".join(parts) if parts else '<span class="sig-neu">─ 無明顯訊號</span>'

def pb_section(pb):
    if pb["all_pass"]:
        bonus_txt = "  +成交量增加分" if pb["bonus"] else ""
        st.markdown(f'<div class="pb-pass">✅ 符合進場條件（必要條件全部通過）{bonus_txt}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="pb-fail">❌ 不符合進場條件（必要條件 {pb["req_passed"]}/{pb["req_total"]} 通過）</div>',
                    unsafe_allow_html=True)
    rows=""
    for r in pb["results"]:
        icon="✅" if r["pass"] else ("❌" if r["required"] else "—")
        color="#00c864" if r["pass"] else ("#ff5555" if r["required"] else "#888")
        bonus=' <small style="color:#888;background:rgba(255,255,255,.06);padding:1px 5px;border-radius:4px">加分</small>' if not r["required"] else ""
        det=f' <span style="font-size:11px;color:#888">（{r["detail"]}）</span>' if r["detail"] else ""
        rows+=f'<div class="pb-row"><span style="font-size:15px">{icon}</span> <span style="color:{color}">{r["label"]}{bonus}{det}</span></div>'
    st.markdown(f'<div style="background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:8px 16px">{rows}</div>',
                unsafe_allow_html=True)

def dim_bar(title, score, max_score, desc):
    pct=score/max_score*100
    color="#00c864" if pct>=70 else ("#f0a500" if pct>=40 else "#ff3c3c")
    st.markdown(f"""
    <div class="dim-card">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div><div style="font-size:12px;color:#888;letter-spacing:1px">{title}</div>
             <div style="font-size:11px;color:#666">{desc}</div></div>
        <div style="font-size:24px;font-weight:700;color:{color}">{score}<span style="font-size:12px;color:#555">/{max_score}</span></div>
      </div>
      <div style="background:rgba(255,255,255,.05);border-radius:4px;height:5px;margin-top:8px">
        <div style="background:{color};border-radius:4px;height:5px;width:{pct}%"></div>
      </div>
    </div>""", unsafe_allow_html=True)

def render_single(r):
    """渲染單一股票詳細分析"""
    df=r["df"]; total=r["total"]; tr=r["tr"]; kl=r["kl"]; ma_=r["ma"]; vl=r["vl"]; pb=r["pb"]
    name=r["display"]

    if total>=80:   vc,vt,vb="#00c864","強力買進訊號","rgba(0,200,100,.15)"
    elif total>=65: vc,vt,vb="#2196f3","可考慮進場","rgba(33,150,243,.15)"
    elif total>=50: vc,vt,vb="#f0a500","觀望為主","rgba(240,165,0,.15)"
    else:           vc,vt,vb="#ff3c3c","不建議進場","rgba(255,60,60,.15)"

    last=df.iloc[-1]; prev=df.iloc[-2]
    chg=last.close-prev.close; chgp=chg/prev.close*100 if prev.close>0 else 0
    vr=last.volume/last.vol_ma20 if pd.notna(last.get("vol_ma20")) and last.vol_ma20>0 else 1

    # Metrics
    c1,c2,c3,c4=st.columns(4)
    c1.metric("最新收盤",f"{last.close:.1f}",f"{chg:+.1f} ({chgp:+.2f}%)")
    c2.metric("當日成交量",f"{int(last.volume):,}")
    c3.metric("量比 vs MA20",f"{vr:.2f}x","放量" if vr>1.2 else ("縮量" if vr<0.8 else "正常"))
    c4.metric("資料日期",str(last.date)[:10])

    st.divider()
    st.subheader("📊 綜合評分")
    col_s,col_d=st.columns([1,3])
    with col_s:
        st.markdown(f'<div class="score-card"><div class="score-num" style="color:{vc}">{total}</div>'
                    f'<div class="score-lbl">綜合評分 / 100</div>'
                    f'<div class="score-vdt" style="background:{vb};color:{vc}">{vt}</div></div>',
                    unsafe_allow_html=True)
    with col_d:
        dim_bar("📈 趨勢分析",tr["score"],25,"轉折波・多空頭辨別")
        dim_bar("🕯️ K線型態",kl["score"],25,"K線組合・變盤訊號")
        dim_bar("📊 均線系統",ma_["score"],25,"葛蘭畢・多空排列")
        dim_bar("📦 成交量分析",vl["score"],25,"量價關係・量能確認")

    st.divider()
    st.subheader("🔔 技術訊號")
    s1,s2,s3,s4=st.columns(4)
    for col,title,res in [(s1,"趨勢",tr),(s2,"K線",kl),(s3,"均線",ma_),(s4,"成交量",vl)]:
        with col:
            st.markdown(f"**{title}訊號**")
            st.markdown(sig_html(res["sigs"]),unsafe_allow_html=True)

    st.divider()
    st.subheader("🎯 回後買上漲 · 進場條件核對")
    pb_section(pb)

    st.divider()
    st.subheader("💡 操作建議")
    ma20v=last.ma20 if pd.notna(last.get("ma20")) else last.close
    bb_up=last.bb_upper if pd.notna(last.get("bb_upper")) else last.close*1.1
    if total>=80:
        act="🟢 積極做多"
        adv=[f"**{name}** 綜合評分 {total} 分，技術面強勢，建議積極做多。"]
        if tr["tdir"]=="多頭": adv.append("趨勢確立多頭，順勢操作，逢低分批佈局。")
        sl=f"停損設於 **{ma20v*0.97:.1f}** 元（20MA下方3%）"
        tgt=f"目標參考 **{bb_up:.1f}** 元（布林上軌）"
    elif total>=65:
        act="🔵 可考慮進場"
        adv=[f"**{name}** 評分 {total} 分，技術面偏多，可考慮分批進場。","建議等待回測均線後再進場。"]
        sl=f"停損建議 **{ma20v*0.98:.1f}** 元（20MA下方2%）"
        tgt=f"短線目標 **{last.close*1.08:.1f}** 元（+8%）"
    elif total>=50:
        act="🟡 觀望為主"
        adv=[f"**{name}** 評分 {total} 分，技術面訊號混雜，建議觀望。"]
        sl="暫不建議進場"; tgt="等待更佳時機"
    else:
        act="🔴 不適合進場"
        adv=[f"**{name}** 評分 {total} 分，技術面偏空，不建議進場。"]
        if tr["tdir"]=="空頭": adv.append("目前空頭趨勢，切忌逆勢做多。")
        sl="持倉者建議設停損出場"; tgt="等待多頭訊號出現"
    hints=[s for s,t in (tr["sigs"]+kl["sigs"]+ma_["sigs"]+vl["sigs"]) if t=="bull"][:5]
    st.markdown(f"""
    <div class="adv-box">
      <div style="font-size:18px;font-weight:700;margin-bottom:10px">{act}</div>
      <div style="font-size:14px;color:#ccc;line-height:1.8">{'<br>'.join(adv)}</div>
      <div style="margin-top:12px;font-size:13px;color:#bbb;line-height:2">
        🛑 <strong>停損：</strong>{sl}<br>🎯 <strong>目標：</strong>{tgt}
        {'<br>✅ <strong>多頭訊號：</strong>'+' · '.join(hints) if hints else ''}
      </div>
    </div>""", unsafe_allow_html=True)

    st.divider()
    col_r,col_m=st.columns(2)
    with col_r:
        st.markdown("**RSI 指標**")
        rv=last.rsi
        if pd.notna(rv):
            color="#ff5555" if rv>70 else ("#00c864" if rv<30 else "#f0a500")
            lbl="⚠️ 超買區" if rv>70 else ("💚 超賣區" if rv<30 else "正常區間")
            st.markdown(f'<span style="font-size:34px;font-weight:700;color:{color}">{rv:.1f}</span> {lbl}',unsafe_allow_html=True)
        st.markdown("**KD 指標**")
        if pd.notna(last.get("kd_k")):
            kc="#00c864" if last.kd_k>last.kd_d else "#ff5555"
            st.markdown(f'K=<span style="font-weight:700;color:{kc}">{last.kd_k:.1f}</span>  D=**{last.kd_d:.1f}**  {"K>D偏多" if last.kd_k>last.kd_d else "K<D偏空"}',unsafe_allow_html=True)
    with col_m:
        st.markdown("**均線對照**")
        ma_data=[]
        for mn,col in [("MA5","ma5"),("MA10","ma10"),("MA20","ma20"),("MA60","ma60")]:
            v=last.get(col)
            if pd.notna(v):
                diff=(last.close-v)/v*100
                ma_data.append({"均線":mn,"數值":f"{v:.1f}","偏離":f"{diff:+.2f}%","位置":"上方" if diff>0 else "下方"})
        if ma_data: st.dataframe(pd.DataFrame(ma_data),use_container_width=True,hide_index=True)

    st.divider()
    st.subheader("📉 技術分析圖表")
    st.plotly_chart(draw_chart(df,name),use_container_width=True)

    with st.expander("📋 原始資料（最近20筆）"):
        cols=["date","open","high","low","close","volume","ma5","ma20","ma60","rsi","kd_k","kd_d"]
        cols=[c for c in cols if c in df.columns]
        st.dataframe(df[cols].tail(20).round(2),use_container_width=True,hide_index=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 技術分析全攻略")
    st.markdown("*朱家泓方法論 · 批次個股評分*")
    st.divider()
    token = st.text_input("FinMind API Token", type="password", placeholder="輸入您的Token")
    stocks_input = st.text_area(
        "批次股票代號",
        value="2330\n2454\n2317",
        height=160,
        placeholder="每行一個，或逗號分隔\n例：\n2330\n2454\n2317"
    )
    days = st.slider("分析天數", 90, 365, 200, 30)
    go_btn = st.button("🔍 批次分析", use_container_width=True, type="primary")
    st.divider()
    st.markdown("""
**評分維度各25分**
- 📈 趨勢（轉折波）
- 🕯️ K線型態
- 📊 均線系統
- 📦 成交量

**進場判斷**
- 🟢 80+ 積極做多
- 🔵 65-79 可考慮進場
- 🟡 50-64 觀望
- 🔴 <50 不適合進場
""")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📊 技術分析全攻略 · 批次個股評分分析系統")
st.caption("依據朱家泓《技術分析全攻略》課程方法論，從趨勢、K線、均線、成交量四大維度評分")

# Session state
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = 0

if not go_btn and not st.session_state.batch_results:
    st.info("👈 請在左側輸入 FinMind Token 與股票代號後點擊「批次分析」")
    st.stop()

# ── Run batch analysis ──
if go_btn:
    if not token:   st.error("請輸入 FinMind API Token"); st.stop()

    # Parse stock list
    import re
    stocks = [s.strip() for s in re.split(r'[\n,，、\s]+', stocks_input) if s.strip()]
    if not stocks: st.error("請輸入至少一個股票代號"); st.stop()

    st.session_state.batch_results = []
    st.session_state.selected_idx  = 0

    progress_bar = st.progress(0, text=f"批次分析中：0 / {len(stocks)}")
    log_area     = st.empty()
    log_lines    = []

    for i, sid in enumerate(stocks):
        log_lines.append(f"📡 分析 {sid}...")
        log_area.markdown("\n".join(log_lines[-8:]))  # 顯示最後8行

        df_raw = fetch_price(sid, token, days)
        if df_raw.empty:
            log_lines.append(f"❌ {sid} 無資料")
        else:
            name = fetch_name(sid, token)
            df   = add_indicators(df_raw.copy())
            tr   = score_trend(df)
            kl   = score_kline(df)
            ma_  = score_ma(df)
            vl   = score_vol(df)
            pb   = check_pullback_buy(df)
            tot  = tr["score"]+kl["score"]+ma_["score"]+vl["score"]
            st.session_state.batch_results.append({
                "stock_id": sid, "name": name, "display": f"{sid} {name}",
                "df": df, "tr": tr, "kl": kl, "ma": ma_, "vl": vl, "pb": pb, "total": tot
            })
            pb_txt = "✅符合進場" if pb["all_pass"] else f"❌{pb['req_passed']}/{pb['req_total']}通過"
            log_lines.append(f"  ✅ {sid} {name}  得分:{tot}  {pb_txt}")

        progress_bar.progress((i+1)/len(stocks), text=f"批次分析中：{i+1} / {len(stocks)}")
        if i < len(stocks)-1: time.sleep(0.3)

    log_area.empty()
    progress_bar.empty()

    if not st.session_state.batch_results:
        st.error("所有股票均無法取得資料，請確認 Token 與股票代號是否正確。")
        st.stop()

    # Sort by total score descending
    st.session_state.batch_results.sort(key=lambda x: x["total"], reverse=True)

results = st.session_state.batch_results
if not results: st.stop()

# ── Summary Table with filters ──
st.subheader(f"📋 批次分析摘要（{len(results)} 檔，依評分排序）")

# Filter controls
fc1,fc2,fc3,fc4=st.columns([2,2,2,1])
with fc1:
    pb_filter=st.selectbox("進場條件篩選",["全部","✅ 符合進場","❌ 不符合"],key="pb_filter")
with fc2:
    score_filter=st.selectbox("評分篩選",["全部","80+ 積極做多","65-79 考慮進場","50-64 觀望","<50 不建議"],key="score_filter")
with fc3:
    kw=st.text_input("搜尋代號/名稱",placeholder="輸入關鍵字...",key="kw_filter")
with fc4:
    sort_col=st.selectbox("排序欄位",["總分","趨勢","K線","均線","成交量","漲跌幅"],key="sort_col")

# Apply filters
def apply_filters(results):
    out=[]
    for r in results:
        last=r["df"].iloc[-1]; prev=r["df"].iloc[-2]
        chgp=(last.close-prev.close)/prev.close*100 if prev.close>0 else 0
        ok_pb = (pb_filter=="全部" or
                 (pb_filter=="✅ 符合進場" and r["pb"]["all_pass"]) or
                 (pb_filter=="❌ 不符合" and not r["pb"]["all_pass"]))
        ok_sc = (score_filter=="全部" or
                 (score_filter=="80+ 積極做多" and r["total"]>=80) or
                 (score_filter=="65-79 考慮進場" and 65<=r["total"]<80) or
                 (score_filter=="50-64 觀望" and 50<=r["total"]<65) or
                 (score_filter=="<50 不建議" and r["total"]<50))
        ok_kw = not kw or kw in r["stock_id"] or kw in r["name"]
        if ok_pb and ok_sc and ok_kw:
            out.append({**r, "_chgp": chgp})
    return out

filtered = apply_filters(results)

# Sort
sort_key_map={"總分":"total","趨勢":"tr","K線":"kl","均線":"ma","成交量":"vl","漲跌幅":"_chgp"}
sk=sort_key_map.get(sort_col,"total")
if sk in ["tr","kl","ma","vl"]:
    filtered.sort(key=lambda x: x[sk]["score"],reverse=True)
else:
    filtered.sort(key=lambda x: x[sk],reverse=True)

st.caption(f"顯示 {len(filtered)} / {len(results)} 檔")

if filtered:
    # Build summary dataframe
    rows=[]
    for r in filtered:
        last=r["df"].iloc[-1]; prev=r["df"].iloc[-2]
        chgp=(last.close-prev.close)/prev.close*100 if prev.close>0 else 0
        pb_txt="✅ 符合" if r["pb"]["all_pass"] else f"❌ {r['pb']['req_passed']}/{r['pb']['req_total']}"
        score_lbl=("積極做多" if r["total"]>=80 else "考慮進場" if r["total"]>=65
                   else "觀望" if r["total"]>=50 else "不建議")
        rows.append({
            "股票": r["display"],
            "總分": f"{r['total']} {score_lbl}",
            "趨勢": r["tr"]["score"],
            "K線":  r["kl"]["score"],
            "均線": r["ma"]["score"],
            "成交量": r["vl"]["score"],
            "漲跌幅": f"{chgp:+.2f}%",
            "回後買進場": pb_txt,
        })
    df_summary=pd.DataFrame(rows)
    st.dataframe(df_summary, use_container_width=True, hide_index=True,
                 column_config={
                     "總分": st.column_config.TextColumn("總分"),
                     "漲跌幅": st.column_config.TextColumn("漲跌幅"),
                     "回後買進場": st.column_config.TextColumn("回後買進場"),
                 })

st.divider()

# ── Individual Stock Detail ──
st.subheader("📈 個股詳細分析")

# Tab selection
display_names=[r["display"] for r in results]
selected_name=st.selectbox("選擇股票",display_names,
                             index=min(st.session_state.selected_idx,len(display_names)-1),
                             key="stock_selector")
sel_idx=next((i for i,r in enumerate(results) if r["display"]==selected_name),0)
st.session_state.selected_idx=sel_idx

# Tab buttons
n=len(results)
cols_per_row=8
for row_start in range(0,n,cols_per_row):
    cols=st.columns(min(cols_per_row,n-row_start))
    for ci,ri in enumerate(range(row_start,min(row_start+cols_per_row,n))):
        r=results[ri]
        pb_ok=r["pb"]["all_pass"]
        color="green" if pb_ok else ("red" if r["total"]<50 else "gray")
        if cols[ci].button(
            f"{r['stock_id']}\n{r['total']}pt",
            key=f"tabBtn_{ri}",
            type="primary" if ri==sel_idx else "secondary",
            use_container_width=True,
        ):
            st.session_state.selected_idx=ri
            st.rerun()

st.divider()

# Render selected stock
sel=results[st.session_state.selected_idx]
st.subheader(f"🏷️ {sel['display']}")
render_single(sel)
