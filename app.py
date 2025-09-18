#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MVP：OpenAI API + 滑桿調整風格（單檔 Flask）

使用方式：
1) pip install flask flask-cors openai python-dotenv
2) 在專案根目錄建立 .env 並寫入：
   OPENAI_API_KEY=你的API金鑰
   OPENAI_MODEL=gpt-4o-mini
3) python app.py 啟動，瀏覽 http://127.0.0.1:5000
"""
import os
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv

# --- 載入 .env ---
load_dotenv()

# --- Flask App ---
app = Flask(__name__)
CORS(app)

# --- OpenAI Client ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ 找不到 OPENAI_API_KEY，請檢查 .env 檔案")
client = OpenAI(api_key=api_key)

# 預設模型，從 .env 讀取，沒有就 fallback
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# --- HTML（由後端直接回傳） ---
INDEX_HTML = r"""
<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>雞掰風格調整 MVP</title>
<style>
:root { --gap: 12px; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, "Noto Sans TC", sans-serif; margin: 0; background: #0b0c10; color: #eaeaea; }
.wrap { max-width: 900px; margin: 0 auto; padding: 24px; }
h1 { font-size: 22px; margin: 0 0 8px; }
.card { background: #121317; border: 1px solid #1f212a; border-radius: 16px; padding: 16px; box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset; }
.row { display: grid; grid-template-columns: 1fr; gap: var(--gap); }
label { font-size: 14px; color: #b8c0cc; }
textarea { width: 100%; min-height: 120px; border-radius: 12px; border: 1px solid #2a2f3a; background:#0f1117; color: #eaeaea; padding: 12px; resize: vertical; }
input[type=range] { width: 100%; }
.hint { font-size: 12px; color: #9aa3af; }
.btn { cursor: pointer; border: 0; border-radius: 12px; padding: 10px 14px; background: #3b82f6; color: #fff; font-weight: 600; }
.btn:disabled { opacity: .6; cursor: not-allowed; }
.out { white-space: pre-wrap; background:#0f1117; border:1px solid #2a2f3a; padding:12px; border-radius:12px; min-height: 100px; }
.bar { display:flex; align-items:center; gap:10px; }
.pill { font-size:12px; background:#1f2937; border:1px solid #2a2f3a; padding:6px 10px; border-radius:999px; }
</style>
</head>
<body>
<div class="wrap">
<h1>雞掰風格調整（0–10）</h1>
<p class="hint">拉動滑桿控制模型回覆的「雞掰程度」。0=中性禮貌，10=極度尖銳、強烈（仍避開仇恨/歧視）。</p>
<div class="card row" style="margin-top:12px;">
<div>
<label>輸入你的訊息</label>
<textarea id="msg" placeholder="例如：幫我回覆同事，他一直拖交作業…"></textarea>
</div>
<div>
<div class="bar">
<label for="style">雞掰程度（0–10）</label>
<span id="styleLabel" class="pill">5</span>
</div>
<input id="style" type="range" min="0" max="10" step="1" value="5" />
<div class="hint" id="styleHint"></div>
</div>
<div class="bar">
<button id="send" class="btn">送出</button>
<span class="hint">模型：<code id="model">(server)</code></span>
</div>
</div>
<div class="card" style="margin-top:12px;">
<label>模型回覆</label>
<div id="out" class="out"></div>
</div>
</div>
<script>
  const style = document.getElementById('style');
  const styleLabel = document.getElementById('styleLabel');
  const styleHint = document.getElementById('styleHint');
  const msg = document.getElementById('msg');
  const out = document.getElementById('out');
  const send = document.getElementById('send');
  const modelEl = document.getElementById('model');

  const hints = [
    '0 超級客觀、完全不攻擊',
    '1 幾乎無情緒，婉轉提醒',
    '2 委婉但立場清楚',
    '3 輕微吐槽、仍禮貌',
    '4 直接一點、收著講',
    '5 中等尖銳，點出問題',
    '6 明顯不客氣，邏輯鞭笞',
    '7 情緒濃，帶刺但不下流',
    '8 火力全開、無髒話或輕微',
    '9 很兇，可能出現髒話',
    '10 極度尖銳（避免仇恨/歧視）'
  ];

  const updateLabels = () => {
    styleLabel.textContent = style.value;
    styleHint.textContent = hints[Number(style.value)] || '';
  };
  updateLabels();
  style.addEventListener('input', updateLabels);

  async function ask() {
    const text = (msg.value || '').trim();
    if (!text) { alert('請先輸入訊息'); return; }

    send.disabled = true; out.textContent = '思考中…';

    // 自動抓 base URL（包含 proxy path）
    const baseURL = window.location.origin + window.location.pathname.replace(/\/$/, '');
    try {
      const res = await fetch(baseURL + '/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, style: Number(style.value) })
      });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      modelEl.textContent = data.model || '(unknown)';
      out.textContent = data.reply || '(沒有回覆)';
    } catch (err) {
      out.textContent = '發生錯誤：' + err.message;
    } finally {
      send.disabled = false;
    }
  }

  send.addEventListener('click', ask);
  msg.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { ask(); }
  });
</script>

</body>
</html>
"""


def build_system_prompt(style_intensity: int) -> str:
    style_intensity = max(0, min(10, int(style_intensity)))
    tone_table = {
        0: "完全中性、專業、無任何攻擊性。",
        1: "極度婉轉、避免負面情緒。",
        2: "禮貌克制，清楚表達立場。",
        3: "稍微直白，輕微吐槽但不人身攻擊。",
        4: "直接表達不滿，仍保持尊重。",
        5: "中度尖銳，點出問題並提出具體要求。",
        6: "不太客氣，語氣帶刺但理性。",
        7: "明顯不悅，語言強硬，可用輕微髒話。",
        8: "非常強硬，允許短促髒話但避免低俗。",
        9: "極為尖銳，允許明顯髒話但避免人身貶損。",
        10:"最高強度：極度尖銳、可用粗話，但不得出現仇恨/歧視/暴力煽動。",
    }
    guardrails = (
        "無論風格強度為何，都必須避免仇恨言論、歧視、針對弱勢族群的人身攻擊、"
        "或任何違法、暴力煽動的內容。允許一般粗話與非仇恨的強烈語氣。"
    )
    return (
        f"你是一個能依照使用者要求調整語氣強度的中文助理。\n"
        f"目標語氣強度（0–10）：{style_intensity}。\n"
        f"語氣描述：{tone_table.get(style_intensity, tone_table[5])}\n"
        f"安全規範：{guardrails}\n"
        "回覆時請以繁體中文輸出；精煉、直接，並根據情境提供可行的具體建議或可直接貼上的文字草稿。"
    )


def decoding_params(style_intensity: int) -> dict:
    temp = 0.2 + (max(0, min(10, style_intensity)) / 10) * 0.8
    max_tokens = 256 if style_intensity <= 5 else 512
    return {"temperature": round(temp, 2), "max_tokens": max_tokens}


@app.route("/")
def index() -> Response:
    return Response(INDEX_HTML, mimetype="text/html; charset=utf-8")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True, silent=True) or {}
    user_msg = (data.get("message") or "").strip()
    style = int(data.get("style") or 0)
    if not user_msg:
        return jsonify({"error": "message is required"}), 400

    sys_prompt = build_system_prompt(style)
    decode = decoding_params(style)

    try:
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=decode["temperature"],
            max_tokens=decode["max_tokens"],
        )
        reply = resp.choices[0].message.content
        return jsonify({
            "reply": reply,
            "model": DEFAULT_MODEL,
            "temperature": decode["temperature"],
            "max_tokens": decode["max_tokens"],
            "style": style,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    print(f"啟動 Flask 在 port {port}，使用模型：{DEFAULT_MODEL}")
    app.run(host="0.0.0.0", port=port, debug=True)
