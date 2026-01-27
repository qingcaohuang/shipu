import streamlit as st
import streamlit.components.v1 as components
import json
import tempfile
import os
import shutil
import textwrap
import requests
import zipfile
import time
import re
import uuid
from openai import OpenAI, RateLimitError, APIError
from datetime import datetime, timedelta
from io import BytesIO
import difflib
import pandas as pd
from pathlib import Path

# 尝试导入 Google Generative AI SDK
try:
    import google.generativeai as genai
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False

# --- 1. 基础配置与视觉样式 ---
st.set_page_config(page_title="AI云端厨房实验室", layout="wide")

# 版本号定义
VERSION = "V1.5.0 (Local-Only)"
CONFIG_FILE = ".ai_configs.json"

# [新增] 注入 JS 拦截浏览器关闭/刷新事件，弹出原生确认对话框
components.html(
    """
    <script>
        window.parent.addEventListener('beforeunload', function (e) {
            e.preventDefault();
            e.returnValue = '';
        });
    </script>
    """,
    height=0,
    width=0
)

st.markdown(f"""
    <style>
    /* 全局字体与背景 - 温暖的米白色背景 */
    .stApp {{
        background-color: #F9F9F9;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }}
    
    /* 侧边栏样式 - 纯白背景加轻微阴影 */
    section[data-testid="stSidebar"] {{
        background-color: #FFFFFF;
        border-right: 1px solid #F0F0F0;
        box-shadow: 2px 0 10px rgba(0,0,0,0.02);
    }}

    /* 调整垂直间距 */
    div[data-testid="stVerticalBlock"] > div {{
        gap: 0.5rem !important;
    }}
    
    /* 按钮通用样式 - 扁平化、圆角 */
    div.stButton > button {{
        border-radius: 8px !important;
        border: 1px solid #E0E0E0 !important;
        background-color: #FFFFFF !important;
        color: #4A4A4A !important;
        font-weight: 500 !important;
        height: 40px !important;
        transition: all 0.2s ease-in-out !important;
    }}
    div.stButton > button:hover {{
        border-color: #FF9F43 !important;
        color: #FF9F43 !important;
        background-color: #FFF8F0 !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }}
    
    /* 主按钮 (Primary) - 橙色主题 */
    div.stButton > button[kind="primary"] {{
        background-color: #FF9F43 !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 2px 5px rgba(255, 159, 67, 0.3);
    }}
    div.stButton > button[kind="primary"]:hover {{
        background-color: #FF8C1A !important;
        box-shadow: 0 4px 10px rgba(255, 159, 67, 0.4);
    }}
    
    /* 导航按钮激活状态 */
    .nav-active button {{
        background-color: #FF9F43 !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 8px rgba(255, 159, 67, 0.2);
    }}
    
    /* 输入框优化 */
    .stTextInput input, .stTextArea textarea {{
        border-radius: 8px !important;
        border: 1px solid #E0E0E0 !important;
        padding: 10px !important;
    }}
    .stTextInput input:focus, .stTextArea textarea:focus {{
        border-color: #FF9F43 !important;
        box-shadow: 0 0 0 1px #FF9F43 !important;
    }}
    
    /* 详情页卡片样式 */
    .detail-card {
    .detail-card {{
        background-color: #FFFFFF;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
        border: 1px solid #F5F5F5;
    }
    }}
    
    .block-container {{ padding-top: 1.5rem !important; }}
    .version-text {{ color: #B0B0B0; font-size: 11px; text-align: center; margin-top: 20px; }}
    </style>
""", unsafe_allow_html=True)

# --- 模型管理辅助函数 ---
def load_ai_configs():
    default_configs = {
        "DeepSeek (默认)": {
            "key": "", 
            "url": "https://api.deepseek.com", 
            "model": "deepseek-chat"
        },
        "OpenAI (官方)": {
            "key": "", 
            "url": "https://api.openai.com/v1", 
            "model": "gpt-4o"
        },
        "Google Gemini": {
            "key": "", 
            "url": "https://generativelanguage.googleapis.com", 
            "model": "gemini-1.5-flash" 
        }
    }
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                saved = json.load(f)
                for k, v in saved.items():
                    if k in default_configs:
                        default_configs[k].update(v)
                    else:
                        default_configs[k] = v
        except: pass

    # [新增] 适配 Streamlit Cloud Secrets
    if hasattr(st, "secrets"):
        if "DEEPSEEK_KEY" in st.secrets:
            default_configs["DeepSeek (默认)"]["key"] = st.secrets["DEEPSEEK_KEY"]
        if "OPENAI_KEY" in st.secrets:
            default_configs["OpenAI (官方)"]["key"] = st.secrets["OPENAI_KEY"]
        if "GEMINI_KEY" in st.secrets:
            default_configs["Google Gemini"]["key"] = st.secrets["GEMINI_KEY"]
            
    return default_configs

def save_ai_configs(configs):
    with open(CONFIG_FILE, "w") as f:
        json.dump(configs, f)

# --- 本地（Excel）存储辅助函数 ---
def get_app_dir():
    return os.path.abspath(os.path.dirname(__file__))

# Excel 文件与字段定义
EXCEL_PATH = Path(get_app_dir()) / "data.xlsx"
SHEET_NAME = "Sheet1"
COLUMNS = ["日期", "菜名", "分类", "食材", "步骤", "小贴士", "故事"]

def ensure_excel(file_path=None):
    target = Path(file_path) if file_path else EXCEL_PATH
    if not target.exists():
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(columns=COLUMNS)
            df.to_excel(target, index=False, sheet_name=SHEET_NAME)
        except: pass

def load_local_recipes(file_path=None):
    target = Path(file_path) if file_path else EXCEL_PATH
    ensure_excel(target)
    try:
        df = pd.read_excel(target, sheet_name=SHEET_NAME, engine="openpyxl")
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = ""
        df = df[COLUMNS]
        records = df.fillna("").to_dict(orient="records")
        return records
    except Exception:
        return []

def save_to_local_full(records, file_path=None):
    target = Path(file_path) if file_path else EXCEL_PATH
    ensure_excel(target)
    df = pd.DataFrame(records or [], columns=COLUMNS)
    df.to_excel(target, index=False, sheet_name=SHEET_NAME)

def save_to_local_append(record, file_path=None):
    records = load_local_recipes(file_path)
    records.append({k: record.get(k, "") for k in COLUMNS})
    save_to_local_full(records, file_path)

def save_to_local_update(match_record, new_record, file_path=None):
    records = load_local_recipes(file_path)
    replaced = False
    for i, r in enumerate(records):
        if r.get('菜名') == match_record.get('菜名') and (('故事' not in match_record) or r.get('故事') == match_record.get('故事')):
            records[i] = {k: new_record.get(k, r.get(k, "")) for k in COLUMNS}
            replaced = True
            break
    if not replaced:
        records.append({k: new_record.get(k, "") for k in COLUMNS})
    save_to_local_full(records, file_path)

def save_to_local_delete(match_record, file_path=None):
    records = load_local_recipes(file_path)
    for i, r in enumerate(records):
        if r.get('菜名') == match_record.get('菜名') and (('故事' not in match_record) or r.get('故事') == match_record.get('故事')):
            records.pop(i)
            break
    save_to_local_full(records, file_path)

# --- 2. 初始化所有 Session State ---
if 'ai_configs' not in st.session_state: 
    st.session_state.ai_configs = load_ai_configs()
if 'current_config_name' not in st.session_state: 
    st.session_state.current_config_name = list(st.session_state.ai_configs.keys())[0]
if 'prev_selection' not in st.session_state:
    st.session_state.prev_selection = st.session_state.current_config_name

# 初始化配置输入框状态
if 'add_model_name' not in st.session_state: st.session_state.add_model_name = ""
if 'add_model_url' not in st.session_state: st.session_state.add_model_url = "https://api.deepseek.com"
if 'add_model_key' not in st.session_state: st.session_state.add_model_key = ""
if 'add_model_id' not in st.session_state: st.session_state.add_model_id = "deepseek-chat"

if 'pending_add_model_sync' in st.session_state:
    sync = st.session_state.pop('pending_add_model_sync') or {}
    st.session_state.add_model_name = sync.get('name', st.session_state.add_model_name)
    st.session_state.add_model_url = sync.get('url', st.session_state.add_model_url)
    st.session_state.add_model_key = sync.get('key', st.session_state.add_model_key)
    st.session_state.add_model_id = sync.get('id', st.session_state.add_model_id)

if 'last_gen' not in st.session_state: st.session_state.last_gen = None
if 'last_import' not in st.session_state: st.session_state.last_import = None
if 'active_recipe' not in st.session_state: st.session_state.active_recipe = None
if 'all_recipes_cache' not in st.session_state: st.session_state.all_recipes_cache = []
if 'reasoning_cache' not in st.session_state: st.session_state.reasoning_cache = None
if 'selected_style' not in st.session_state: st.session_state.selected_style = ""
if 'active_index' not in st.session_state: st.session_state.active_index = None
if 'nav_choice' not in st.session_state: st.session_state.nav_choice = "✨ AI生成"
if 'manage_mode' not in st.session_state: st.session_state.manage_mode = False

if 'current_excel_path' not in st.session_state:
    # [新增] 云端多用户隔离：默认使用带随机后缀的文件名，避免冲突
    st.session_state.current_excel_path = str((Path(get_app_dir()) / f"data_{str(uuid.uuid4())[:8]}.xlsx").resolve())

if not st.session_state.all_recipes_cache:
    try: st.session_state.all_recipes_cache = load_local_recipes()
    except Exception: st.session_state.all_recipes_cache = []

FONT_PATH = "SimHei.ttf" 

# --- 3. 核心逻辑函数 ---

def rerun_safe():
    try:
        if hasattr(st, 'experimental_rerun'): st.experimental_rerun()
        elif hasattr(st, 'rerun'): st.rerun()
        else: st.stop()
    except Exception: pass

def fetch_web_content(url):
    try:
        jina_url = f"https://r.jina.ai/{url}"
        response = requests.get(jina_url, timeout=15)
        return response.text if response.status_code == 200 else f"抓取失败 ({response.status_code})"
    except Exception as e: return f"连接出错: {e}"

def format_steps(steps):
    if not steps: return ""
    if isinstance(steps, str): steps = [s.strip() for s in steps.split('\n') if s.strip()]
    formatted = []
    import re
    for i, step in enumerate(steps):
        clean_step = re.sub(r'^[\d\.\-\s、第步骤]+[:：\s]*', '', step)
        if clean_step: formatted.append(f"{i+1}. {clean_step}")
    return "\n".join(formatted)

def test_google_models(api_key):
    """诊断函数：测试 Google API Key 并列出可用模型"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key.strip()}"
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            # 过滤出 generateContent 支持的模型
            chat_models = [m['name'].replace('models/', '') for m in models if 'generateContent' in m.get('supportedGenerationMethods', [])]
            return True, chat_models, "连接成功"
        else:
            return False, [], f"HTTP {response.status_code}: {response.text}"
    except Exception as e:
        return False, [], str(e)

def call_deepseek(config, mode="generate", **kwargs):
    # 构造 Prompts
    if mode == "generate":
        system_prompt = (
            "你是一位顶级大厨。请基于提供的材料和灵感创作专业食谱。\n"
            "用材灵活：基于用户提供的材料，【不仅限于】这些材料，根据专业需要自主补充配料、调料以达到最佳风味。\n"
            "步骤专业：制作步骤要突出前后逻辑合理性和有序性，必须有具体的、可操作的指导性（如火候、油温、面团状态、时间控制、手法细节等）。\n"
            "提示贴心：要针对用户容易忽略的细节或容易犯错的环节，加强提示和指导。可在操作步骤之外提供进一步的风味扩展思路或建议。注明用于内容生成的AI模型名称。\n"
            "分类按照最接近原则在下列选项中选择：家常、西餐、烘培、发酵物、饮品、川菜、蘸料、其他。\n"
            "输出必须是 JSON：{\"title\": \"名称\", \"story\": \"200字背景\", \"category\": \"分类\", "
            "\"ingredients_list\": [\"食材+克数\"], \"steps_list\": [\"详细动作\"], \"tips\": \"烹饪要点和秘诀\"}"
        )
        user_prompt = f"名称：{kwargs.get('name')}\n材料：{kwargs.get('ing')}\n风格：{kwargs.get('style')}\n要求：{kwargs.get('notes')}"
    else:
        system_prompt = (
            "你是一位食谱整理专家。从中提取食谱核心信息，并重构、润色成标准格式。\n"
            "即使缺失关键细节，也请根据专业常识进行补全，但对相关内容进行提示。\n"
            "有助于理解食谱的背景信息要进行收录，可以体现在tips。如果整理内容来源于网络，在提示中列出网址。注明用于内容生成的AI模型名称。\n"
            "分类按照最接近原则在下列选项中选择：家常、西餐、烘培、发酵物、饮品、川菜、蘸料、其他。\n"
            "内容要使用中文进行输出。\n"
            "输出必须是 JSON：{\"title\": \"名称\", \"story\": \"背景\", \"category\": \"分类\", "
            "\"ingredients_list\": [\"食材+克数\"], \"steps_list\": [\"步骤\"], \"tips\": \"烹饪要点\"}"
        )
        user_prompt = f"内容：\n{kwargs.get('raw_text')}"

    # 清洗数据
    raw_key = config.get('key', '').strip()
    raw_model = config.get('model', '').strip()
    raw_url = config.get('url', '').strip()

    is_google = "googleapis.com" in raw_url or "gemini" in raw_model.lower()
    
    # === 分支 A: Google 调用 ===
    if is_google:
        model_id = raw_model
        if model_id.startswith("models/"): model_id = model_id[7:]
        if not model_id: model_id = "gemini-1.5-flash"

        def process_gemini_content(text_content):
            if "```json" in text_content: text_content = text_content.split("```json")[1].split("```")[0]
            elif "```" in text_content: text_content = text_content.split("```")[1].split("```")[0]
            try:
                res = json.loads(text_content.strip())
                return {
                    "菜名": res.get('title') or "未命名", "分类": res.get('category') or "家常菜",
                    "故事": res.get('story') or "", "食材": "\n".join(res.get('ingredients_list') or []),
                    "步骤": format_steps(res.get('steps_list') or []), "小贴士": res.get('tips') or ""
                }, None
            except:
                st.error("JSON 解析失败，AI 返回了非标准格式。")
                return None, None

        # 优先 HTTP REST 方式 (最通用)
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={raw_key}"
        headers = {'Content-Type': 'application/json'}
        combined_text = f"System Instructions:\n{system_prompt}\n\nUser Request:\n{user_prompt}"
        payload = { "contents": [{ "parts": [{"text": combined_text}] }] }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.post(api_url, headers=headers, json=payload, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    if 'candidates' in data and data['candidates']:
                        text_content = data['candidates'][0]['content']['parts'][0]['text']
                        return process_gemini_content(text_content)
                    else:
                        st.error(f"Google 响应为空: {data}") # 安全政策拦截等
                        return None, None
                elif resp.status_code == 429:
                    wait = 20
                    if attempt < max_retries - 1:
                        st.warning(f"Google API 繁忙 (429)，{wait}秒后重试..."); time.sleep(wait); continue
                    else: st.error("Google API 配额耗尽。"); return None, None
                elif resp.status_code == 404:
                    st.error(f"Google Error 404: 找不到模型 '{model_id}'。请使用侧边栏的【测试连接】按钮查看您的 API Key 支持哪些模型。")
                    return None, None
                else:
                    st.error(f"Google HTTP Error {resp.status_code}: {resp.text}")
                    return None, None
            except Exception as e:
                st.error(f"请求失败: {e}"); return None, None
        return None, None

    # === 分支 B: OpenAI / DeepSeek SDK 调用 ===
    else:
        client = OpenAI(api_key=raw_key, base_url=raw_url)
        is_deepseek_r1 = "deepseek.com" in raw_url and kwargs.get('use_r1')
        model_name = "deepseek-reasoner" if is_deepseek_r1 else raw_model

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name, 
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                )
                content = response.choices[0].message.content
                reasoning = getattr(response.choices[0].message, 'reasoning_content', None)
                
                if "```json" in content: content = content.split("```json")[1].split("```")[0]
                elif "```" in content: content = content.split("```")[1].split("```")[0]
                
                res = json.loads(content.strip())
                return {
                    "菜名": res.get('title') or "未命名", "分类": res.get('category') or "家常菜",
                    "故事": res.get('story') or "", "食材": "\n".join(res.get('ingredients_list') or []),
                    "步骤": format_steps(res.get('steps_list') or []), "小贴士": res.get('tips') or ""
                }, reasoning
            except RateLimitError as e:
                err_str = str(e)
                wait_seconds = 20
                match = re.search(r'retry in (\d+(\.\d+)?)s', err_str)
                if match: wait_seconds = float(match.group(1)) + 1
                if attempt < max_retries - 1:
                    st.warning(f"触发频率限制，{wait_seconds:.1f} 秒后重试..."); time.sleep(wait_seconds); continue
                else: st.error(f"❌ API 繁忙或配额耗尽。"); return None, None
            except Exception as e:
                st.error(f"❌ 系统错误: {e}"); return None, None
    return None, None

def generate_pdf(recipe):
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    import re
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    try:
        pdfmetrics.registerFont(TTFont('SimHei', FONT_PATH))
        f_n = 'SimHei'
    except: f_n = 'Helvetica'

    def draw_text_block(text, x, y, max_w, line_height=15):
        p.setFont(f_n, 10)
        paragraphs = str(text).split('\n')
        for para in paragraphs:
            if not para.strip(): continue
            indent = 0
            match = re.match(r'^(\d+\.|•|\d+、)\s*', para)
            if match:
                prefix = match.group(0)
                indent = pdfmetrics.stringWidth(prefix, f_n, 10) + 2
            words = list(para)
            line = ""
            is_first_subline = True
            for char in words:
                test_line = line + char
                current_indent = 0 if is_first_subline else indent
                if pdfmetrics.stringWidth(test_line, f_n, 10) < (max_w - current_indent):
                    line = test_line
                else:
                    if y < 60:
                        p.showPage(); y = height - 50; p.setFont(f_n, 10)
                    draw_x = x if is_first_subline else x + indent
                    p.drawString(draw_x, y, line)
                    line = char; y -= line_height; is_first_subline = False
            if line:
                if y < 60: p.showPage(); y = height - 50; p.setFont(f_n, 10)
                draw_x = x if is_first_subline else x + indent
                p.drawString(draw_x, y, line); y -= (line_height + 2)
        return y

    p.setFont(f_n, 20); p.drawCentredString(width/2, height - 60, recipe['菜名'])
    p.setStrokeColor(colors.grey); p.setLineWidth(0.5); p.line(50, height - 75, width - 50, height - 75)
    curr_y = height - 100
    if recipe.get('故事'): pass
    p.setFillColor(colors.black); p.setFont(f_n, 12); p.drawString(50, curr_y, "【 材料准备 】"); curr_y -= 20
    curr_y = draw_text_block(recipe['食材'], 70, curr_y, width - 140); curr_y -= 10
    p.setFont(f_n, 12); p.drawString(50, curr_y, "【 操作步骤 】"); curr_y -= 20
    curr_y = draw_text_block(recipe['步骤'], 70, curr_y, width - 140, line_height=16)
    if recipe.get('小贴士'):
        curr_y -= 10; p.setFont(f_n, 12); p.drawString(50, curr_y, "【 贴士 】"); curr_y -= 20
        p.setFont(f_n, 10); curr_y = draw_text_block(recipe['小贴士'], 70, curr_y, width - 140)
    p.setFont(f_n, 8); p.setFillColor(colors.lightgrey)
    p.drawString(50, 30, f"Generated by Cook Lab {VERSION} | {datetime.now().strftime('%Y-%m-%d')}")
    p.showPage(); p.save(); buffer.seek(0)
    return buffer

# --- 4. 侧边栏布局 ---
side_col, main_col = st.columns([1.6, 4.5])

with side_col:
    st.markdown(f'<div style="text-align:center; font-weight:bold; font-size:1.2em; color:#FF9F43; margin-bottom:10px;">🍳 智汇厨房</div>', unsafe_allow_html=True)
    
    # [新增] 启动时的安全提示
    if 'safety_warning_shown' not in st.session_state:
        st.info(
            "📢 **数据安全提示**\n\n"
            "如果你是**新用户**，请记得在关闭程序前下载并保存数据；\n\n"
            "如果你是**老用户**，可以选择上传原有数据，并在关闭程序前下载并更新数据，否则新旧数据可能会出现覆盖等未知风险。\n\n"
            "👉 **数据的上传和下载请在【📚 菜谱目录 -> 管理】界面进行**。"
        )
        st.session_state.safety_warning_shown = True
    
    # sc1, sc2 = st.columns([4, 1]) # 移除状态灯列
    with st.container(border=True):
        with st.expander("🔑 AI 接口管理", expanded=False):
            model_options = list(st.session_state.ai_configs.keys())
            try: curr_idx = model_options.index(st.session_state.current_config_name)
            except: curr_idx = 0

            selected_name = st.selectbox("选择当前模型", model_options, index=curr_idx)
            
            if selected_name != st.session_state.prev_selection:
                st.session_state.current_config_name = selected_name
                st.session_state.prev_selection = selected_name
                cfg = st.session_state.ai_configs[selected_name]
                st.session_state.pending_add_model_sync = {"name": selected_name, "url": cfg.get("url", ""), "key": cfg.get("key", ""), "id": cfg.get("model", "")}
                st.rerun()

            st.divider()
            
            # --- 新增诊断工具 ---
            if "Google" in selected_name or "gemini" in st.session_state.add_model_id.lower():
                st.caption("Google 连接诊断")
                if st.button("🔍 测试连接 & 列出可用模型", use_container_width=True):
                    test_key = st.session_state.add_model_key if st.session_state.add_model_key else st.session_state.ai_configs.get(selected_name, {}).get("key", "")
                    if not test_key:
                        st.error("请先输入 API Key")
                    else:
                        with st.spinner("正在连接 Google API..."):
                            success, models, msg = test_google_models(test_key)
                            if success:
                                st.success(f"连接成功！您的 Key 支持以下模型：")
                                st.code("\n".join(models), language="text")
                                st.info("请从上方列表中复制一个模型名称填入下方的 'Model ID'。")
                            else:
                                st.error(f"连接失败: {msg}")

            st.caption("添加/编辑模型配置")
            col_preset1, col_preset2 = st.columns(2)
            with col_preset1:
                if st.button("OpenAI 预设", use_container_width=True):
                     st.session_state.pending_add_model_sync = {"name": "OpenAI (自定义)", "url": "https://api.openai.com/v1", "key": "", "id": "gpt-4o"}
                     st.rerun()
            with col_preset2:
                if st.button("Google 预设", use_container_width=True):
                     st.session_state.pending_add_model_sync = {"name": "Google Gemini", "url": "https://generativelanguage.googleapis.com", "key": "", "id": "gemini-2.5-flash"}
                     st.rerun()

            new_name = st.text_input("配置名称", key="add_model_name")
            new_url = st.text_input("API Base URL", key="add_model_url", help="Google: 使用默认即可，系统会自动使用原生SDK或REST")
            new_key = st.text_input("API Key", type="password", key="add_model_key")
            new_model = st.text_input("Model ID", key="add_model_id")
            
            b1, b2 = st.columns(2)
            with b1:
                if st.button("💾 保存配置", use_container_width=True):
                    if new_name and new_key:
                        st.session_state.ai_configs[new_name] = {"key": new_key, "url": new_url, "model": new_model}
                        save_ai_configs(st.session_state.ai_configs)
                        st.session_state.current_config_name = new_name
                        st.session_state.prev_selection = new_name
                        st.session_state.pending_add_model_sync = {"name": "", "url": "https://api.deepseek.com", "key": "", "id": "deepseek-chat"}
                        st.success("已保存")
                        st.rerun()
                    else: st.error("缺失名称/Key")
            with b2:
                if st.button("🗑️ 删除配置", use_container_width=True):
                    if len(st.session_state.ai_configs) > 1:
                        del st.session_state.ai_configs[st.session_state.current_config_name]
                        save_ai_configs(st.session_state.ai_configs)
                        st.session_state.current_config_name = list(st.session_state.ai_configs.keys())[0]
                        st.session_state.prev_selection = st.session_state.current_config_name
                        st.session_state.pending_add_model_sync = {"name": "", "url": "https://api.deepseek.com", "key": "", "id": "deepseek-chat"}
                        st.warning("已删除")
                        st.rerun()
                    else: st.error("需保留一项")

    # 2x2 网格导航
    st.markdown("###") # Spacer
    nav_config = [("✨ AI 生成", "✨ AI生成"), ("📥 AI 提取", "📥 AI提取"), ("📚 菜谱目录", "📚 菜谱目录"), ("🔍 全文搜索", "🔍 全文搜索")]
    for i in range(0, 4, 2):
        nc1, nc2 = st.columns(2)
        for idx, col in enumerate([nc1, nc2]):
            lbl, val = nav_config[i+idx]
            is_active = st.session_state.nav_choice == val
            with col:
                st.markdown(f'<div class="{"nav-active" if is_active else ""}">', unsafe_allow_html=True)
                if st.button(lbl, key=f"btn_{val}", use_container_width=True):
                    st.session_state.nav_choice = val
                    if val == "🔍 全文搜索": st.session_state.active_recipe = None
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown(f'<div class="version-text">Cook Lab {VERSION}</div>', unsafe_allow_html=True)
    
    current_ak_config = st.session_state.ai_configs.get(st.session_state.current_config_name, {"key": ""})

    if st.session_state.nav_choice == "✨ AI生成":
        if st.button("🆕 新创作", use_container_width=True):
            st.session_state.last_gen = None; st.session_state.reasoning_cache = None; st.session_state.gen_saved = False; st.rerun()
        an = st.text_input("菜名灵感", placeholder="输入菜名")
        ai = st.text_input("现有食材")
        tc = st.columns(4)
        tags = ["家常", "川菜", "西餐", "减脂", "烘焙", "饮品", "汤羹", "小吃"]
        for i, t in enumerate(tags):
            if tc[i%4].button(t, key=f"t_{t}"): st.session_state.selected_style = t
        cs = st.text_input("风格", value=st.session_state.selected_style)
        ai_notes = st.text_input("个性化要求", placeholder="如：少油、适合儿童...")
        ur = st.toggle("R1 思考 (DeepSeek专用)", value=True)
        st.markdown("###")
        if st.button("🪄 生成", type="primary", use_container_width=True):
            with st.spinner("撰写中..."):
                res, rsn = call_deepseek(current_ak_config, mode="generate", name=an, ing=ai, style=cs, notes=ai_notes, use_r1=ur)
                if res: st.session_state.last_gen = res; st.session_state.reasoning_cache = rsn; st.rerun()

    elif st.session_state.nav_choice == "📥 AI提取":
        if st.button("🆕 重新提取", use_container_width=True):
            st.session_state.last_import = None; st.session_state.reasoning_cache = None
            if "import_raw_input" in st.session_state: st.session_state["import_raw_input"] = ""; st.session_state.imp_saved = False
            st.rerun()
        ri = st.text_area("内容/链接", height=180, key="import_raw_input")
        if st.button("🧠 解析", type="primary", use_container_width=True):
            with st.spinner("识别中..."):
                txt = ri.strip()
                if txt.startswith("http"): txt = fetch_web_content(txt)
                res, rsn = call_deepseek(current_ak_config, mode="import", raw_text=txt, use_r1=True)
                if res: st.session_state.last_import = res; st.session_state.reasoning_cache = rsn; st.rerun()

    elif st.session_state.nav_choice == "📚 菜谱目录":
        colr, colm = st.columns([1,1])
        with colr:
            if st.button("🔄 刷新目录", use_container_width=True):
                try:
                    st.session_state.all_recipes_cache = load_local_recipes(st.session_state.current_excel_path)
                    st.toast(f"已刷新，共 {len(st.session_state.all_recipes_cache)} 条")
                    st.session_state.all_recipes_cache = load_local_recipes(st.session_state.current_excel_path)
                    st.toast(f"已刷新，共 {len(st.session_state.all_recipes_cache)} 条")
                except Exception as e: st.warning(f"刷新失败: {e}")
        with colm:
            if st.button("🗂️ 食谱管理", use_container_width=True):
                st.session_state.manage_mode = not st.session_state.manage_mode
                if st.session_state.manage_mode:
                    st.session_state.active_recipe = None
                    st.session_state.manage_view = False
                else: st.session_state.manage_view = False
                rerun_safe()

        if st.session_state.manage_mode:
            # [修改] 简化为本地数据上传/下载模式，隐藏路径细节
            with st.expander("📂 数据存取 (本地 <-> 云端)", expanded=True):
                st.caption("当前操作的是云端临时数据。您可以上传本地 Excel 恢复工作，或将当前数据下载到本地保存。")
                
                col_up, col_down = st.columns(2)
                with col_up:
                    up_file = st.file_uploader("📤 上传本地 Excel (覆盖当前)", type=["xlsx"], key="manage_uploader")
                    if up_file:
                        if st.button("⚠️ 确认覆盖并加载", use_container_width=True):
                            target_p = st.session_state.current_excel_path
                            with open(target_p, "wb") as f:
                                f.write(up_file.getbuffer())
                            st.session_state.all_recipes_cache = load_local_recipes(target_p)
                            st.toast(f"已加载数据，共 {len(st.session_state.all_recipes_cache)} 条")
                            time.sleep(1); st.rerun()
                
                with col_down:
                    st.write("⬇️ 保存数据到本地")
                    st.caption("下载至本机【下载】目录")
                    target_p = st.session_state.current_excel_path
                    if os.path.exists(target_p):
                        with open(target_p, "rb") as f:
                            st.download_button("💾 下载 Excel 文件", data=f, file_name=f"recipes_{datetime.now().strftime('%Y%m%d')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
                    else:
                        st.info("暂无数据")

            records_all = st.session_state.all_recipes_cache or []
            categories = ["全部"] + list(dict.fromkeys([ (r.get('分类') or '未分类') for r in records_all ]))
            if not categories: st.info("无食谱。")
            else:
                sel_cat = st.selectbox("选择分类", options=categories)
                if sel_cat == "全部": filtered = list(enumerate(records_all))
                else: filtered = [(idx, r) for idx, r in enumerate(records_all) if (r.get('分类') or '未分类') == sel_cat]
                
                if not filtered: st.info("无数据。")
                else:
                    st.caption(f"共 {len(filtered)} 项。")
                    cols_per_row = 4
                    checked_indices = []
                    for i in range(0, len(filtered), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j in range(cols_per_row):
                            if i + j >= len(filtered): break
                            idx_global, rec = filtered[i + j]
                            label = f"{rec.get('菜名','未命名')}  ({rec.get('日期','')})"
                            if cols[j].checkbox(label, key=f"manage_chk_{idx_global}"):
                                checked_indices.append(idx_global)
                            if cols[j].button("查看", key=f"view_{idx_global}", use_container_width=True):
                                st.session_state.active_recipe = rec
                                st.session_state.active_index = idx_global + 2
                                st.session_state.manage_view = True
                                rerun_safe()
                    if checked_indices:
                        a1, a2 = st.columns(2)
                        with a1:
                            if st.button("🗑️ 删除所选", key='action_delete_selected'):
                                st.session_state.pending_action = {"type": "delete_selected", "indices": checked_indices.copy(), "count": len(checked_indices)}
                        with a2:
                            if st.button("📤 导出PDF", key='action_export_selected'):
                                st.session_state.pending_action = {"type": "export_selected", "indices": checked_indices.copy(), "count": len(checked_indices)}

                    if st.session_state.get('pending_action'):
                        pa = st.session_state.pending_action
                        st.warning(f"确认操作 {pa.get('count',0)} 个对象？")
                        c1, c2 = st.columns([1,1])
                        with c1:
                            if st.button("确认", key='pending_confirm'):
                                if pa.get('type') == 'delete_selected':
                                    records = load_local_recipes()
                                    to_remove = set(pa.get('indices', []))
                                    new_records = [r for i, r in enumerate(records) if i not in to_remove]
                                    save_to_local_full(new_records)
                                    st.success("已删除。"); st.session_state.pending_action = None
                                    st.session_state.all_recipes_cache = load_local_recipes()
                                    st.session_state.manage_view = False; rerun_safe()
                                elif pa.get('type') == 'export_selected':
                                    zipbuf = BytesIO()
                                    with zipfile.ZipFile(zipbuf, mode='w') as zf:
                                        records = load_local_recipes()
                                        for idx in pa.get('indices', []):
                                            if 0 <= idx < len(records):
                                                rec = records[idx]
                                                pdfbuf = generate_pdf(rec)
                                                zf.writestr(f"{rec.get('菜名','recipe')}_{idx}.pdf", pdfbuf.getvalue())
                                    zipbuf.seek(0)
                                    st.session_state.prepared_zip_bytes = zipbuf.getvalue()
                                    st.session_state.prepared_zip_filename = f"PDF导出_{datetime.now().strftime('%Y%m%d')}.zip"
                                    st.success("ZIP 已就绪。"); st.session_state.pending_action = None
                        with c2:
                            if st.button("取消", key='pending_cancel'):
                                st.session_state.pending_action = None; rerun_safe()

                    if st.session_state.get('prepared_zip_bytes'):
                        st.download_button(
                            label="⬇️ 下载导出文件 (ZIP)",
                            data=st.session_state.prepared_zip_bytes,
                            file_name=st.session_state.get('prepared_zip_filename', "recipes_export.zip"),
                            mime="application/zip",
                            key='download_zip_btn'
                        )

        if not st.session_state.manage_mode:
            itms = st.session_state.all_recipes_cache
            for i in range(0, len(itms), 2):
                cl1, cl2 = st.columns(2)
                r1 = itms[i]
                if cl1.button(f"{r1.get('菜名')[:12]}", key=f"l_{i}", use_container_width=True):
                    st.session_state.active_recipe = r1; st.session_state.active_index = i + 2; st.rerun()
                if i + 1 < len(itms):
                    r2 = itms[i+1]
                    if cl2.button(f"{r2.get('菜名')[:12]}", key=f"l_{i+1}", use_container_width=True):
                        st.session_state.active_recipe = r2; st.session_state.active_index = i + 3; st.rerun()

    elif st.session_state.nav_choice == "🔍 全文搜索":
        kw = st.text_input("关键词", placeholder="搜索...")
        if kw and st.session_state.all_recipes_cache:
            rlts = []
            for i, r in enumerate(st.session_state.all_recipes_cache):
                txt = f"{r['菜名']}{r['食材']}{r['分类']}".lower()
                score = difflib.SequenceMatcher(None, kw.lower(), txt).ratio()
                if kw.lower() in txt: score += 0.5
                if score > 0.1: rlts.append((score, i, r))
            rlts.sort(key=lambda x: x[0], reverse=True)
            for i in range(0, len(rlts), 2):
                sc1, sc2 = st.columns(2)
                _, idx1, r1 = rlts[i]
                if sc1.button(f"🔍 {r1.get('菜名')[:12]}", key=f"s_{idx1}", use_container_width=True):
                    st.session_state.active_recipe = r1; st.session_state.active_index = idx1 + 2; st.rerun()
                if i + 1 < len(rlts):
                    _, idx2, r2 = rlts[i+1]
                    if sc2.button(f"🔍 {r2.get('菜名')[:12]}", key=f"s_{idx2}", use_container_width=True):
                        st.session_state.active_recipe = r2; st.session_state.active_index = idx2 + 2; st.rerun()

# --- 5. 主界面内容 ---
with main_col:
    if st.session_state.nav_choice == "✨ AI生成" and st.session_state.last_gen:
        r = st.session_state.last_gen
        st.subheader(f"✨ {r['菜名']}")
        if st.session_state.reasoning_cache:
            with st.expander("AI 思路"): st.code(st.session_state.reasoning_cache)
        with st.form("gen_f"):
            cn = st.text_input("菜名", r['菜名'])
            cat = st.text_input("分类", r.get('分类', '家常菜'))
            ci = st.text_area("食材", r['食材'], height=130)
            cs_steps = st.text_area("步骤", r['步骤'], height=220)
            ct = st.text_area("贴士", r['小贴士'], height=80)
            if st.form_submit_button("🚀 录入云端临时库", use_container_width=True):
                record = {"日期": datetime.now().strftime("%Y-%m-%d"), "菜名": cn, "分类": cat, "食材": ci, "步骤": cs_steps, "小贴士": ct, "故事": r['故事']}
                save_to_local_append(record, file_path=st.session_state.current_excel_path)
                st.session_state.gen_saved = True
                st.toast("已录入云端临时库", icon="✅")

        if st.session_state.get('gen_saved'):
            st.success("✅ 已保存至云端临时库。\n\n请前往 **【📚 菜谱目录 -> 管理】** 界面下载备份数据。")

    elif st.session_state.nav_choice == "📥 AI提取" and st.session_state.last_import:
        r = st.session_state.last_import
        st.subheader(f"📥 {r['菜名']}")
        if st.session_state.reasoning_cache:
            with st.expander("AI 解析"): st.code(st.session_state.reasoning_cache)
        with st.form("imp_f"):
            cn = st.text_input("菜名", r['菜名'])
            cat = st.text_input("分类", r.get('分类', '家常菜'))
            ci = st.text_area("食材", r['食材'], height=130)
            cs_steps = st.text_area("步骤", r['步骤'], height=220)
            ct = st.text_area("贴士", r['小贴士'], height=80)
            if st.form_submit_button("🚀 录入云端临时库", use_container_width=True):
                record = {"日期": datetime.now().strftime("%Y-%m-%d"), "菜名": cn, "分类": cat, "食材": ci, "步骤": cs_steps, "小贴士": ct, "故事": r['故事']}
                save_to_local_append(record, file_path=st.session_state.current_excel_path)
                st.session_state.imp_saved = True
                st.toast("已录入云端临时库", icon="✅")

        if st.session_state.get('imp_saved'):
            st.success("✅ 已保存至云端临时库。\n\n请前往 **【📚 菜谱目录 -> 管理】** 界面下载备份数据。")

    elif st.session_state.nav_choice in ["📚 菜谱目录", "🔍 全文搜索"] and st.session_state.active_recipe and (not st.session_state.manage_mode or st.session_state.manage_view):
        r = st.session_state.active_recipe
        v, e = st.columns([2, 1])
        with v:
            # 使用 HTML/CSS 渲染卡片式详情
            st.markdown(f"""
            <div class="detail-card">
                <div style="font-size:28px; font-weight:bold; color:#2C3E50; margin-bottom:10px; border-bottom:2px solid #FF9F43; padding-bottom:10px;">
                    {r['菜名']}
                </div>
                <div style="color:#666; font-style:italic; margin-bottom:20px;">{r.get('故事', '')}</div>
                <div style="font-size:18px; font-weight:bold; color:#FF9F43; margin-bottom:8px;">🥘 食材清单</div>
                <div style="white-space: pre-wrap; line-height:1.6; color:#444; margin-bottom:20px;">{r['食材']}</div>
                <div style="font-size:18px; font-weight:bold; color:#FF9F43; margin-bottom:8px;">👨‍🍳 制作步骤</div>
                <div style="white-space: pre-wrap; line-height:1.6; color:#444;">{r['步骤']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if r.get('小贴士'): 
                st.info(f"💡 **大厨贴士**：\n\n{r['小贴士']}")
                
        with e:
            st.caption("📝 编辑模式")
            un = st.text_input("菜名", r['菜名'])
            uc = st.text_input("分类", r.get('分类',''))
            ui = st.text_area("原料", r['食材'], height=110)
            us = st.text_area("方法", r['步骤'], height=180)
            ut = st.text_area("备注", r.get('小贴士',''), height=80)
            cur = {"菜名": un, "食材": ui, "步骤": us, "小贴士": ut, "分类": uc, "故事": r.get('故事','')}
            if st.button("💾 保存更新", use_container_width=True):
                match = {"菜名": r.get('菜名'), "故事": r.get('故事','')}
                new_rec = {"日期": datetime.now().strftime("%Y-%m-%d"), "菜名": un, "分类": uc, "食材": ui, "步骤": us, "小贴士": ut, "故事": r.get('故事','')}
                save_to_local_update(match, new_rec, file_path=st.session_state.current_excel_path)
                st.success("本地已更新。")
                st.session_state.active_recipe.update(cur); st.rerun()
            st.divider()
            st.download_button("📥 PDF", data=generate_pdf(cur), file_name=f"{un}.pdf", mime="application/pdf", use_container_width=True)
            if st.button("🗑️ 彻底删除", type="primary", use_container_width=True):
                save_to_local_delete(r, file_path=st.session_state.current_excel_path)
                st.success("已删除。")
                st.session_state.all_recipes_cache = []; st.session_state.active_recipe = None; st.rerun()
    else:
        st.title("👋 私房云端厨房")
        st.info("← 请从左侧选择功能模块开始。")