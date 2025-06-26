import streamlit as st
import json
import toml
import os
from pathlib import Path

# Google GenAI SDKのインポート
try:
    import google.genai as genai
    from google.genai import types
    NEW_SDK = True
except ImportError:
    try:
        import google.generativeai as genai
        NEW_SDK = False
        st.warning("古いライブラリを使用しています。pip install google-genai で新しいライブラリに更新してください。")
    except ImportError:
        st.error("Google GenAI ライブラリがインストールされていません。pip install google-genai を実行してください。")
        st.stop()

# Vertex AI用のインポート
try:
    from google.auth import default
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False

# Load secrets
secrets_path = Path(__file__).parent / "secrets.toml"
if secrets_path.exists():
    secrets = toml.load(secrets_path)
    google_api_key = secrets.get("google_api_key", "")
    vertex_ai_project_id = secrets.get("vertex_ai_project_id", "")
    vertex_ai_location = secrets.get("vertex_ai_location", "us-central1")
    default_model = secrets.get("default_model", "gemini-2.0-flash-exp")
else:
    # 環境変数から取得
    google_api_key = os.environ.get("GOOGLE_API_KEY", "")
    vertex_ai_project_id = os.environ.get("VERTEX_AI_PROJECT_ID", "")
    vertex_ai_location = os.environ.get("VERTEX_AI_LOCATION", "us-central1")
    default_model = os.environ.get("DEFAULT_MODEL", "gemini-2.0-flash-exp")

# AI プロバイダーオプション
default_model_options = [
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

# Vertex AI プロバイダーオプション
vertex_model_options = [
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash-002",
    "gemini-1.5-pro-002"
]

# Load correction prompts
def load_prompt(filename):
    prompt_path = Path(__file__).parent / filename
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

# Load all prompts (パターン類似度校正プロンプトを除く)
try:
    tonmana_prompt = load_prompt("トンマナ校正プロンプト.txt")
    japanese_prompt = load_prompt("日本語校正プロンプト.txt")
    logic_prompt = load_prompt("ロジック校正プロンプト.txt")
    comprehensive_prompt = load_prompt("総合校正プロンプト.txt")
except Exception as e:
    st.error(f"Error loading prompts: {e}")
    st.stop()

# Google AI/Vertex AI設定関数
def setup_ai_provider(provider, model_name, api_key=None, project_id=None, location=None):
    """AI プロバイダーを設定する"""
    try:
        if provider == "Google AI":
            if NEW_SDK:
                client = genai.Client(api_key=api_key)
            else:
                genai.configure(api_key=api_key)
                client = None
            return client, model_name
        elif provider == "Vertex AI" and VERTEX_AI_AVAILABLE:
            if NEW_SDK:
                client = genai.Client(
                    vertexai=True,
                    project=project_id,
                    location=location
                )
            else:
                genai.configure(project=project_id, location=location)
                client = None
            return client, model_name
        else:
            st.error(f"プロバイダー {provider} はサポートされていません")
            return None, None
    except Exception as e:
        st.error(f"AI プロバイダーの設定に失敗しました: {e}")
        return None, None

# Function to call Gemini API
def call_gemini(prompt, user_message, provider, model_name, api_key=None, project_id=None, location=None):
    """Gemini APIを呼び出す"""
    try:
        client, model = setup_ai_provider(provider, model_name, api_key, project_id, location)
        
        # メッセージを構築
        full_message = f"{prompt}\n\n{user_message}"
        
        if NEW_SDK:
            if client:
                response = client.models.generate_content(
                    model=model,
                    contents=full_message,
                    config=types.GenerateContentConfig(
                        max_output_tokens=1000,
                        temperature=0.1,
                    )
                )
                return response.text
            else:
                st.error("クライアントの初期化に失敗しました")
                return None
        else:
            model_obj = genai.GenerativeModel(model)
            response = model_obj.generate_content(
                full_message,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1000,
                    temperature=0.1,
                )
            )
            return response.text
            
    except Exception as e:
        st.error(f"Gemini API呼び出しエラー: {e}")
        return None

# Function to parse JSON from Claude's response
def parse_json_response(response):
    try:
        # Find JSON in the response
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            return json.loads(json_str)
        else:
            st.warning("Could not find JSON in the response")
            return None
    except Exception as e:
        st.error(f"Error parsing JSON: {e}")
        st.code(response)  # Display the raw response for debugging
        return None

# Main app
st.title("mimiko校正システム")

# Initialize variables with default values
api_key_input = google_api_key
project_id_input = vertex_ai_project_id
location_input = vertex_ai_location

# Settings section
with st.expander("⚙️ 設定", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        # プロバイダー選択
        ai_provider = st.selectbox(
            "🤖 AIプロバイダー",
            ["Google AI", "Vertex AI"] if VERTEX_AI_AVAILABLE else ["Google AI"],
            key="ai_provider"
        )
        
        # モデル選択
        if ai_provider == "Google AI":
            available_models = default_model_options
        else:
            available_models = vertex_model_options
            
        selected_model = st.selectbox(
            "🎯 モデル",
            available_models,
            index=0 if default_model not in available_models else available_models.index(default_model),
            key="selected_model"
        )
    
    with col2:
        if ai_provider == "Google AI":
            api_key_input = st.text_input(
                "🔑 Google API Key",
                value=google_api_key,
                type="password",
                help="Google AI Studio で取得したAPIキーを入力してください"
            )
        else:
            project_id_input = st.text_input(
                "📁 Project ID",
                value=vertex_ai_project_id,
                help="Google Cloud プロジェクトIDを入力してください"
            )
            location_input = st.text_input(
                "🌍 Location",
                value=vertex_ai_location,
                help="Vertex AI のリージョンを指定してください"
            )

# Input section
st.header("入力")
user_question = st.text_area("ユーザーからの質問", height=100)

# キーワードカテゴリ選択（最大4つまで）
st.subheader("使用されたキーワードカテゴリ")
col1, col2 = st.columns(2)

with col1:
    category1 = st.selectbox("カテゴリ1", ["なし", "ハウス", "サイン", "天体", "エレメント", "MP軸", "タロット"], key="cat1")
    if category1 != "なし":
        keyword1 = st.text_input("キーワード1", placeholder="例: 第1ハウス", key="kw1")
    else:
        keyword1 = ""

with col2:
    category2 = st.selectbox("カテゴリ2", ["なし", "ハウス", "サイン", "天体", "エレメント", "MP軸", "タロット"], key="cat2")
    if category2 != "なし":
        keyword2 = st.text_input("キーワード2", placeholder="例: 牡羊座", key="kw2")
    else:
        keyword2 = ""

col3, col4 = st.columns(2)

with col3:
    category3 = st.selectbox("カテゴリ3", ["なし", "ハウス", "サイン", "天体", "エレメント", "MP軸", "タロット"], key="cat3")
    if category3 != "なし":
        keyword3 = st.text_input("キーワード3", placeholder="例: 太陽", key="kw3")
    else:
        keyword3 = ""

with col4:
    category4 = st.selectbox("カテゴリ4", ["なし", "ハウス", "サイン", "天体", "エレメント", "MP軸", "タロット"], key="cat4")
    if category4 != "なし":
        keyword4 = st.text_input("キーワード4", placeholder="例: 火", key="kw4")
    else:
        keyword4 = ""

ai_answer = st.text_area("AI占い師の回答", height=150)

# Initialize session state
if 'corrections' not in st.session_state:
    st.session_state.corrections = {}
if 'tonmana_problems' not in st.session_state:
    st.session_state.tonmana_problems = []
if 'japanese_improvements' not in st.session_state:
    st.session_state.japanese_improvements = []
if 'logic_improvements' not in st.session_state:
    st.session_state.logic_improvements = []
if 'correction_done' not in st.session_state:
    st.session_state.correction_done = False
if 'user_input' not in st.session_state:
    st.session_state.user_input = {"question": "", "keywords": [], "answer": ""}

# Process button
if st.button("校正を実行") or st.session_state.correction_done:
    if not user_question or not ai_answer:
        st.error("質問と回答を入力してください")
    else:
        # キーワード情報の整理
        keywords = []
        if category1 != "なし" and keyword1:
            keywords.append(f"{category1}: {keyword1}")
        if category2 != "なし" and keyword2:
            keywords.append(f"{category2}: {keyword2}")
        if category3 != "なし" and keyword3:
            keywords.append(f"{category3}: {keyword3}")
        if category4 != "なし" and keyword4:
            keywords.append(f"{category4}: {keyword4}")
        
        # Save user input to session state
        st.session_state.user_input = {
            "question": user_question,
            "keywords": keywords,
            "answer": ai_answer
        }
        
        # Set correction_done flag to true
        st.session_state.correction_done = True
        
        # 設定の準備
        if ai_provider == "Google AI":
            current_api_key = api_key_input
            current_project_id = None
            current_location = None
        else:
            current_api_key = None
            current_project_id = project_id_input
            current_location = location_input

        # Only call APIs if not already done
        if 'tonmana_result' not in st.session_state:
            with st.spinner("トンマナ校正中..."):
                # 1. トンマナ校正
                tonmana_message = f"""##QUESTION##
{user_question}

##KEYWORDS##
{', '.join(keywords) if keywords else 'なし'}

##ANSWER_CAND##
{ai_answer}
"""
                tonmana_result = call_gemini(
                    tonmana_prompt, 
                    tonmana_message,
                    ai_provider,
                    selected_model,
                    current_api_key,
                    current_project_id,
                    current_location
                )
                if tonmana_result:
                    tonmana_json = parse_json_response(tonmana_result)
                    if tonmana_json:
                        st.session_state.tonmana_result = tonmana_result
                        st.session_state.tonmana_json = tonmana_json
                        
                        # Store in session state
                        st.session_state.corrections["tonmana"] = {
                            "score": tonmana_json.get('style_score', 'N/A'),
                            "comment": tonmana_json.get('comment', 'N/A'),
                            "problems": tonmana_json.get('problems', [])
                        }
        
        if 'japanese_result' not in st.session_state:
            with st.spinner("日本語校正中..."):
                # 2. 日本語校正
                japanese_message = ai_answer
                japanese_result = call_gemini(
                    japanese_prompt,
                    japanese_message,
                    ai_provider,
                    selected_model,
                    current_api_key,
                    current_project_id,
                    current_location
                )
                if japanese_result:
                    japanese_json = parse_json_response(japanese_result)
                    if japanese_json:
                        st.session_state.japanese_result = japanese_result
                        st.session_state.japanese_json = japanese_json
                        
                        # Store in session state
                        st.session_state.corrections["japanese"] = {
                            "score": japanese_json.get('score', 'N/A'),
                            "improvements": japanese_json.get('improvements', [])
                        }
        
        if 'logic_result' not in st.session_state and logic_prompt.strip():
            with st.spinner("ロジック校正中..."):
                # 3. ロジック校正
                logic_message = f"""質問: {user_question}
使用キーワード: {', '.join(keywords) if keywords else 'なし'}
回答: {ai_answer}
"""
                logic_result = call_gemini(
                    logic_prompt,
                    logic_message,
                    ai_provider,
                    selected_model,
                    current_api_key,
                    current_project_id,
                    current_location
                )
                if logic_result:
                    logic_json = parse_json_response(logic_result)
                    if logic_json:
                        st.session_state.logic_result = logic_result
                        st.session_state.logic_json = logic_json
                        
                        # Store in session state
                        st.session_state.corrections["logic"] = {
                            "score": logic_json.get('score', 'N/A'),
                            "improvements": logic_json.get('improvements', [])
                        }
        
        # Calculate total score
        total_score = 0
        max_score = 15
        valid_scores = 0
        
        for correction_type in ["tonmana", "japanese", "logic"]:
            if correction_type in st.session_state.corrections:
                score = st.session_state.corrections[correction_type]["score"]
                if isinstance(score, (int, float)) and score != 'N/A':
                    total_score += score
                    valid_scores += 1
        
        # Display total score
        if valid_scores > 0:
            st.header(f"総合スコア: {total_score}/{max_score}点")
        
        # Display results
        # 1. トンマナ校正結果
        if 'tonmana_json' in st.session_state:
            st.subheader("トンマナ校正結果")
            tonmana_json = st.session_state.tonmana_json
            st.write(f"スコア: {tonmana_json.get('style_score', 'N/A')}/5")
            st.write(f"コメント: {tonmana_json.get('comment', 'N/A')}")
            
            problems = tonmana_json.get('problems', [])
            if problems:
                st.write("改善点:")
                for i, problem in enumerate(problems):
                    # Create a unique key for each checkbox
                    checkbox_key = f"tonmana_cb_{i}"
                    
                    # Check if the problem is in the selected problems list
                    is_selected = problem in st.session_state.tonmana_problems
                    
                    # Display checkbox
                    checked = st.checkbox(problem, key=checkbox_key, value=is_selected)
                    
                    # Update selected problems list based on checkbox state
                    if checked and problem not in st.session_state.tonmana_problems:
                        st.session_state.tonmana_problems.append(problem)
                    elif not checked and problem in st.session_state.tonmana_problems:
                        st.session_state.tonmana_problems.remove(problem)
            else:
                st.write("改善点はありません")
        
        # 2. 日本語校正結果
        if 'japanese_json' in st.session_state:
            st.subheader("日本語校正結果")
            japanese_json = st.session_state.japanese_json
            st.write(f"スコア: {japanese_json.get('score', 'N/A')}/5")
            
            improvements = japanese_json.get('improvements', [])
            if improvements:
                st.write("改善点:")
                for i, improvement in enumerate(improvements):
                    # Create a unique key for each checkbox
                    checkbox_key = f"japanese_cb_{i}"
                    
                    # Check if the improvement is in the selected improvements list
                    is_selected = improvement in st.session_state.japanese_improvements
                    
                    # Display checkbox
                    checked = st.checkbox(improvement, key=checkbox_key, value=is_selected)
                    
                    # Update selected improvements list based on checkbox state
                    if checked and improvement not in st.session_state.japanese_improvements:
                        st.session_state.japanese_improvements.append(improvement)
                    elif not checked and improvement in st.session_state.japanese_improvements:
                        st.session_state.japanese_improvements.remove(improvement)
            else:
                st.write("改善点はありません")
        
        # 3. ロジック校正結果
        if 'logic_json' in st.session_state:
            st.subheader("ロジック校正結果")
            logic_json = st.session_state.logic_json
            st.write(f"スコア: {logic_json.get('score', 'N/A')}/5")
            
            improvements = logic_json.get('improvements', [])
            if improvements:
                st.write("改善点:")
                for i, improvement in enumerate(improvements):
                    # Create a unique key for each checkbox
                    checkbox_key = f"logic_cb_{i}"
                    
                    # Check if the improvement is in the selected improvements list
                    is_selected = improvement in st.session_state.logic_improvements
                    
                    # Display checkbox
                    checked = st.checkbox(improvement, key=checkbox_key, value=is_selected)
                    
                    # Update selected improvements list based on checkbox state
                    if checked and improvement not in st.session_state.logic_improvements:
                        st.session_state.logic_improvements.append(improvement)
                    elif not checked and improvement in st.session_state.logic_improvements:
                        st.session_state.logic_improvements.remove(improvement)
            else:
                st.write("改善点はありません")
        elif logic_prompt.strip():
            st.warning("ロジック校正プロンプトが空です。この校正はスキップされます。")
        
        # Show button to proceed to comprehensive correction
        st.session_state.show_comprehensive_button = True

# Reset button
if st.button("リセット"):
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # Rerun the app
    st.rerun()

# Comprehensive correction button
if st.session_state.get("show_comprehensive_button", False):
    st.header("総合校正")
    if st.button("選択した改善点で総合校正を実行") or 'comprehensive_result' in st.session_state:
        if 'comprehensive_result' not in st.session_state:
            with st.spinner("総合校正中..."):
                # 設定の準備（comprehensive correction用）
                if ai_provider == "Google AI":
                    current_api_key = api_key_input
                    current_project_id = None
                    current_location = None
                else:
                    current_api_key = None
                    current_project_id = project_id_input
                    current_location = location_input
                # Prepare selected improvements
                selected_improvements = {
                    "トンマナ校正": st.session_state.get("tonmana_problems", []),
                    "日本語校正": st.session_state.get("japanese_improvements", []),
                    "ロジック校正": st.session_state.get("logic_improvements", [])
                }
                
                # Prepare scores
                scores = {}
                if "tonmana" in st.session_state.corrections:
                    scores["トンマナ校正"] = st.session_state.corrections["tonmana"]["score"]
                if "japanese" in st.session_state.corrections:
                    scores["日本語校正"] = st.session_state.corrections["japanese"]["score"]
                if "logic" in st.session_state.corrections:
                    scores["ロジック校正"] = st.session_state.corrections["logic"]["score"]
                
                # Create message for comprehensive correction
                comprehensive_message = f"""AI占い師の回答:
{st.session_state.user_input["answer"]}

使用されたキーワード:
{', '.join(st.session_state.user_input["keywords"]) if st.session_state.user_input["keywords"] else 'なし'}

各校正AIの採点結果:
"""
                for correction_type, score in scores.items():
                    comprehensive_message += f"{correction_type}: {score}/5\n"
                
                comprehensive_message += "\n選択された改善点:\n"
                for correction_type, improvements in selected_improvements.items():
                    if improvements:
                        comprehensive_message += f"\n{correction_type}:\n"
                        for i, improvement in enumerate(improvements):
                            comprehensive_message += f"{i+1}. {improvement}\n"
                
                # Call Gemini for comprehensive correction
                comprehensive_result = call_gemini(
                    comprehensive_prompt,
                    comprehensive_message,
                    ai_provider,
                    selected_model,
                    current_api_key,
                    current_project_id,
                    current_location
                )
                if comprehensive_result:
                    # Save the result to session state
                    st.session_state.comprehensive_result = comprehensive_result
        
        # Display comprehensive correction result
        if 'comprehensive_result' in st.session_state:
            st.subheader("総合校正結果")
            st.write(st.session_state.comprehensive_result)