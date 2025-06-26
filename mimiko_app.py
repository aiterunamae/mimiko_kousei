import streamlit as st
import json
import toml
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import io

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
# Streamlit Cloudの場合
if hasattr(st, "secrets"):
    try:
        # 新しい形式に対応
        vertex_ai_project_id = st.secrets["api"]["vertex_project"] if "api" in st.secrets and "vertex_project" in st.secrets["api"] else ""
        vertex_ai_location = st.secrets["api"]["vertex_location"] if "api" in st.secrets and "vertex_location" in st.secrets["api"] else "us-central1"
        default_model = st.secrets.get("default_model", "gemini-2.0-flash-exp")
        
        # サービスアカウント情報
        gcp_service_account = dict(st.secrets["gcp_service_account"]) if "gcp_service_account" in st.secrets else None
    except Exception as e:
        st.error(f"Secretsの読み込みエラー: {e}")
        vertex_ai_project_id = ""
        vertex_ai_location = "us-central1"
        default_model = "gemini-2.0-flash-exp"
        gcp_service_account = None
else:
    # ローカル環境の場合
    secrets_path = Path(__file__).parent / "secrets.toml"
    if secrets_path.exists():
        secrets = toml.load(secrets_path)
        api_config = secrets.get("api", {})
        vertex_ai_project_id = api_config.get("vertex_project", "")
        vertex_ai_location = api_config.get("vertex_location", "us-central1")
        default_model = secrets.get("default_model", "gemini-2.0-flash-exp")
        gcp_service_account = secrets.get("gcp_service_account", None)
    else:
        # 環境変数から取得
        vertex_ai_project_id = os.environ.get("VERTEX_AI_PROJECT_ID", "")
        vertex_ai_location = os.environ.get("VERTEX_AI_LOCATION", "us-central1")
        default_model = os.environ.get("DEFAULT_MODEL", "gemini-2.0-flash-exp")
        gcp_service_account = None

# Vertex AI モデルオプション
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

# Vertex AI設定関数
def setup_vertex_ai(model_name, project_id=None, location=None, service_account=None):
    """Vertex AI を設定する"""
    try:
        if not VERTEX_AI_AVAILABLE:
            st.error("Vertex AI ライブラリがインストールされていません")
            return None, None
            
        if not project_id:
            st.error("Project IDが設定されていません")
            return None, None
        
        # サービスアカウント認証を使用
        if service_account:
            try:
                from google.oauth2 import service_account as sa
                credentials = sa.Credentials.from_service_account_info(
                    service_account,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
            except Exception as e:
                st.error(f"サービスアカウント認証エラー: {e}")
                credentials = None
        else:
            credentials = None
            
        if NEW_SDK:
            if credentials:
                # 認証情報を使用
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ''  # Clear any existing
                client = genai.Client(
                    vertexai=True,
                    project=project_id,
                    location=location,
                    credentials=credentials
                )
            else:
                client = genai.Client(
                    vertexai=True,
                    project=project_id,
                    location=location
                )
        else:
            genai.configure(project=project_id, location=location)
            client = None
        return client, model_name
    except Exception as e:
        st.error(f"Vertex AI の設定に失敗しました: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None

# Function to call Gemini API  
def call_gemini(prompt, user_message, model_name, project_id=None, location=None, service_account=None):
    """Gemini APIを呼び出す"""
    try:
        client, model = setup_vertex_ai(model_name, project_id, location, service_account)
        
        if client is None and model is None:
            return None
            
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
        import traceback
        st.error(traceback.format_exc())
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
project_id_input = vertex_ai_project_id
location_input = vertex_ai_location

# Project IDが設定されていない場合の警告
if not vertex_ai_project_id:
    st.warning("⚠️ Vertex AI Project IDが設定されていません。設定セクションで入力してください。")

# Settings section
with st.expander("⚙️ 設定", expanded=not vertex_ai_project_id):
    col1, col2 = st.columns(2)
    
    with col1:
        # モデル選択
        selected_model = st.selectbox(
            "🎯 モデル",
            vertex_model_options,
            index=0 if default_model not in vertex_model_options else vertex_model_options.index(default_model),
            key="selected_model"
        )
    
    with col2:
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

# 入力モード選択
input_mode = st.radio(
    "入力方法を選択",
    ["手動入力", "CSV一括処理"],
    key="input_mode"
)

if input_mode == "手動入力":
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
    
else:  # CSV一括処理モード
    st.info("生成アプリで出力されたCSVファイルをアップロードしてください")
    
    uploaded_file = st.file_uploader("CSVファイルを選択", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # CSVファイルを読み込み
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            
            # 必要な列の存在確認
            required_columns = ["id", "質問", "回答"]
            if not all(col in df.columns for col in required_columns):
                st.error(f"必須列が不足しています: {required_columns}")
            else:
                # キーワード列の検出（動的に対応）
                keyword_columns = []
                for col in df.columns:
                    # カテゴリ名で終わる列を検出（例: ハウス1, サイン2, など）
                    if any(col.endswith(str(i)) for i in range(1, 5)):
                        for cat in ["ハウス", "サイン", "天体", "エレメント", "MP軸", "タロット"]:
                            if col.startswith(cat):
                                keyword_columns.append(col)
                                break
                
                st.success(f"✅ {len(df)}件のデータを読み込みました")
                st.write(f"検出されたキーワード列: {keyword_columns}")
                
                # プレビュー表示
                with st.expander("データプレビュー", expanded=False):
                    st.dataframe(df.head())
                
                # セッション状態に保存
                if 'csv_data' not in st.session_state:
                    st.session_state.csv_data = df
                    st.session_state.keyword_columns = keyword_columns
                
        except Exception as e:
            st.error(f"CSVファイルの読み込みエラー: {e}")

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
if input_mode == "手動入力":
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
            current_project_id = project_id_input
            current_location = location_input
            current_service_account = gcp_service_account

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
                    selected_model,
                    current_project_id,
                    current_location,
                    current_service_account
                )
                if tonmana_result:
                    tonmana_json = parse_json_response(tonmana_result)
                    if tonmana_json:
                        st.session_state.tonmana_result = tonmana_result
                        st.session_state.tonmana_json = tonmana_json
                        
                        # Store in session state
                        # プロンプトではimprovementsという名前だが、内部的にproblemsとして扱う
                        problems = tonmana_json.get('improvements', tonmana_json.get('problems', []))
                        st.session_state.corrections["tonmana"] = {
                            "score": tonmana_json.get('style_score', 'N/A'),
                            "comment": tonmana_json.get('comment', 'N/A'),
                            "problems": problems
                        }
        
        if 'japanese_result' not in st.session_state:
            with st.spinner("日本語校正中..."):
                # 2. 日本語校正
                japanese_message = ai_answer
                japanese_result = call_gemini(
                    japanese_prompt,
                    japanese_message,
                    selected_model,
                    current_project_id,
                    current_location,
                    current_service_account
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
                    selected_model,
                    current_project_id,
                    current_location,
                    current_service_account
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
            
            problems = tonmana_json.get('improvements', tonmana_json.get('problems', []))
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
                current_project_id = project_id_input
                current_location = location_input
                current_service_account = gcp_service_account
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
                    selected_model,
                    current_project_id,
                    current_location,
                    current_service_account
                )
                if comprehensive_result:
                    # Save the result to session state
                    st.session_state.comprehensive_result = comprehensive_result
        
        # Display comprehensive correction result
        if 'comprehensive_result' in st.session_state:
            st.subheader("総合校正結果")
            st.write(st.session_state.comprehensive_result)

# CSV一括処理モード
elif input_mode == "CSV一括処理" and 'csv_data' in st.session_state:
    if st.button("一括校正を実行"):
        df = st.session_state.csv_data
        keyword_columns = st.session_state.keyword_columns
        
        # 結果を保存するための列を追加
        df['トンマナスコア'] = 0
        df['日本語スコア'] = 0
        df['ロジックスコア'] = 0
        df['総合スコア'] = 0
        df['改善点'] = ""
        df['総合校正結果'] = ""
        
        # 設定の準備
        current_project_id = project_id_input
        current_location = location_input
        current_service_account = gcp_service_account
        
        # プログレスバー
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 各行を処理
        for index, row in df.iterrows():
            status_text.text(f"処理中: {index + 1}/{len(df)}")
            
            # 質問と回答を取得
            current_question = row['質問']
            current_answer = row['回答']
            
            # キーワードを整理
            keywords = []
            for col in keyword_columns:
                if pd.notna(row[col]):
                    # カテゴリ名を抽出（例: "ハウス1" -> "ハウス"）
                    category = ''.join([c for c in col if not c.isdigit()])
                    keywords.append(f"{category}: {row[col]}")
            
            # 1. トンマナ校正
            tonmana_message = f"""##QUESTION##
{current_question}

##KEYWORDS##
{', '.join(keywords) if keywords else 'なし'}

##ANSWER_CAND##
{current_answer}
"""
            tonmana_result = call_gemini(
                tonmana_prompt, 
                tonmana_message,
                selected_model,
                current_project_id,
                current_location,
                current_service_account
            )
            
            tonmana_json = parse_json_response(tonmana_result) if tonmana_result else None
            if tonmana_json:
                df.at[index, 'トンマナスコア'] = tonmana_json.get('style_score', 0)
                problems = tonmana_json.get('improvements', tonmana_json.get('problems', []))
                if problems:
                    df.at[index, '改善点'] += f"【トンマナ】{', '.join(problems)}\n"
            
            # 2. 日本語校正
            japanese_result = call_gemini(
                japanese_prompt,
                current_answer,
                selected_model,
                current_project_id,
                current_location,
                current_service_account
            )
            
            japanese_json = parse_json_response(japanese_result) if japanese_result else None
            if japanese_json:
                df.at[index, '日本語スコア'] = japanese_json.get('score', 0)
                improvements = japanese_json.get('improvements', [])
                if improvements:
                    df.at[index, '改善点'] += f"【日本語】{', '.join(improvements)}\n"
            
            # 3. ロジック校正
            logic_message = f"""質問: {current_question}
使用キーワード: {', '.join(keywords) if keywords else 'なし'}
回答: {current_answer}
"""
            logic_result = call_gemini(
                logic_prompt,
                logic_message,
                selected_model,
                current_project_id,
                current_location,
                current_service_account
            )
            
            logic_json = parse_json_response(logic_result) if logic_result else None
            if logic_json:
                df.at[index, 'ロジックスコア'] = logic_json.get('score', 0)
                improvements = logic_json.get('improvements', [])
                if improvements:
                    df.at[index, '改善点'] += f"【ロジック】{', '.join(improvements)}\n"
            
            # 総合スコア計算
            total_score = df.at[index, 'トンマナスコア'] + df.at[index, '日本語スコア'] + df.at[index, 'ロジックスコア']
            df.at[index, '総合スコア'] = total_score
            
            # プログレス更新
            progress_bar.progress((index + 1) / len(df))
        
        status_text.text("処理完了!")
        
        # 結果表示
        st.subheader("校正結果サマリー")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_tonmana = df['トンマナスコア'].mean()
            st.metric("平均トンマナスコア", f"{avg_tonmana:.2f}/5")
        
        with col2:
            avg_japanese = df['日本語スコア'].mean()
            st.metric("平均日本語スコア", f"{avg_japanese:.2f}/5")
        
        with col3:
            avg_logic = df['ロジックスコア'].mean()
            st.metric("平均ロジックスコア", f"{avg_logic:.2f}/5")
        
        with col4:
            avg_total = df['総合スコア'].mean()
            st.metric("平均総合スコア", f"{avg_total:.2f}/15")
        
        # 結果プレビュー
        with st.expander("結果プレビュー", expanded=True):
            st.dataframe(df[['id', '質問', 'トンマナスコア', '日本語スコア', 'ロジックスコア', '総合スコア', '改善点']].head(10))
        
        # CSV出力
        output_buffer = io.StringIO()
        df.to_csv(output_buffer, index=False, encoding='utf-8-sig')
        
        st.download_button(
            label="📥 校正結果をCSVでダウンロード",
            data=output_buffer.getvalue(),
            file_name=f"mimiko_correction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )