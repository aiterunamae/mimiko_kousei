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
    try:
        from google.genai.types import GenerateContentConfig, ThinkingConfig
        HAS_THINKING_CONFIG = True
    except ImportError:
        # 古いバージョンの場合
        HAS_THINKING_CONFIG = False
    NEW_SDK = True
except ImportError:
    try:
        import google.generativeai as genai
        NEW_SDK = False
        HAS_THINKING_CONFIG = False
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
        default_model = st.secrets.get("default_model", "gemini-2.0-flash")
        
        # サービスアカウント情報
        gcp_service_account = dict(st.secrets["gcp_service_account"]) if "gcp_service_account" in st.secrets else None
    except Exception as e:
        st.error(f"Secretsの読み込みエラー: {e}")
        vertex_ai_project_id = ""
        vertex_ai_location = "us-central1"
        default_model = "gemini-2.0-flash"
        gcp_service_account = None
else:
    # ローカル環境の場合
    secrets_path = Path(__file__).parent / "secrets.toml"
    if secrets_path.exists():
        secrets = toml.load(secrets_path)
        api_config = secrets.get("api", {})
        vertex_ai_project_id = api_config.get("vertex_project", "")
        vertex_ai_location = api_config.get("vertex_location", "us-central1")
        default_model = secrets.get("default_model", "gemini-2.0-flash")
        gcp_service_account = secrets.get("gcp_service_account", None)
    else:
        # 環境変数から取得
        vertex_ai_project_id = os.environ.get("VERTEX_AI_PROJECT_ID", "")
        vertex_ai_location = os.environ.get("VERTEX_AI_LOCATION", "us-central1")
        default_model = os.environ.get("DEFAULT_MODEL", "gemini-2.0-flash")
        gcp_service_account = None

# Vertex AI モデルオプション
vertex_model_options = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-pro"
]

# Load correction prompts
def load_prompt(filename):
    prompt_path = Path(__file__).parent / filename
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

# Load keyword CSV files
def load_keyword_csv(category):
    """カテゴリに応じたキーワードCSVを読み込む"""
    keyword_files = {
        "ハウス": "ハウスキーワード.csv",
        "サイン": "サインキーワード.csv",
        "天体": "天体キーワード.csv",
        "エレメント": "エレメントキーワード.csv",
        "MP軸": "MP軸キーワード.csv",
        "タロット": "タロットキーワード.csv"
    }
    
    if category not in keyword_files:
        return None
    
    try:
        csv_path = Path(__file__).parent / keyword_files[category]
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        return df
    except Exception as e:
        st.warning(f"{category}キーワードの読み込みに失敗しました: {e}")
        return None

# Get keyword details from CSV
def normalize_numbers(text):
    """全角数字を半角数字に変換"""
    if not isinstance(text, str):
        return text
    # 全角数字を半角数字に変換
    trans_table = str.maketrans('０１２３４５６７８９', '0123456789')
    return text.translate(trans_table)

def get_keyword_details(keywords_list):
    """キーワードリストから詳細情報を取得"""
    keyword_details = []
    
    for keyword_str in keywords_list:
        # "カテゴリ: キーワード" の形式から分解
        if ": " in keyword_str:
            category, keyword = keyword_str.split(": ", 1)
            
            # カテゴリに応じたCSVを読み込み
            df = load_keyword_csv(category)
            if df is not None:
                # キーワードに一致する行を検索
                # 最初の列（名称列）で検索
                name_col = df.columns[0]
                
                # 数字を正規化して検索
                normalized_keyword = normalize_numbers(keyword)
                
                # CSV側の数字も正規化して比較
                df_normalized = df.copy()
                df_normalized[name_col] = df_normalized[name_col].apply(normalize_numbers)
                
                # 完全一致で検索
                matching_rows = df_normalized[df_normalized[name_col] == normalized_keyword]
                
                # 完全一致が見つからない場合、「第」を含む/含まないパターンも試す
                if matching_rows.empty and category == "ハウス":
                    if normalized_keyword.startswith("第"):
                        # 「第」を除いて検索
                        alt_keyword = normalized_keyword[1:]
                        matching_rows = df_normalized[df_normalized[name_col] == alt_keyword]
                    else:
                        # 「第」を追加して検索
                        alt_keyword = "第" + normalized_keyword
                        matching_rows = df_normalized[df_normalized[name_col] == alt_keyword]
                
                if not matching_rows.empty:
                    # 最初の一致行のインデックスを取得
                    matched_idx = matching_rows.index[0]
                    # 元のデータフレームから行を取得
                    row = df.iloc[matched_idx]
                    
                    # キーワードの詳細情報を辞書形式で保存
                    detail = {
                        "カテゴリ": category,
                        "キーワード": keyword,
                        "CSVファイル": f"{category}キーワード.csv",
                        "検索列": name_col,
                        "一致": "○",
                        "正規化後": normalized_keyword
                    }
                    
                    # 他の列の情報も追加
                    for col in df.columns[1:]:
                        if pd.notna(row[col]):
                            detail[col] = row[col]
                    
                    keyword_details.append(detail)
                else:
                    # 一致しない場合は基本情報のみ
                    keyword_details.append({
                        "カテゴリ": category,
                        "キーワード": keyword,
                        "CSVファイル": f"{category}キーワード.csv",
                        "検索列": name_col,
                        "一致": "×",
                        "正規化後": normalized_keyword,
                        "注意": "詳細情報が見つかりませんでした"
                    })
            else:
                # CSVファイルが読み込めない場合
                keyword_details.append({
                    "カテゴリ": category,
                    "キーワード": keyword,
                    "注意": "CSVファイルが読み込めませんでした"
                })
    
    return keyword_details

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
def call_gemini(prompt, user_message, model_name, project_id=None, location=None, service_account=None, max_tokens=2000, thinking_budget=1024):
    """Gemini APIを呼び出す"""
    try:
        client, model = setup_vertex_ai(model_name, project_id, location, service_account)
        
        if client is None and model is None:
            return None
            
        # メッセージを構築
        full_message = f"{prompt}\n\n{user_message}"
        
        if NEW_SDK:
            if client:
                # GenerateContentConfigが利用可能かチェック
                if HAS_THINKING_CONFIG and "2.5" in model:
                    # Gemini 2.5モデルの場合、ThinkingConfigを使用可能
                    config = GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.1,
                        thinking_config=ThinkingConfig(
                            thinking_budget=thinking_budget,
                            include_thoughts=False  # 推論過程は含めない
                        )
                    )
                elif HAS_THINKING_CONFIG:
                    config = GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.1,
                    )
                else:
                    # 古いtypes.GenerateContentConfigを使用
                    config = types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.1,
                    )
                
                response = client.models.generate_content(
                    model=model,
                    contents=full_message,
                    config=config
                )
                
                # 応答のテキストを取得
                result_text = None
                
                try:
                    # まず標準的な.textプロパティを試す
                    if hasattr(response, 'text'):
                        if callable(response.text):
                            # textがメソッドの場合
                            result_text = response.text()
                        else:
                            # textがプロパティの場合
                            result_text = response.text
                except Exception as e:
                    st.warning(f"response.textの取得に失敗: {e}")
                    
                    # 他の方法を試す
                    if hasattr(response, 'candidates') and response.candidates:
                        try:
                            # 最初の候補からテキストを取得
                            candidate = response.candidates[0]
                            if hasattr(candidate, 'content'):
                                content = candidate.content
                                if hasattr(content, 'parts') and content.parts:
                                    # partsから最初のテキストパートを取得
                                    for part in content.parts:
                                        if hasattr(part, 'text'):
                                            result_text = part.text
                                            break
                                elif hasattr(content, 'text'):
                                    result_text = content.text
                        except Exception as e2:
                            st.error(f"候補からのテキスト取得に失敗: {e2}")
                    
                    # それでも取得できない場合、応答オブジェクトの詳細を表示
                    if result_text is None:
                        st.error(f"応答からテキストを取得できませんでした")
                        st.error(f"応答タイプ: {type(response)}")
                        
                        # 応答の属性を詳しく調査
                        if hasattr(response, '__dict__'):
                            st.error(f"応答の内容: {response.__dict__}")
                        
                        # candidatesの詳細を表示
                        if hasattr(response, 'candidates') and response.candidates:
                            st.error(f"候補の数: {len(response.candidates)}")
                            if response.candidates:
                                st.error(f"最初の候補の型: {type(response.candidates[0])}")
                                if hasattr(response.candidates[0], '__dict__'):
                                    st.error(f"最初の候補の内容: {response.candidates[0].__dict__}")
                
                return result_text
            else:
                st.error("クライアントの初期化に失敗しました")
                return None
        else:
            model_obj = genai.GenerativeModel(model)
            response = model_obj.generate_content(
                full_message,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.1,
                )
            )
            return response.text
            
    except Exception as e:
        st.error(f"Gemini API呼び出しエラー: {e}")
        st.error(f"モデル名: {model_name}")
        st.error(f"プロジェクトID: {project_id}")
        st.error(f"ロケーション: {location}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Function to parse JSON from Claude's response
def parse_json_response(response):
    try:
        # 応答がNoneまたは空の場合
        if not response:
            st.error("空の応答を受信しました")
            return None
            
        # デバッグ用：最初の100文字を表示（本番環境ではコメントアウト）
        # st.info(f"応答の最初の100文字: {response[:100]}...")
        
        # マークダウンのコードブロックを除去
        if response.strip().startswith("```json"):
            response = response.strip()[7:]  # ```json を除去
            if response.endswith("```"):
                response = response[:-3]  # 末尾の ``` を除去
        elif response.strip().startswith("```"):
            response = response.strip()[3:]  # ``` を除去
            if response.endswith("```"):
                response = response[:-3]  # 末尾の ``` を除去
        
        # JSONの開始と終了位置を探す
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        # JSONが切れている可能性をチェック
        if start_idx >= 0:
            # 開き括弧の数と閉じ括弧の数を数える
            open_brackets = response.count('[')
            close_brackets = response.count(']')
            open_braces = response.count('{')
            close_braces = response.count('}')
            
            # 切れている場合の処理
            if open_brackets > close_brackets or open_braces > close_braces:
                st.warning("JSONが途中で切れている可能性があります")
                # 応答が切れている場合、デフォルト値を返す
                return {
                    "score": 3,  # 中間スコア
                    "improvements": ["応答が途中で切れたため、完全な評価ができませんでした"]
                }
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as je:
                st.warning(f"JSON解析エラー: {je}")
                # 部分的なJSONを解析してみる
                try:
                    # 不完全なJSONを修復する試み
                    if '"improvements": [' in json_str and not json_str.rstrip().endswith(']'):
                        json_str = json_str.rstrip() + ']}' if not json_str.rstrip().endswith('}') else json_str
                    return json.loads(json_str)
                except:
                    return {
                        "score": 3,
                        "improvements": ["JSON解析に失敗しました"]
                    }
        else:
            st.warning("Could not find JSON in the response")
            with st.expander("生の応答を表示", expanded=True):
                st.code(response)
            return None
    except Exception as e:
        st.error(f"Error parsing JSON: {e}")
        with st.expander("生の応答を表示", expanded=True):
            st.code(response)
        return None

# Main app
# カスタムCSSを適用
st.markdown("""
<style>
    /* メインタイトルのスタイリング */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 3rem;
    }
    
    /* ボタンのスタイリング */
    .stButton > button {
        transition: all 0.3s ease;
        border-radius: 10px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    /* メトリックのスタイリング */
    [data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* プログレスバーのスタイリング */
    .stProgress > div > div > div {
        height: 20px;
        border-radius: 10px;
    }
    
    /* エキスパンダーのスタイリング */
    .streamlit-expanderHeader {
        font-weight: 600;
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* サクセスメッセージのスタイリング */
    .element-container:has(.stSuccess), 
    .element-container:has(.stInfo), 
    .element-container:has(.stWarning), 
    .element-container:has(.stError) {
        animation: fadeIn 0.5s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* セクションヘッダーのスタイリング */
    h2 {
        color: #4a5568;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    
    h3 {
        color: #4a5568;
        margin-top: 1.5rem;
    }
    
    /* チェックボックスのスタイリング */
    .stCheckbox {
        padding: 0.5rem;
        border-radius: 8px;
        transition: background-color 0.3s ease;
    }
    
    .stCheckbox:hover {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# メインヘッダー
st.markdown("""
<div class="main-header">
    <h1>🌙 mimiko校正システム</h1>
</div>
""", unsafe_allow_html=True)

# Project IDが設定されていない場合の警告
if not vertex_ai_project_id:
    st.error("⚠️ Vertex AI Project IDが設定されていません。secrets.tomlファイルに設定してください。")

# Settings section
with st.expander("⚙️ 詳細設定", expanded=False):
    # モデル選択
    selected_model = st.selectbox(
        "🎯 モデル",
        vertex_model_options,
        index=0 if default_model not in vertex_model_options else vertex_model_options.index(default_model),
        key="selected_model"
    )
    
    # Thinking Budget設定（2.5モデルの場合のみ）
    thinking_budget = 1024  # デフォルト値
    if "2.5" in selected_model:
        st.write("### 🧠 推論設定")
        if "2.5-flash" in selected_model:
            thinking_budget = st.slider(
                "Thinking Budget",
                min_value=0,
                max_value=8192,
                value=1024,
                step=128,
                help="推論に使用するトークン数。0に設定すると推論機能を無効化します。"
            )
        elif "2.5-pro" in selected_model:
            thinking_budget = st.slider(
                "Thinking Budget",
                min_value=128,
                max_value=32768,
                value=1024,
                step=128,
                help="推論に使用するトークン数。Proモデルは最小128トークンが必要です。"
            )
    
    # 校正ON/OFF設定
    st.write("### 📋 校正設定")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        enable_tonmana = st.checkbox("🎨 トンマナ校正", value=True, key="enable_tonmana")
    
    with col2:
        enable_japanese = st.checkbox("📝 日本語校正", value=False, key="enable_japanese")
    
    with col3:
        enable_logic = st.checkbox("🔍 ロジック校正", value=True, key="enable_logic")

# Input section
st.header("入力")

st.info("生成アプリで出力されたCSVファイルをアップロードしてください")

uploaded_file = st.file_uploader("CSVファイルを選択", type=['csv'])

if uploaded_file is not None:
    # ファイル名をキーとして使用
    file_key = f"file_{uploaded_file.name}_{uploaded_file.size}"
    
    # ファイルが変わったかチェック
    if 'current_file_key' not in st.session_state or st.session_state.current_file_key != file_key:
        st.session_state.current_file_key = file_key
        # 新しいファイルの場合、関連するセッション状態をクリア
        for key in list(st.session_state.keys()):
            if key.startswith('correction_') or key.startswith('selected_') or key == 'csv_data':
                del st.session_state[key]
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
            st.session_state.csv_data = df
            st.session_state.keyword_columns = keyword_columns
            
    except Exception as e:
        st.error(f"CSVファイルの読み込みエラー: {e}")

# CSV処理
if 'csv_data' in st.session_state:
    df = st.session_state.csv_data
    keyword_columns = st.session_state.keyword_columns
    
    # 行選択
    st.subheader("📝 データ選択")
    
    # データ一覧を表示
    with st.expander("データ一覧", expanded=False):
        for idx, row in df.iterrows():
            st.write(f"**[{idx+1}]** ID: {row['id']} - {row['質問'][:80]}...")
    
    # 数値入力で選択
    col1, col2 = st.columns([1, 3])
    with col1:
        # number_inputを使用
        row_number = st.number_input(
            "データ番号",
            min_value=1,
            max_value=len(df),
            value=1,
            step=1,
            help=f"1から{len(df)}の番号を入力"
        )
        selected_row_idx = row_number - 1  # 0ベースのインデックスに変換
    
    with col2:
        # 選択されたデータの簡易表示
        if 0 <= selected_row_idx < len(df):
            row = df.iloc[selected_row_idx]
            st.write(f"**選択中:** ID: {row['id']} - {row['質問'][:50]}...")
    
    # 選択された行のデータ
    selected_row = df.iloc[selected_row_idx]
    
    # プレビュー表示
    with st.expander("選択されたデータの詳細", expanded=True):
        st.write(f"**ID:** {selected_row['id']}")
        st.write(f"**質問:** {selected_row['質問']}")
        
        # キーワード表示
        keywords_display = []
        for col in keyword_columns:
            if pd.notna(selected_row[col]):
                category = ''.join([c for c in col if not c.isdigit()])
                keywords_display.append(f"{category}: {selected_row[col]}")
        if keywords_display:
            st.write(f"**キーワード:** {', '.join(keywords_display)}")
        
        st.write(f"**回答:**")
        st.text_area("", value=selected_row['回答'], height=150, disabled=True)
    
    # 校正ボタンとリセットボタン
    col_btn1, col_btn2 = st.columns([2, 1])
    
    with col_btn1:
        do_correction = st.button("🔍 この回答を校正する", use_container_width=True)
    
    with col_btn2:
        if f'correction_done_{selected_row_idx}' in st.session_state:
            if st.button("🔄 結果をリセット", use_container_width=True):
                # 選択されたデータに関連するセッション状態をクリア
                keys_to_remove = []
                for key in st.session_state.keys():
                    if f'_{selected_row_idx}' in key:
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del st.session_state[key]
                
                st.success("校正結果をリセットしました")
                st.rerun()
    
    # 個別校正実行
    if do_correction or f'correction_done_{selected_row_idx}' in st.session_state:
        # セッション状態の初期化
        if f'corrections_{selected_row_idx}' not in st.session_state:
            st.session_state[f'corrections_{selected_row_idx}'] = {}
        
        # 設定の準備
        current_project_id = vertex_ai_project_id
        current_location = vertex_ai_location
        current_service_account = gcp_service_account
        
        # 質問と回答を取得
        current_question = selected_row['質問']
        current_answer = selected_row['回答']
        
        # キーワードを整理
        keywords = []
        for col in keyword_columns:
            if pd.notna(selected_row[col]):
                category = ''.join([c for c in col if not c.isdigit()])
                keywords.append(f"{category}: {selected_row[col]}")
        
        # 1. トンマナ校正
        if f'tonmana_result_{selected_row_idx}' not in st.session_state:
            if st.session_state.get('enable_tonmana', True):
                with st.spinner("トンマナ校正中..."):
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
                        current_service_account,
                        thinking_budget=thinking_budget
                    )
                    
                    if tonmana_result:
                        tonmana_json = parse_json_response(tonmana_result)
                        if tonmana_json:
                            st.session_state[f'tonmana_result_{selected_row_idx}'] = tonmana_result
                            st.session_state[f'tonmana_json_{selected_row_idx}'] = tonmana_json
                            
                            st.session_state[f'corrections_{selected_row_idx}']['tonmana'] = {
                                'score': tonmana_json.get('score', 0),
                                'improvements': tonmana_json.get('improvements', [])
                            }
                        else:
                            # JSONパースに失敗した場合、生のレスポンスを表示
                            st.error("トンマナ校正のJSON解析に失敗しました")
                            with st.expander("デバッグ情報", expanded=True):
                                st.code(tonmana_result)
            else:
                # トンマナ校正がOFFの場合
                st.session_state[f'tonmana_json_{selected_row_idx}'] = {'score': 5, 'improvements': []}
                st.session_state[f'corrections_{selected_row_idx}']['tonmana'] = {
                    'score': 5,
                    'improvements': []
                }
        
        # 2. 日本語校正
        if f'japanese_result_{selected_row_idx}' not in st.session_state:
            if st.session_state.get('enable_japanese', False):
                with st.spinner("日本語校正中..."):
                    japanese_result = call_gemini(
                        japanese_prompt,
                        current_answer,
                        selected_model,
                        current_project_id,
                        current_location,
                        current_service_account,
                        thinking_budget=thinking_budget
                    )
                    
                    if japanese_result:
                        japanese_json = parse_json_response(japanese_result)
                        if japanese_json:
                            st.session_state[f'japanese_result_{selected_row_idx}'] = japanese_result
                            st.session_state[f'japanese_json_{selected_row_idx}'] = japanese_json
                            
                            st.session_state[f'corrections_{selected_row_idx}']['japanese'] = {
                                'score': japanese_json.get('score', 0),
                                'improvements': japanese_json.get('improvements', [])
                            }
            else:
                # 日本語校正がOFFの場合
                st.session_state[f'japanese_json_{selected_row_idx}'] = {'score': 5, 'improvements': []}
                st.session_state[f'corrections_{selected_row_idx}']['japanese'] = {
                    'score': 5,
                    'improvements': []
                }
        
        # 3. ロジック校正
        if f'logic_result_{selected_row_idx}' not in st.session_state:
            if st.session_state.get('enable_logic', True):
                with st.spinner("ロジック校正中..."):
                    # キーワードの詳細情報を取得
                    keyword_details = get_keyword_details(keywords)
                    
                    # 元キーワードとアレンジキーワードを取得
                    original_keywords = []
                    arrange_keywords = []
                    
                    # CSVの列から元キーワードとアレンジキーワードを探す
                    for col in df.columns:
                        if '元キーワード' in col and pd.notna(selected_row[col]):
                            original_keywords.append(selected_row[col])
                        elif 'アレンジキーワード' in col and pd.notna(selected_row[col]):
                            arrange_keywords.append(selected_row[col])
                    
                    # キーワード詳細情報をJSON形式で整形
                    keyword_info = json.dumps(keyword_details, ensure_ascii=False, indent=2)
                    
                    # デバッグ用にキーワード情報を保存
                    st.session_state[f'keyword_details_{selected_row_idx}'] = keyword_details
                    st.session_state[f'keyword_info_{selected_row_idx}'] = keyword_info
                    st.session_state[f'original_keywords_{selected_row_idx}'] = original_keywords
                    st.session_state[f'arrange_keywords_{selected_row_idx}'] = arrange_keywords
                    
                    logic_message = f"""質問: {current_question}

使用キーワード:
{keyword_info}

元キーワード: {', '.join(original_keywords) if original_keywords else 'なし'}

アレンジキーワード: {', '.join(arrange_keywords) if arrange_keywords else 'なし'}

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
                    
                    if logic_result:
                        logic_json = parse_json_response(logic_result)
                        if logic_json:
                            st.session_state[f'logic_result_{selected_row_idx}'] = logic_result
                            st.session_state[f'logic_json_{selected_row_idx}'] = logic_json
                            
                            st.session_state[f'corrections_{selected_row_idx}']['logic'] = {
                                'score': logic_json.get('score', 0),
                                'improvements': logic_json.get('improvements', [])
                            }
            else:
                # ロジック校正がOFFの場合
                st.session_state[f'logic_json_{selected_row_idx}'] = {'score': 5, 'improvements': []}
                st.session_state[f'corrections_{selected_row_idx}']['logic'] = {
                    'score': 5,
                    'improvements': []
                }
        
        # 校正完了フラグ
        st.session_state[f'correction_done_{selected_row_idx}'] = True
        
        # 結果表示
        st.header("📊 校正結果")
        
        # 総合スコア計算
        total_score = 0
        if f'corrections_{selected_row_idx}' in st.session_state:
            corrections = st.session_state[f'corrections_{selected_row_idx}']
            for correction_type in ['tonmana', 'japanese', 'logic']:
                if correction_type in corrections:
                    total_score += corrections[correction_type].get('score', 0)
        
        # スコアに応じて色分けしたメッセージ
        score_percentage = (total_score / 15) * 100
        
        # スコアメッセージをカード形式で表示
        if score_percentage >= 80:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #4caf50 0%, #8bc34a 100%); 
                        padding: 20px; border-radius: 15px; text-align: center; color: white;
                        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);'>
                <h2 style='margin: 0; color: white;'>🎉 総合スコア: {total_score}/15点</h2>
            </div>
            """, unsafe_allow_html=True)
        elif score_percentage >= 60:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #2196f3 0%, #64b5f6 100%); 
                        padding: 20px; border-radius: 15px; text-align: center; color: white;
                        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);'>
                <h2 style='margin: 0; color: white;'>📊 総合スコア: {total_score}/15点</h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #ff9800 0%, #ffb74d 100%); 
                        padding: 20px; border-radius: 15px; text-align: center; color: white;
                        box-shadow: 0 4px 15px rgba(255, 152, 0, 0.3);'>
                <h2 style='margin: 0; color: white;'>⚠️ 総合スコア: {total_score}/15点</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 改善点選択用のセッション状態初期化
        if f'selected_tonmana_{selected_row_idx}' not in st.session_state:
            st.session_state[f'selected_tonmana_{selected_row_idx}'] = []
        if f'selected_japanese_{selected_row_idx}' not in st.session_state:
            st.session_state[f'selected_japanese_{selected_row_idx}'] = []
        if f'selected_logic_{selected_row_idx}' not in st.session_state:
            st.session_state[f'selected_logic_{selected_row_idx}'] = []
        
        # 1. トンマナ校正結果
        if f'tonmana_json_{selected_row_idx}' in st.session_state:
            with st.container():
                st.subheader("🎨 トンマナ校正結果")
                tonmana_json = st.session_state[f'tonmana_json_{selected_row_idx}']
                
                # スコアをプログレスバーで表示
                col1, col2, col3 = st.columns([2, 1, 3])
                with col1:
                    score = tonmana_json.get('score', 0)
                    st.progress(score / 5, text=f"スコア: {score}/5")
                
                with col2:
                    # スコアに応じた絵文字
                    if score >= 4:
                        st.markdown("### ✅")
                    elif score >= 3:
                        st.markdown("### ⚠️")
                    else:
                        st.markdown("### ❌")
                
                # 校正がOFFの場合の表示
                if not st.session_state.get('enable_tonmana', True):
                    with col3:
                        st.info("トンマナ校正はスキップされました")
            
            improvements = tonmana_json.get('improvements', [])
            if improvements:
                with st.expander(f"💡 改善点 ({len(improvements)}件)", expanded=True):
                    st.caption("改善を適用したい項目を選択してください：")
                    for i, improvement in enumerate(improvements):
                        # 改善点のテキストを安全に処理
                        safe_improvement = str(improvement).replace('\n', ' ').replace('\r', ' ').strip()
                        # 長すぎる場合は切り詰める
                        display_text = safe_improvement[:200] + "..." if len(safe_improvement) > 200 else safe_improvement
                        
                        is_selected = improvement in st.session_state[f'selected_tonmana_{selected_row_idx}']
                        
                        # 選択状態に応じて背景色を変更
                        if is_selected:
                            st.markdown(f"<div style='background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin: 5px 0;'>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div style='background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin: 5px 0;'>", unsafe_allow_html=True)
                        
                        if st.checkbox(f"改善点 {i+1}", key=f"tonmana_cb_{selected_row_idx}_{i}", value=is_selected, help=safe_improvement):
                            if improvement not in st.session_state[f'selected_tonmana_{selected_row_idx}']:
                                st.session_state[f'selected_tonmana_{selected_row_idx}'].append(improvement)
                        else:
                            if improvement in st.session_state[f'selected_tonmana_{selected_row_idx}']:
                                st.session_state[f'selected_tonmana_{selected_row_idx}'].remove(improvement)
                        
                        st.caption(display_text)
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.success("✅ 改善点はありません")
        
        # 2. 日本語校正結果
        if f'japanese_json_{selected_row_idx}' in st.session_state:
            with st.container():
                st.subheader("📝 日本語校正結果")
                japanese_json = st.session_state[f'japanese_json_{selected_row_idx}']
                
                # スコアをプログレスバーで表示
                col1, col2, col3 = st.columns([2, 1, 3])
                with col1:
                    score = japanese_json.get('score', 0)
                    st.progress(score / 5, text=f"スコア: {score}/5")
                
                with col2:
                    # スコアに応じた絵文字
                    if score >= 4:
                        st.markdown("### ✅")
                    elif score >= 3:
                        st.markdown("### ⚠️")
                    else:
                        st.markdown("### ❌")
                
                # 校正がOFFの場合の表示
                if not st.session_state.get('enable_japanese', False):
                    with col3:
                        st.info("日本語校正はスキップされました")
                
                improvements = japanese_json.get('improvements', [])
                if improvements:
                    with st.expander(f"💡 改善点 ({len(improvements)}件)", expanded=True):
                        st.caption("改善を適用したい項目を選択してください：")
                        for i, improvement in enumerate(improvements):
                            # 改善点のテキストを安全に処理
                            safe_improvement = str(improvement).replace('\n', ' ').replace('\r', ' ').strip()
                            # 長すぎる場合は切り詰める
                            display_text = safe_improvement[:200] + "..." if len(safe_improvement) > 200 else safe_improvement
                            
                            is_selected = improvement in st.session_state[f'selected_japanese_{selected_row_idx}']
                            
                            # 選択状態に応じて背景色を変更
                            if is_selected:
                                st.markdown(f"<div style='background-color: #fff3e0; padding: 10px; border-radius: 5px; margin: 5px 0;'>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<div style='background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin: 5px 0;'>", unsafe_allow_html=True)
                            
                            if st.checkbox(f"改善点 {i+1}", key=f"japanese_cb_{selected_row_idx}_{i}", value=is_selected, help=safe_improvement):
                                if improvement not in st.session_state[f'selected_japanese_{selected_row_idx}']:
                                    st.session_state[f'selected_japanese_{selected_row_idx}'].append(improvement)
                            else:
                                if improvement in st.session_state[f'selected_japanese_{selected_row_idx}']:
                                    st.session_state[f'selected_japanese_{selected_row_idx}'].remove(improvement)
                            
                            st.caption(display_text)
                            st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.success("✅ 改善点はありません - 日本語として完璧です！")
        
        # 3. ロジック校正結果
        if f'logic_json_{selected_row_idx}' in st.session_state:
            with st.container():
                st.subheader("🔍 ロジック校正結果")
                logic_json = st.session_state[f'logic_json_{selected_row_idx}']
                
                # スコアをプログレスバーで表示
                col1, col2, col3 = st.columns([2, 1, 3])
                with col1:
                    score = logic_json.get('score', 0)
                    st.progress(score / 5, text=f"スコア: {score}/5")
                
                with col2:
                    # スコアに応じた絵文字
                    if score >= 4:
                        st.markdown("### ✅")
                    elif score >= 3:
                        st.markdown("### ⚠️")
                    else:
                        st.markdown("### ❌")
                
                # 校正がOFFの場合の表示
                if not st.session_state.get('enable_logic', True):
                    with col3:
                        st.info("ロジック校正はスキップされました")
            
            # デバッグ情報：キーワード詳細を表示
            with st.expander("🔧 デバッグ: AIに送信されたキーワード情報", expanded=False):
                # カテゴリ別キーワードリスト
                st.write("**カテゴリ別キーワード（CSVから取得）:**")
                st.write(f"{', '.join(keywords) if keywords else 'なし'}")
                
                # 元キーワードとアレンジキーワード
                if f'original_keywords_{selected_row_idx}' in st.session_state:
                    st.write("\n**元キーワード:**")
                    orig_kws = st.session_state[f'original_keywords_{selected_row_idx}']
                    st.write(f"{', '.join(orig_kws) if orig_kws else 'なし'}")
                
                if f'arrange_keywords_{selected_row_idx}' in st.session_state:
                    st.write("\n**アレンジキーワード:**")
                    arr_kws = st.session_state[f'arrange_keywords_{selected_row_idx}']
                    st.write(f"{', '.join(arr_kws) if arr_kws else 'なし'}")
                
                if f'keyword_details_{selected_row_idx}' in st.session_state:
                    keyword_details = st.session_state[f'keyword_details_{selected_row_idx}']
                    st.write("\n**キーワード詳細情報（各キーワードの内容）:**")
                    
                    for i, detail in enumerate(keyword_details, 1):
                        st.write(f"\n**[{i}] {detail.get('カテゴリ', '')}: {detail.get('キーワード', '')}**")
                        
                        # 詳細情報を表形式で表示
                        detail_items = []
                        for key, value in detail.items():
                            if key not in ['カテゴリ', 'キーワード']:
                                detail_items.append(f"- **{key}**: {value}")
                        
                        if detail_items:
                            for item in detail_items:
                                st.write(item)
                        else:
                            st.write("- 詳細情報が見つかりませんでした")
                    
                    st.write("\n**JSON形式（AIに送信された内容）:**")
                    st.code(st.session_state[f'keyword_info_{selected_row_idx}'], language='json')
            
            improvements = logic_json.get('improvements', [])
            if improvements:
                with st.expander(f"💡 改善点 ({len(improvements)}件)", expanded=True):
                    st.caption("改善を適用したい項目を選択してください：")
                    for i, improvement in enumerate(improvements):
                        # 改善点のテキストを安全に処理
                        safe_improvement = str(improvement).replace('\n', ' ').replace('\r', ' ').strip()
                        # 長すぎる場合は切り詰める
                        display_text = safe_improvement[:200] + "..." if len(safe_improvement) > 200 else safe_improvement
                        
                        is_selected = improvement in st.session_state[f'selected_logic_{selected_row_idx}']
                        
                        # 選択状態に応じて背景色を変更
                        if is_selected:
                            st.markdown(f"<div style='background-color: #ffebee; padding: 10px; border-radius: 5px; margin: 5px 0;'>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div style='background-color: #f5f5f5; padding: 10px; border-radius: 5px; margin: 5px 0;'>", unsafe_allow_html=True)
                        
                        if st.checkbox(f"改善点 {i+1}", key=f"logic_cb_{selected_row_idx}_{i}", value=is_selected, help=safe_improvement):
                            if improvement not in st.session_state[f'selected_logic_{selected_row_idx}']:
                                st.session_state[f'selected_logic_{selected_row_idx}'].append(improvement)
                        else:
                            if improvement in st.session_state[f'selected_logic_{selected_row_idx}']:
                                st.session_state[f'selected_logic_{selected_row_idx}'].remove(improvement)
                        
                        st.caption(display_text)
                        st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.success("✅ 改善点はありません - ロジックは完璧です！")
        
        # 総合校正ボタン
        st.header("✨ 総合校正")
        if st.button("選択した改善点で総合校正を実行") or f'comprehensive_result_{selected_row_idx}' in st.session_state:
            if f'comprehensive_result_{selected_row_idx}' not in st.session_state:
                with st.spinner("総合校正中..."):
                    # 選択された改善点を整理
                    selected_improvements = {
                        "トンマナ校正": st.session_state.get(f'selected_tonmana_{selected_row_idx}', []),
                        "日本語校正": st.session_state.get(f'selected_japanese_{selected_row_idx}', []),
                        "ロジック校正": st.session_state.get(f'selected_logic_{selected_row_idx}', [])
                    }
                    
                    # スコアを整理
                    scores = {}
                    if f'corrections_{selected_row_idx}' in st.session_state:
                        corrections = st.session_state[f'corrections_{selected_row_idx}']
                        if 'tonmana' in corrections:
                            scores["トンマナ校正"] = corrections['tonmana'].get('score', 0)
                        if 'japanese' in corrections:
                            scores["日本語校正"] = corrections['japanese'].get('score', 0)
                        if 'logic' in corrections:
                            scores["ロジック校正"] = corrections['logic'].get('score', 0)
                    
                    # 総合校正メッセージ作成
                    comprehensive_message = f"""AI占い師の回答:
{current_answer}

使用されたキーワード:
{', '.join(keywords) if keywords else 'なし'}

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
                    
                    # 総合校正実行（長い出力のためトークン数を増やす）
                    comprehensive_result = call_gemini(
                        comprehensive_prompt,
                        comprehensive_message,
                        selected_model,
                        current_project_id,
                        current_location,
                        current_service_account,
                        max_tokens=4000,  # 総合校正は長い文章を出力するため大きく設定
                        thinking_budget=thinking_budget
                    )
                    
                    if comprehensive_result:
                        st.session_state[f'comprehensive_result_{selected_row_idx}'] = comprehensive_result
            
            # 総合校正結果表示
            if f'comprehensive_result_{selected_row_idx}' in st.session_state:
                st.subheader("📝 総合校正結果")
                
                # 元の回答と比較表示
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔸 元の回答**")
                    with st.container():
                        st.markdown(f"<div style='background-color: #fff3e0; padding: 15px; border-radius: 10px; max-height: 400px; overflow-y: auto;'>{current_answer}</div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("**✨ 校正後の回答**")
                    with st.container():
                        corrected_text = st.session_state[f'comprehensive_result_{selected_row_idx}']
                        st.markdown(f"<div style='background-color: #e8f5e9; padding: 15px; border-radius: 10px; max-height: 400px; overflow-y: auto;'>{corrected_text}</div>", unsafe_allow_html=True)
                
                # 適用された改善点のサマリー
                if any([
                    st.session_state.get(f'selected_tonmana_{selected_row_idx}', []),
                    st.session_state.get(f'selected_japanese_{selected_row_idx}', []),
                    st.session_state.get(f'selected_logic_{selected_row_idx}', [])
                ]):
                    with st.expander("📌 適用された改善点", expanded=False):
                        if st.session_state.get(f'selected_tonmana_{selected_row_idx}'):
                            st.markdown("**🎨 トンマナ改善:**")
                            for imp in st.session_state[f'selected_tonmana_{selected_row_idx}']:
                                st.write(f"- {imp}")
                        
                        if st.session_state.get(f'selected_japanese_{selected_row_idx}'):
                            st.markdown("**📝 日本語改善:**")
                            for imp in st.session_state[f'selected_japanese_{selected_row_idx}']:
                                st.write(f"- {imp}")
                        
                        if st.session_state.get(f'selected_logic_{selected_row_idx}'):
                            st.markdown("**🔍 ロジック改善:**")
                            for imp in st.session_state[f'selected_logic_{selected_row_idx}']:
                                st.write(f"- {imp}")
    
    # 区切り線
    st.divider()
    
    # 個別校正結果のダウンロード
    st.header("📥 個別校正結果のダウンロード")
    
    # 校正済みのデータを収集
    corrected_data = []
    for idx in range(len(df)):
        if f'correction_done_{idx}' in st.session_state:
            row_data = df.iloc[idx].to_dict()
            
            # 校正結果を追加
            if f'corrections_{idx}' in st.session_state:
                corrections = st.session_state[f'corrections_{idx}']
                
                # スコアを追加
                row_data['トンマナスコア'] = corrections.get('tonmana', {}).get('score', 0)
                row_data['日本語スコア'] = corrections.get('japanese', {}).get('score', 0)
                row_data['ロジックスコア'] = corrections.get('logic', {}).get('score', 0)
                row_data['総合スコア'] = (
                    row_data['トンマナスコア'] + 
                    row_data['日本語スコア'] + 
                    row_data['ロジックスコア']
                )
                
                # 改善点を追加
                improvements = []
                if corrections.get('tonmana', {}).get('improvements'):
                    improvements.extend([f"【トンマナ】{imp}" for imp in corrections['tonmana']['improvements']])
                if corrections.get('japanese', {}).get('improvements'):
                    improvements.extend([f"【日本語】{imp}" for imp in corrections['japanese']['improvements']])
                if corrections.get('logic', {}).get('improvements'):
                    improvements.extend([f"【ロジック】{imp}" for imp in corrections['logic']['improvements']])
                
                row_data['改善点'] = '\n'.join(improvements) if improvements else ''
                
                # 総合校正結果を追加
                if f'comprehensive_result_{idx}' in st.session_state:
                    row_data['総合校正結果'] = st.session_state[f'comprehensive_result_{idx}']
                else:
                    row_data['総合校正結果'] = ''
                
            corrected_data.append(row_data)
    
    if corrected_data:
        # スコアサマリー表示
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        
        # データフレームに変換
        result_df = pd.DataFrame(corrected_data)
        
        # 平均スコア計算
        avg_tonmana = result_df['トンマナスコア'].mean() if 'トンマナスコア' in result_df else 0
        avg_japanese = result_df['日本語スコア'].mean() if '日本語スコア' in result_df else 0
        avg_logic = result_df['ロジックスコア'].mean() if 'ロジックスコア' in result_df else 0
        
        with col_sum1:
            st.metric("🎨 トンマナ平均", f"{avg_tonmana:.2f}/5")
        with col_sum2:
            st.metric("📝 日本語平均", f"{avg_japanese:.2f}/5")
        with col_sum3:
            st.metric("🔍 ロジック平均", f"{avg_logic:.2f}/5")
        
        st.success(f"✅ {len(corrected_data)}件の校正済みデータがあります")
        
        # プレビュー表示
        with st.expander("📊 校正済みデータのプレビュー", expanded=False):
            display_columns = ['id', '質問', 'トンマナスコア', '日本語スコア', 'ロジックスコア', '総合スコア']
            # スコアに応じて色付け
            styled_df = result_df[display_columns].style.applymap(
                lambda x: 'background-color: #e8f5e9' if isinstance(x, (int, float)) and x >= 4 else 
                         'background-color: #fff3e0' if isinstance(x, (int, float)) and x >= 2 else 
                         'background-color: #ffebee' if isinstance(x, (int, float)) and x < 2 else '',
                subset=['トンマナスコア', '日本語スコア', 'ロジックスコア', '総合スコア']
            )
            st.dataframe(styled_df)
        
        # CSVダウンロードボタン
        output_buffer = io.StringIO()
        result_df.to_csv(output_buffer, index=False, encoding='utf-8-sig')
        
        col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
        with col_dl2:
            st.download_button(
                label="📥 個別校正結果をCSVでダウンロード",
                data=output_buffer.getvalue(),
                file_name=f"mimiko_individual_corrections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )
    else:
        st.info("まだ校正済みのデータがありません。上記で個別に校正を実行してください。")
    
    # 区切り線
    st.divider()
    
    # 一括処理ボタン
    with st.container():
        st.header("🚀 一括処理")
        
        if len(df) > 20:
            st.warning(f"⚠️ 大量のデータ（{len(df)}件）の一括処理には時間がかかります")
        
        col_batch1, col_batch2, col_batch3 = st.columns([1, 2, 1])
        with col_batch2:
            if st.button("🎯 全データを一括校正", type="secondary", use_container_width=True):
                # 結果を保存するための列を追加
                df['トンマナスコア'] = 0
                df['日本語スコア'] = 0
                df['ロジックスコア'] = 0
                df['総合スコア'] = 0
                df['改善点'] = ""
                df['総合校正結果'] = ""
                
                # 設定の準備
                current_project_id = vertex_ai_project_id
                current_location = vertex_ai_location
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
                    if st.session_state.get('enable_tonmana', True):
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
                            current_service_account,
                            thinking_budget=thinking_budget
                        )
                        
                        tonmana_json = parse_json_response(tonmana_result) if tonmana_result else None
                        if tonmana_json:
                            df.at[index, 'トンマナスコア'] = tonmana_json.get('score', 0)
                            improvements = tonmana_json.get('improvements', [])
                            if improvements:
                                df.at[index, '改善点'] += f"【トンマナ】{', '.join(improvements)}\n"
                    else:
                        df.at[index, 'トンマナスコア'] = 5
            
                    # 2. 日本語校正
                    if st.session_state.get('enable_japanese', False):
                        japanese_result = call_gemini(
                            japanese_prompt,
                            current_answer,
                            selected_model,
                            current_project_id,
                            current_location,
                            current_service_account,
                            thinking_budget=thinking_budget
                        )
                        
                        japanese_json = parse_json_response(japanese_result) if japanese_result else None
                        if japanese_json:
                            df.at[index, '日本語スコア'] = japanese_json.get('score', 0)
                            improvements = japanese_json.get('improvements', [])
                            if improvements:
                                df.at[index, '改善点'] += f"【日本語】{', '.join(improvements)}\n"
                    else:
                        df.at[index, '日本語スコア'] = 5
            
                    # 3. ロジック校正
                    if st.session_state.get('enable_logic', True):
                        # キーワードの詳細情報を取得
                        keyword_details = get_keyword_details(keywords)
                        
                        # 元キーワードとアレンジキーワードを取得
                        original_keywords = []
                        arrange_keywords = []
                        
                        # CSVの列から元キーワードとアレンジキーワードを探す
                        for col in df.columns:
                            if '元キーワード' in col and pd.notna(row[col]):
                                original_keywords.append(row[col])
                            elif 'アレンジキーワード' in col and pd.notna(row[col]):
                                arrange_keywords.append(row[col])
                        
                        # キーワード詳細情報をJSON形式で整形
                        keyword_info = json.dumps(keyword_details, ensure_ascii=False, indent=2)
                        
                        logic_message = f"""質問: {current_question}

使用キーワード:
{keyword_info}

元キーワード: {', '.join(original_keywords) if original_keywords else 'なし'}

アレンジキーワード: {', '.join(arrange_keywords) if arrange_keywords else 'なし'}

回答: {current_answer}
"""
                        logic_result = call_gemini(
                            logic_prompt,
                            logic_message,
                            selected_model,
                            current_project_id,
                            current_location,
                            current_service_account,
                            thinking_budget=thinking_budget
                        )
                        
                        logic_json = parse_json_response(logic_result) if logic_result else None
                        if logic_json:
                            df.at[index, 'ロジックスコア'] = logic_json.get('score', 0)
                            improvements = logic_json.get('improvements', [])
                            if improvements:
                                df.at[index, '改善点'] += f"【ロジック】{', '.join(improvements)}\n"
                    else:
                        df.at[index, 'ロジックスコア'] = 5
            
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