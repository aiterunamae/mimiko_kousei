import streamlit as st
import anthropic
import json
import toml
import os
from pathlib import Path

# Load secrets
secrets_path = Path(__file__).parent / "secrets.toml"
secrets = toml.load(secrets_path)
api_key = secrets["anthropic_api_key"]
default_model = secrets["default_model"]

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=api_key)

# Load correction prompts
def load_prompt(filename):
    prompt_path = Path(__file__).parent / filename
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

# Load all prompts
try:
    tonmana_prompt = load_prompt("トンマナ校正プロンプト.txt")
    japanese_prompt = load_prompt("日本語校正プロンプト.txt")
    pattern_prompt = load_prompt("パターン類似度校正プロンプト.txt")
    logic_prompt = load_prompt("ロジック校正プロンプト.txt")
    comprehensive_prompt = load_prompt("総合校正プロンプト.txt")
except Exception as e:
    st.error(f"Error loading prompts: {e}")
    st.stop()

# Function to call Claude API
def call_claude(prompt, user_message):
    try:
        response = client.messages.create(
            model=default_model,
            max_tokens=1000,
            system=prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        return response.content[0].text
    except Exception as e:
        st.error(f"Error calling Claude API: {e}")
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
st.title("AI占い師校正システム")

# Input section
st.header("入力")
user_question = st.text_area("ユーザーからの質問", height=100)
user_tensei = st.selectbox(
    "ユーザーの通変星",
    [
        "比肩（ひけん）", "劫財（ごうざい）", "食神（しょくじん）", "傷官（しょうかん）",
        "偏財（へんざい）", "正財（せいざい）", "偏官（へんかん）", "正官（せいかん）",
        "偏印（へんいん）", "印綬（いんじゅ）"
    ]
)
ai_answer = st.text_area("AI占い師の回答", height=150)

# Initialize session state
if 'corrections' not in st.session_state:
    st.session_state.corrections = {}
if 'tonmana_problems' not in st.session_state:
    st.session_state.tonmana_problems = []
if 'japanese_improvements' not in st.session_state:
    st.session_state.japanese_improvements = []
if 'pattern_improvements' not in st.session_state:
    st.session_state.pattern_improvements = []
if 'logic_improvements' not in st.session_state:
    st.session_state.logic_improvements = []
if 'correction_done' not in st.session_state:
    st.session_state.correction_done = False
if 'user_input' not in st.session_state:
    st.session_state.user_input = {"question": "", "tensei": "", "answer": ""}

# Process button
if st.button("校正を実行") or st.session_state.correction_done:
    if not user_question or not ai_answer:
        st.error("質問と回答を入力してください")
    else:
        # Save user input to session state
        st.session_state.user_input = {
            "question": user_question,
            "tensei": user_tensei,
            "answer": ai_answer
        }
        
        # Set correction_done flag to true
        st.session_state.correction_done = True
        
        # Only call APIs if not already done
        if 'tonmana_result' not in st.session_state:
            with st.spinner("トンマナ校正中..."):
                # 1. トンマナ校正
                tonmana_message = f"""##QUESTION##
{user_question}

##ANSWER_CAND##
{ai_answer}
"""
                tonmana_result = call_claude(tonmana_prompt, tonmana_message)
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
                japanese_result = call_claude(japanese_prompt, japanese_message)
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
        
        if 'pattern_result' not in st.session_state:
            with st.spinner("パターン類似度校正中..."):
                # 3. パターン類似度校正
                pattern_message = f"""質問: {user_question}
回答: {ai_answer}
"""
                pattern_result = call_claude(pattern_prompt, pattern_message)
                if pattern_result:
                    pattern_json = parse_json_response(pattern_result)
                    if pattern_json:
                        st.session_state.pattern_result = pattern_result
                        st.session_state.pattern_json = pattern_json
                        
                        # Store in session state
                        st.session_state.corrections["pattern"] = {
                            "score": pattern_json.get('score', 'N/A'),
                            "improvements": pattern_json.get('improvements', [])
                        }
        
        if 'logic_result' not in st.session_state and logic_prompt.strip():
            with st.spinner("ロジック校正中..."):
                # 4. ロジック校正
                logic_message = f"""質問: {user_question}
通変星: {user_tensei}
回答: {ai_answer}
"""
                logic_result = call_claude(logic_prompt, logic_message)
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
        
        # 3. パターン類似度校正結果
        if 'pattern_json' in st.session_state:
            st.subheader("パターン類似度校正結果")
            pattern_json = st.session_state.pattern_json
            st.write(f"スコア: {pattern_json.get('score', 'N/A')}/5")
            
            improvements = pattern_json.get('improvements', [])
            if improvements:
                st.write("改善点:")
                for i, improvement in enumerate(improvements):
                    # Create a unique key for each checkbox
                    checkbox_key = f"pattern_cb_{i}"
                    
                    # Check if the improvement is in the selected improvements list
                    is_selected = improvement in st.session_state.pattern_improvements
                    
                    # Display checkbox
                    checked = st.checkbox(improvement, key=checkbox_key, value=is_selected)
                    
                    # Update selected improvements list based on checkbox state
                    if checked and improvement not in st.session_state.pattern_improvements:
                        st.session_state.pattern_improvements.append(improvement)
                    elif not checked and improvement in st.session_state.pattern_improvements:
                        st.session_state.pattern_improvements.remove(improvement)
            else:
                st.write("改善点はありません")
        
        # 4. ロジック校正結果
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
    st.experimental_rerun()

# Comprehensive correction button
if st.session_state.get("show_comprehensive_button", False):
    st.header("総合校正")
    if st.button("選択した改善点で総合校正を実行") or 'comprehensive_result' in st.session_state:
        if 'comprehensive_result' not in st.session_state:
            with st.spinner("総合校正中..."):
                # Prepare selected improvements
                selected_improvements = {
                    "トンマナ校正": st.session_state.get("tonmana_problems", []),
                    "日本語校正": st.session_state.get("japanese_improvements", []),
                    "パターン類似度校正": st.session_state.get("pattern_improvements", []),
                    "ロジック校正": st.session_state.get("logic_improvements", [])
                }
                
                # Prepare scores
                scores = {}
                if "tonmana" in st.session_state.corrections:
                    scores["トンマナ校正"] = st.session_state.corrections["tonmana"]["score"]
                if "japanese" in st.session_state.corrections:
                    scores["日本語校正"] = st.session_state.corrections["japanese"]["score"]
                if "pattern" in st.session_state.corrections:
                    scores["パターン類似度校正"] = st.session_state.corrections["pattern"]["score"]
                if "logic" in st.session_state.corrections:
                    scores["ロジック校正"] = st.session_state.corrections["logic"]["score"]
                
                # Create message for comprehensive correction
                comprehensive_message = f"""AI占い師の回答:
{st.session_state.user_input["answer"]}

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
                
                # Call Claude for comprehensive correction
                comprehensive_result = call_claude(comprehensive_prompt, comprehensive_message)
                if comprehensive_result:
                    # Save the result to session state
                    st.session_state.comprehensive_result = comprehensive_result
        
        # Display comprehensive correction result
        if 'comprehensive_result' in st.session_state:
            st.subheader("総合校正結果")
            st.write(st.session_state.comprehensive_result)
