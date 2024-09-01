import streamlit as st

# セッションステートに入力項目のリストを保持
if "inputs" not in st.session_state:
    st.session_state.inputs = []

# 新しい入力項目を追加する関数
def add_input():
    st.session_state.inputs.append({"range": (0, 100), "percentage": 0})

# アコーディオンのタイトルを更新する関数
def update_title():
    titles = [
        f"Input {i+1}: Range {inp['range'][0]} - {inp['range'][1]}, {inp['percentage']}%"
        for i, inp in enumerate(st.session_state.inputs)
    ]
    st.session_state.title = " | ".join(titles) if titles else "Adjust Values"

# 初期タイトル設定
if "title" not in st.session_state:
    st.session_state.title = "Adjust Values"

# アコーディオンの中に入力項目を追加
with st.expander(st.session_state.title, expanded=True):
    # 現在の入力項目を表示
    for i, inp in enumerate(st.session_state.inputs):
        col1, col2 = st.columns([0.7, 0.3])
        
        with col1:
            inp["range"] = st.slider(
                f"Select range for Input {i+1}",
                min_value=0,
                max_value=100,
                value=inp["range"],
                key=f"range_{i}"
            )

        with col2:
            inp["percentage"] = st.number_input(
                f"Enter percentage for Input {i+1}",
                min_value=0,
                max_value=100,
                value=inp["percentage"],
                key=f"percentage_{i}"
            )
    
    # 「追加」ボタンが押された場合の処理
    if st.button("追加"):
        add_input()

# タイトルを更新
update_title()
