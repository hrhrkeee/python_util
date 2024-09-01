import streamlit as st

def on_value_change():
    print(f"何かが変更されました")

def get_default_input_dict():
    return {
        "年齢": {
            "input_count": 1,
            "inputs": [{
                "max" : 20,
                "min" : 10,
                "percent" : 100,
            }],
        },
        "年齢（範囲指定）": {
            "input_count": 1,
            "inputs": [{
                "range" : (10, 20),
                "percent" : 100,
            }],
        }
    }

if "input_dict" not in st.session_state:
    st.session_state.input_dict = get_default_input_dict()

def add_input(input_label_name):
    st.session_state.input_dict[input_label_name]["input_count"] += 1
    st.session_state.input_dict[input_label_name]["inputs"].append(
        get_default_input_dict()[input_label_name]["inputs"][0]
    )
    return

def remove_input(input_label_name, index):
    if st.session_state.input_dict[input_label_name]["input_count"] == 1:
        return

    st.session_state.input_dict[input_label_name]["inputs"].pop(index)
    st.session_state.input_dict[input_label_name]["input_count"] -= 1
    return

def age_input(input_dict, input_label_name="年齢"):
    with st.expander(input_label_name, expanded=False):
        cols = st.columns([3,3,1,2,1])

        labels = ["最小値", "最大値", "", "割合", "削除"]
        for col, label in zip(cols, labels):
            col.write(label)

        for i in range(input_dict["input_count"]):
            with cols[0]:
                input_dict["inputs"][i]["min"] = st.number_input(
                    label="Insert a number", 
                    value=get_default_input_dict()["年齢"]["inputs"][0]["min"], 
                    min_value=0, 
                    max_value=100, 
                    placeholder="Type a number...",
                    key=f"input_{input_label_name}_min_{i}",
                    label_visibility="collapsed",
                )
            with cols[1]:
                input_dict["inputs"][i]["max"] = st.number_input(
                    label="Insert a number", 
                    value=get_default_input_dict()["年齢"]["inputs"][0]["max"], 
                    min_value=0, 
                    max_value=100, 
                    placeholder="Type a number...",
                    key=f"input_{input_label_name}_max_{i}",
                    label_visibility="collapsed",
                )
            with cols[2]:
                st.write("")
            with cols[3]:
                input_dict["inputs"][i]["percent"] = st.number_input(
                    label="Insert a number", 
                    value=get_default_input_dict()["年齢"]["inputs"][0]["percent"], 
                    min_value=1, 
                    max_value=100, 
                    placeholder="Type a number...",
                    key=f"input_{input_label_name}_percent_{i}",
                    label_visibility="collapsed",
                )
            with cols[4]:
                st.button("🗑️", key=f"remove_{input_label_name}_{i}", on_click=remove_input, args=(input_label_name,i,))

        st.button("追加", key=f"add_{input_label_name}", on_click=add_input, args=(input_label_name,))

    return input_dict

def age_input2(input_dict, input_label_name="年齢"):
    with st.expander(input_label_name, expanded=False):
        cols = [6,1,2,1]

        labels = ["範囲(歳)", "", "割合(%)", "削除"]
        for col, label in zip(st.columns(cols), labels):
            col.write(label)

        for i in range(input_dict["input_count"]):
            st_columns = st.columns(cols)
            with st.container():
                with st_columns[0]:
                    input_dict["inputs"][i]["range"] = st.slider(
                        f"Select range for Input {i+1}",
                        min_value=0,
                        max_value=100,
                        value=get_default_input_dict()["年齢（範囲指定）"]["inputs"][0]["range"],
                        key=f"range_{input_label_name}_range_{i}",
                        label_visibility="collapsed",
                    )
                with st_columns[1]:
                    st.write("")
                with st_columns[2]:
                    input_dict["inputs"][i]["percent"] = st.number_input(
                        label="Insert a number", 
                        value=get_default_input_dict()["年齢（範囲指定）"]["inputs"][0]["percent"], 
                        min_value=1, 
                        max_value=100, 
                        placeholder="Type a number...",
                        key=f"input_{input_label_name}_percent_{i}",
                        label_visibility="collapsed",
                    )
                with st_columns[3]:
                    st.button("🗑️", key=f"remove_{input_label_name}_{i}", on_click=remove_input, args=(input_label_name,i,))

        st.button("追加", key=f"add_{input_label_name}", on_click=add_input, args=(input_label_name,))

    return input_dict

def main():

    for input_label_name in st.session_state.input_dict.keys():
        
        if input_label_name == "年齢":
            st.session_state.input_dict[input_label_name] = age_input(st.session_state.input_dict[input_label_name], input_label_name)
        
        if input_label_name == "年齢（範囲指定）":
            st.session_state.input_dict[input_label_name] = age_input2(st.session_state.input_dict[input_label_name], input_label_name)


    st.write(st.session_state.input_dict)

    pass

if __name__ == '__main__':

    main()

    on_value_change()
