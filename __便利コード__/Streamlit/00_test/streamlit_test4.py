import streamlit as st

if "input_dict" not in st.session_state:
    st.session_state.input_dict = {
        "test": {
            "input_count": 1,
            "inputs": [{
                "test_range": (10, 20),
            }],
        }
    }


def main():
    
    for i in range(st.session_state.input_dict["test"]["input_count"]):
        st.session_state.input_dict["test"]["inputs"][i]["test_range"] = st.slider(
            label="test",
            min_value=0,
            max_value=100,
            value=(10, 20),
            key=f"input_test_{i}",
        )

    st.write(st.session_state.input_dict) 

    pass

if __name__ == '__main__':
    main()