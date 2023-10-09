import pandas as pd
import streamlit as st
from huggingface_hub import InferenceClient

client = InferenceClient(token=st.secrets["hf_key"], model=st.secrets["model"])

st.subheader("Failure Mode Classification")

st.sidebar.info("Training Data")
uploaded_file = st.sidebar.file_uploader(
    "Please upload a csv file containing failure description and \
                                         failure codes. See example csv file below.",
    type=["csv"],
)
sample_train = pd.read_csv("train.csv")
st.sidebar.dataframe(sample_train, hide_index=True)

if uploaded_file is not None:
    train_csv = pd.read_csv(uploaded_file)
    st.dataframe(train_csv.head(10))
    failure_text = st.text_input(label="Please enter failure description", value="")

    if st.button("Get Failure Code"):

        newline = "\n"
        prompt_header = f"""
        Your answer should contain only the failure mode and nothing else. Valid failure modes are:

        {(newline).join(list(train_csv.iloc[:,1].unique()))}

        """
        in_context_learning = []
        for i, row in train_csv.iterrows():
            in_context_row = f"""
            Determine the failure mode associated with the following sentence:  
            sentence: {row.iloc[0]}
            Response: {row.iloc[1]}
            """
            in_context_learning.append(in_context_row)

        input_description = f"""
            Determine the failure mode associated with the following sentence:  
            sentence: {failure_text}
            Response:
            """
        in_context_learning.append(input_description)
        prompt = newline.join(in_context_learning) + newline + prompt_header

        response = client.text_generation(prompt=prompt, max_new_tokens=5)
        c1, _, _ = st.columns(3)
        with c1:
            st.success(response)
else:
    st.warning(
        "Please upload a csv file (training data) following the instructions in the side-bar"
    )
