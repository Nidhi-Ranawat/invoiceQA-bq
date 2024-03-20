# from importBiginputToSqlite import df_cleaned
from dotenv import load_dotenv
from bq_sql_gen import get_response, get_response_df
import streamlit as st

load_dotenv()

def main():
    st.set_page_config(page_title="Ask your Invoice Database")
    st.header("Ask your Invoice database 📈")

    input = st.text_input("Ask a question about your Biginput: ")
    
    data = "Sorry! Info not found"
    if input is not None and input != "":
        if "list" in input.lower():
            with st.spinner(text="In progress..."):
                data = get_response_df(input)
                st.write(data)
        else:
            with st.spinner(text="In progress..."):
                print("not a list")
                data = get_response(input)
                st.write(data)
                
    st.write(" ")
    st.write("For eg : show me one random PNR number")

if __name__ == "__main__":
    main()
