import sys
import os
from langchain import PromptTemplate, OpenAI, LLMChain
from google.cloud import bigquery
import json
import argparse
from dotenv import load_dotenv

load_dotenv()
root = os.getcwd()
file_path = os.path.join(root, "credential.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = file_path

project = "data-23-24"
dataset = "invoice_dataset"
table_id = "bu18dec"
fully_qualified_table_id = f"{project}.{dataset}.{table_id}"

TEMPLATE = '''
Write a BigQuery SQL that achieves the following.
```
{{ content }}
```

The format of the target tables is as follows.
```json
{{ schema }}
```

Output the SQL in raw text.
    '''

TEMPLATE2 = '''
Write a case insensitive BigQuery SQL that achieves the following.
```
{{ content }} for 25 records max include following columns only Customer, `Passenger Name`, PNR, `Ticket No`, `Base Fare`, `Total Inv`, Agent, `Amt In INR` aways use wildcards for names in WHERE conditions.
```

The format of the target tables is as follows.
```json
{{ schema }}
```

Output the SQL in raw text.
    '''

def get_schema(table_name: str) -> str:
    client = bigquery.Client()
    table = client.get_table(table_name)
    project_id = table.project
    dataset_id = table.dataset_id
    table_id = table.table_id
    schema = list(map(lambda x: x.to_api_repr(), table.schema))
    return {'project':project_id,'dataset':dataset_id,'table':table_id,'schema':schema}

def get_schemas(table_names: list[str]):
    return json.dumps([get_schema(n) for n in table_names])

def predict(content: str, table_names: list[str], verbose: bool = False):
    prompt = PromptTemplate(
        input_variables=["content","schema"],
        template=TEMPLATE,
        template_format='jinja2',
    )
    llm_chain = LLMChain(
        llm=OpenAI(model="gpt-3.5-turbo-instruct",temperature=0), 
        prompt=prompt, 
        verbose=verbose,
    )
    return llm_chain.predict(content=content, schema=get_schemas(table_names))

def predict_df(content: str, table_names: list[str], verbose: bool = False):
    prompt = PromptTemplate(
        input_variables=["content","schema"],
        template=TEMPLATE2,
        template_format='jinja2',
    )
    llm_chain = LLMChain(
        llm=OpenAI(model="gpt-3.5-turbo-instruct",temperature=0), 
        prompt=prompt, 
        verbose=verbose,
    )
    return llm_chain.predict(content=content, schema=get_schemas(table_names))

def execute_bigquery_query(query):
    client = bigquery.Client()
    df = client.query(query).to_dataframe()
    return df

def get_response(input):
    query = predict(input, [fully_qualified_table_id], True)
    df = execute_bigquery_query(query)
    return df

def get_response_df(input):
    query = predict_df(input, [fully_qualified_table_id], True)
    print(query)
    df = execute_bigquery_query(query)
    return df
    

# if __name__ == '__main__':
#     project = "data-23-24"
#     dataset = "invoice_dataset"
#     table_id = "bu18dec"
#     fully_qualified_table_id = f"{project}.{dataset}.{table_id}"

#     print(predict("show me 1 random PNR number", [fully_qualified_table_id], True))

#     query = predict("show me 1 random PNR number", [fully_qualified_table_id], True)
#     df = execute_bigquery_query(query)
#     print(df)