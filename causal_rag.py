import os
import io
import sys
import traceback
import pandas as pd
from gemini import completion
from google.cloud import bigquery
import constant

def identify_company_name(question):
    """
    Identifies the company name mentioned in the question using LLMs.
    If no specific company is mentioned, returns "unknown".
    """
    company_prompt = f"""
    You are a helpful assistant that identifies company names from financial analysis questions.
    Please identify the company name mentioned in the question below. If no specific company is mentioned, return "Unknown".
    Question: "{question}"
    """
    company_prompt = company_prompt.format(question=question)
    company_name = completion(
        company_prompt,
        temperature=0.1
    ).strip()
    return company_name.lower()

def classify_question(question):
    """
    Classifies the user's question into one of the predefined categories using LLMs.
    Categories: balance_sheet, income_statement, others
    """
    classification_prompt = f"""
    You are a helpful assistant that classifies financial analysis questions.
    Please classify question below into one of the following items and return the item name without explanation.
    - balance_sheet​​: Applicable to analytical questions involving balance sheet items (such as assets, liabilities, shareholders equity, etc.).
    ​- ​income_statement​​: Applicable to analytical questions involving income statement items (such as revenue, expenses, profit, etc.).
    ​​- others​​: Applicable to questions that do not involve causal analysis or are too broad to be classified under either of the above two categories.
    Question: "{question}"
    """
    classification_prompt = classification_prompt.format(question=question)
    purpose = completion(
        classification_prompt,
        temperature=0.1
    ).strip()
    return purpose

def retrieve_causal_info(purpose):
    """
    Retrieves the causal graph and sentiments among variables from BigQuery based on the purpose."""
    client = bigquery.Client(project=constant.GCP_PROJECT_ID)
    query = f"""
        SELECT causal_graph, sentiment
        FROM `{constant.GCP_PROJECT_ID}.{constant.BIGQUERY_DATASET}.{constant.TAB_CAUSAL_CONFIG}`
        WHERE purpose = @purpose
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("purpose", "STRING", purpose)
        ]
    )
    query_job = client.query(query, job_config=job_config)
    result = query_job.result()
    row = next(result, None)
    if row:
        return row.causal_graph, row.sentiment
    else:
        return None, None

def analyze_facts(question, purpose, causal_graph, sentiments):
    """
    Analyzes facts from data based on the question, causal graph and sentiments among variables.
    """
    annual_csv_path = os.path.join('data',  purpose + '.csv')
    annual_df = pd.read_csv(annual_csv_path)
    sample = annual_df.head(1)

    analytics_prompt = """
    You are a data analyst that good at data analysis and statistics
    Based on the causal graph, sentiments among variables and data schema provided, please generate code to list facts which can help answer the following question.
    ## REQUIREMENTS
    - Load data from folder {path} based on the schema, sample and question provided
    - pay attention to the causal graph and sentiments among variables
    - analyze as deep as possible to find the root cause
    - only output code
    - the code should be runnable in Python environment
    - DO NOT get to conclusions, only list facts you found from data
    - DO NOT use pd.StringIO function, use io.StringIO instead
    - DO NOT make up any data, read data from the path provided
    - Print facts you found
    
    ## RULES
    - all data in data source are numeric, including year column
    - convert FYxx to numeric year, e.g. FY18 to 2018

    ## Causal Graph
    {graph}

    ## Sentiments
    {sentiment}

    ## Data Schema & Sample
    {sample}

    ## Question
    {question}
    """

    MAX_LOOP = 3
    analyzed_facts = None
    error_msg = None
    code_generated = None

    for i in range(MAX_LOOP):
        #print(f"\n=================4.1 Code Generation Attempt {i+1} =================\n")
        
        analytics_prompt = analytics_prompt.format(
                graph=causal_graph,
                sentiment=sentiments,
                path=annual_csv_path,
                sample=sample.to_string(index=False),
                question=question
        )

        code_generated = completion(
            analytics_prompt,
            temperature=0.9
        )

        code_generated = code_generated.replace("```python", "").replace("```", "").strip()
        #print(code_generated)

        stdout_buffer = io.StringIO()
        try:
            sys_stdout = sys.stdout
            sys.stdout = stdout_buffer

            exec_namespace = {}
            exec(code_generated, exec_namespace)
            analyzed_facts = stdout_buffer.getvalue()
            error_msg = None
            break
        except Exception as e:
            error_msg = f"{str(e)}\nTraceback:\n{traceback.format_exc()}"
            error_msg = error_msg.replace("'", "")
            analyzed_facts = f"Error: {error_msg}"
        finally:
            sys.stdout = sys_stdout
            #print("\n=================4.2 Facts =================\n")
            #print(analyzed_facts) 
    return analyzed_facts

def get_insights(question, causal_graph, sentiments, analyzed_facts):
    """
    Generates insights using LLMs based on the causal graph, sentiments among variables and analyzed facts
    """
    insights_prompt = """
    You are a Causal Model that is good at answering Why question with cause-effect and facts
    Please answer the following question based on the causal graph, sentiments among variables and facts provided:
    ## REQUIREMENTS
    - pay attention to the causal graph and sentiments among variables
    - find the root cause as deep as possible
    - use facts provided to support your answer

    ## Causal Graph 
    {graph}

    ## Sentiments 
    {sentiment} 

    ## Facts
    {facts}

    ## Question
    {question}
    """

    insights_prompt = insights_prompt.format(graph=causal_graph, 
                            sentiment=sentiments, 
                            facts=analyzed_facts,
                            question=question)
    insights = completion(
            insights_prompt,
            temperature=0.9
    )
    return insights

def vector_search(question: str):
    """
    Uses LLM to dynamically generate the entire SQL query for searching relevant chunks,
    including pre-filtering by report_year and company_name based on the question.
    """
    client = bigquery.Client(project=constant.GCP_PROJECT_ID)
    print(f"--- Performing Vector Search for Question: '{question}' ---")
    
    # Use LLM to generate the full SQL query with appropriate filters
    sql_prompt = f"""
    You are a helpful assistant that writes BigQuery SQL for semantic search in a table of PDF chunks.

    ## WORKFLOW
    1. Analyze the question to determine if it mentions a specific company_name and/or report_year (which could be a specific year or a range).
    2. If company_name and/or report_year is mentioned, extract their values for filtering.
    3. Generate a SQL query that uses the VECTOR_SEARCH and pre-filters the table by company_name and report_year if applicable.
    
    ## REQUIREMENTS
    Given the question below, generate a SQL query that:
    - Pre-filters the `{constant.GCP_PROJECT_ID}.{constant.BIGQUERY_DATASET}.{constant.TAB_PDF_CHUNKS_EMBEDDING}` table by company_name and report_year (which could be a specific year or a range) if mentioned in the question.
    - Uses VECTOR_SEARCH to find the top 5 most relevant chunks based on the question embedding.
    - Uses the embedding model `{constant.GCP_PROJECT_ID}.{constant.BIGQUERY_DATASET}.{constant.EMBEDDING_MODEL}`.
    - Returns the chunk_text and distance.
    - Uses parameterized queries for all user inputs.
    - If company_name or report_year is not mentioned, do not filter on them.
    - The parameter for the search query text should be @query.
    - Generate value based on question for company_name.
    - Generate value or a range based on question for report_year as appropriate.
    - Only output the SQL code, no explanation.

    ## EXAMPLE SQL
    For question "What is the revenue of Salesforce in 2025?", the SQL should look like:
    ```SELECT
        base.chunk_text,
        base.report_year,
        base.company_name,
        distance
        FROM
            VECTOR_SEARCH(
                (
                    SELECT *
                    FROM `{constant.GCP_PROJECT_ID}.{constant.BIGQUERY_DATASET}.{constant.TAB_PDF_CHUNKS_EMBEDDING}`
                    WHERE report_year = 2025
                    AND company_name = 'salesforce'
                ),
                'embedding',
                (
                    SELECT ml_generate_embedding_result
                    FROM ML.GENERATE_EMBEDDING(
                        MODEL `{constant.GCP_PROJECT_ID}.{constant.BIGQUERY_DATASET}.{constant.EMBEDDING_MODEL}`,
                        (SELECT @query AS content)
                    )
                ),
                top_k => 8,
                distance_type => 'COSINE'
            );
    ```
    Question: "{question}"
    """

    sql_code = completion(sql_prompt, temperature=0.1).strip()
    # Remove code block markers if present
    sql_code = sql_code.replace("```sql", "").replace("```", "").strip()
    #print("--- Generated SQL ---\n", sql_code)

    # Only set the @query parameter, as company_name and report_year are already in the generated SQL
    query_params = [bigquery.ScalarQueryParameter("query", "STRING", question)]
    #print("Executing search query with parameters: {'query': question}")
    results = client.query(sql_code, job_config=bigquery.QueryJobConfig(query_parameters=query_params)).result()
    chunks = [row.chunk_text for row in results]
    return chunks

def finalize_answer(question, insights, chunks):
    """
    Generates the final answer using LLMs based on the insights and supporting evidence (chunks).
    """
    final_prompt = """
    You are a financial analyst assistant.
    Based on the insights below and the supporting evidence (chunks), please generate a comprehensive and concise final answer to the user's question.
    ## REQUIREMENTS
    - Use the insights as your main reasoning
    - Please keep insights as-is, do not modify it
    - Use the supporting evidence (chunks) to back up insights
    - If the insights and chunks conflict, prioritize insights
    - Make sure your answer is clear 

    ## Insights
    {insights}

    ## Supporting Evidence (Chunks)
    {chunks}

    ## Question
    {question}
    """

    final_prompt = final_prompt.format(
        insights=insights,
        chunks="\n".join([str(chunk) for chunk in chunks]),
        question=question
    )

    final_answer = completion(
        final_prompt,
        temperature=0.7
    )

    return final_answer
    
def main(question):
    
    company_name = None
    purpose = None
    causal_graph = None 
    sentiments = None
    analyzed_facts = None
    insights = None
    final_answer = None
    chunks = None

    # 1. Identify company name from the question
    print("\n=================1. Company Identification =================\n")
    company_name = identify_company_name(question)
    print(f"Identified company name: {company_name}")
    if company_name == "unknown":
        print("No specific company mentioned in the question. Exiting.")
        return
    
    # 2. Classify user question with LLMs to determine the purpose of this question
    print("\n=================2. Question Classification =================\n")
    purpose = classify_question(question)
    print(f"'{question}' is classified as a question in  '{purpose}' area.")
    
    if purpose not in ['balance_sheet', 'income_statement']:
        print(f"Question '{question}' is classified as '{purpose}', which may not be supported.")
        return
    
    #3. Retrieve causal graph and sentiments among variables from BigQuery based on the purpose
    print("\n=================3. Causal Information Retrieval =================\n")
    causal_graph, sentiments = retrieve_causal_info(purpose)
    if not causal_graph or not sentiments:
        print(f"No causal information found for purpose '{purpose}'.")
        return

    #4. Analyze facts from data based on the question, causal graph and sentiments among variables
    print("\n=================4. Facts Analysis =================\n")
    analyzed_facts = analyze_facts(question, purpose, causal_graph, sentiments)
    if not analyzed_facts:
        print("Failed to analyze facts from data.")
        return
    print("Fact analyzed... ")

    #5. Generates insights using LLMs based on the causal graph, sentiments among variables and analyzed facts
    print("\n=================5. Insights Generation =================\n")
    insights = get_insights(question, causal_graph, sentiments, analyzed_facts)
    print("Insights Generated...")

    #6. Use vector search to find the most relevant chunks from the PDF embeddings table
    print("\n=================6. Vector Search =================\n")
    chunks = vector_search(question)
    print(f"Retrieved {len(chunks)} relevant chunks from vector search.")

    #7. Generates the final answer using LLMs based on the insights and supporting evidence (chunks).
    print("\n=================7. Final Answer Generation =================\n")
    final_answer = finalize_answer(question, insights, chunks)
    print(final_answer)
        

if __name__ == "__main__":
    default_question = "What were the main drivers behind the profit margin improvement of Salesforce in 2025 compared to last year?"
    question = input(f"Please enter your question (press Enter to use the default question: '{default_question}'): ")
    if not question.strip():
        question = default_question
    main(question)