
from google.cloud import bigquery
import constant

def launch_client():
    """
    Initializes the Google BigQuery client.
    """
    # The client will automatically use the project set in your environment
    # or the one specified here.
    return bigquery.Client(project=constant.GCP_PROJECT_ID)

def completion(
        prompt,
        temperature=0.1,
    ):
    """
    Generates a response using BigQuery ML's ML.GENERATE_TEXT function.

    Note: Streaming is not supported with this method.
    """
    client = launch_client()
    #prompt = flatten_prompt_to_single_line(prompt)

    # Construct the SQL query
    query = f"""
        SELECT ml_generate_text_llm_result AS generated_text
        FROM
          ML.GENERATE_TEXT(
            MODEL `{constant.BIGQUERY_DATASET}.{constant.MODEL_ID}`,
            (SELECT @prompt_text AS prompt),
            STRUCT(
              {temperature} AS temperature,
              65535 AS max_output_tokens,
              TRUE AS flatten_json_output
            )
          )
    """
    # 2. Configure the query job to define the parameter's value
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("prompt_text", "STRING", prompt),
        ]
    )

    query_job = client.query(query, job_config=job_config)
    results = query_job.result()
    full_response = ""
    for row in results:
        full_response = row.generated_text
        break # We expect only one row for a single prompt

    return full_response
