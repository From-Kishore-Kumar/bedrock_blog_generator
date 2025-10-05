import boto3
import botocore.config
import json
from datetime import datetime
import re
import logging

# === Logger setup ===
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# === Constants ===
S3_BUCKET = "aws-bedrock-stuffs"
S3_PREFIX = "blog_output"
REGION = "ap-south-1"
MODEL_ID = "meta.llama3-8b-instruct-v1:0"
MAX_GEN_TOKENS = 384

# === AWS Clients ===
s3 = boto3.client("s3")
bedrock_runtime = boto3.client(
    "bedrock-runtime",
    region_name=REGION,
    config=botocore.config.Config(read_timeout=300, retries={'max_attempts': 3})
)

# === Helper Functions ===

def build_prompt(topic: str) -> str:
    return f"""
<s>[INST]
You are a professional blog writer.

Write a clear, concise, and well-structured 200-word blog post on the topic: "{topic}".
Format it using:
- A title in heading style
- Subheadings for sections
- Bullet points where applicable
Avoid repeating the prompt or adding instructional tokens. End the blog naturally.
[/INST]
""".strip()


def clean_llama_output(raw_text: str) -> str:
    """
    Cleans model output by removing special tokens but keeping the text inside.
    """
    cleaned = re.sub(r"</?s>", "", raw_text)
    cleaned = re.sub(r"\[/?INST\]", "", cleaned)
    cleaned = cleaned.strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def generate_blog(topic: str) -> str:
    """
    Calls Bedrock to generate a blog post for a given topic.
    """
    prompt = build_prompt(topic)
    payload = {
        "prompt": prompt,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_gen_len": MAX_GEN_TOKENS
    }

    try:
        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload)
        )
        response_body = response["body"].read()
        response_json = json.loads(response_body)
        raw_blog = response_json.get("generation", "")
        return clean_llama_output(raw_blog)

    except Exception as e:
        logger.error(f"Bedrock model invocation failed: {e}")
        return ""


def save_to_s3(content: str) -> str:
    """
    Saves content to S3 with a timestamped key.
    Returns the S3 path or raises on error.
    """
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    s3_key = f"{S3_PREFIX}/{timestamp}.txt"

    try:
        s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=content.encode("utf-8"))
        logger.info(f"Blog saved to S3: s3://{S3_BUCKET}/{s3_key}")
        return f"s3://{S3_BUCKET}/{s3_key}"
    except Exception as e:
        logger.error(f"Failed to save to S3: {e}")
        raise


# === Lambda Handler ===

def lambda_handler(event, context):
    """
    AWS Lambda entry point.
    Expects event["body"] to contain JSON: { "topic": "..." }
    """
    try:
        body = json.loads(event.get("body", "{}"))
        topic = body.get("topic")

        if not topic:
            return {
                "statusCode": 400,
                "headers": {"Access-Control-Allow-Origin": "*"},
                "body": json.dumps({"error": "Missing 'topic' in request body"})
            }

        logger.info(f"Received topic: {topic}")

        blog = generate_blog(topic)

        if not blog or len(blog) < 100:
            logger.warning("Generated blog is too short or empty.")
            return {
                "statusCode": 500,
                "headers": {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "Content-Type"},
                "body": json.dumps({"error": "Blog generation failed or returned empty content."})
            }

        s3_path = save_to_s3(blog)

        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*", 
                "Access-Control-Allow-Headers": "Content-Type"
            },
            "body": blog
        }

    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")
        return {
            "statusCode": 500,
            "headers": {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "Content-Type"},
            "body": json.dumps({"error": "Internal server error."})
        }
