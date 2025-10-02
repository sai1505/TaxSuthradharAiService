import boto3
import os
from dotenv import load_dotenv

# --- Cloudflare R2 Configuration ---
# Load environment variables from the .env file
load_dotenv()

# Load R2 credentials and configuration from environment
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")

# Validate that all required environment variables are set
if not all([CLOUDFLARE_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME]):
    raise ValueError("ERROR: Missing one or more required Cloudflare R2 environment variables in the .env file.")

# Construct the unique R2 endpoint URL
R2_ENDPOINT_URL = f"https://{CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com"

# Initialize the S3 client configured for Cloudflare R2
# This client object is the bridge between our Python app and R2.
s3_client = boto3.client(
    's3',
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name="auto" # This must be 'auto' for R2
)

print("✅ S3 Client configured successfully for Cloudflare R2.")
