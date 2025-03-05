from azure.storage.blob import BlobServiceClient
import os, shutil
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

local_path = os.path.expanduser("~/new_data")
if not os.path.exists(local_path):
    logger.info(f"Creating directory {local_path}")
    os.makedirs(local_path)

connection_string = f"DefaultEndpointsProtocol=https;AccountName=odyingest;AccountKey=;EndpointSuffix=core.windows.net"
container_name = "3droutput"
prefix_path = "xgrids/18918869513146821/output/developer_data/perspective"

logger.info(f"Connecting to Azure Storage account: odyingest")
logger.info(f"Container: {container_name}")
logger.info(f"Prefix path: {prefix_path}")

# Create the blob service client
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client(container_name)

# Only process actual files (blobs with size > 0)
downloaded = 0
skipped = 0
errors = 0

for blob in container_client.list_blobs(name_starts_with=prefix_path):
    if blob.size > 0:  # Only process actual files
        file_path = os.path.join(local_path, blob.name)
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if not os.path.exists(file_path):
            try:
                logger.info(f"Downloading {blob.name} (size: {blob.size/1024:.2f} KB)...")
                with open(file_path, "wb") as file:
                    data = container_client.download_blob(blob.name).readall()
                    file.write(data)
                downloaded += 1
                if downloaded % 10 == 0:
                    logger.info(f"Progress: {downloaded} files downloaded, {skipped} skipped, {errors} errors")
            except Exception as e:
                logger.error(f"Error downloading {blob.name}: {str(e)}")
                errors += 1
        else:
            logger.debug(f"Skipping {blob.name} - already exists")
            skipped += 1

logger.info("Download complete!")
logger.info(f"Final statistics:")
logger.info(f" - Files downloaded: {downloaded}")
logger.info(f" - Files skipped: {skipped}")
logger.info(f" - Errors encountered: {errors}")
