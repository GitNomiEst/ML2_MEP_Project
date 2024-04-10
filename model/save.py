import os
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import argparse
from dotenv import load_dotenv

load_dotenv()

azure_connection_string = os.getenv("AZURE_CONNECTION_STRING")

try:
    print("Azure Blob Storage Python quickstart sample")

    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient.from_connection_string(azure_connection_string)

    
    exists = False
    suffix = 0
    containers = blob_service_client.list_containers(include_metadata=True)

    for container in containers:
        if container.name.startswith("asteroid-"):
                try:
                    current_suffix = int(container.name.split("-")[-1])
                    suffix = max(suffix, current_suffix)
                except ValueError:
                    pass  # Ignore if the suffix cannot be converted to int


    suffix += 1
    container_name = str("asteroid-" + str(suffix))
    print("new container name: "+container_name)


    for container in containers:            
        print("\t" + container['name'])
        if container_name in container['name']:
            print("EXISTIERTT BEREITS!")
            exists = True

    if not exists:
        # Create the container
        container_client = blob_service_client.create_container(container_name)

    print("Container created")

    print("Current working directory:", os.getcwd())
    
    local_file_name = "model.py"
    upload_file_path = os.path.join(os.getcwd(), 'model', local_file_name)


    # Create a blob client using the local file name as the name for the blob
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=local_file_name)
    print("\nUploading to Azure Storage as blob:\n\t" + local_file_name)

    # Upload the created file
    with open(file=upload_file_path, mode="rb") as data:
        blob_client.upload_blob(data)

    print ("Upload completed")

except Exception as ex:
    print('Exception:')
    print(ex)
    exit(1)
