import os
import requests
import zipfile

def download_and_unzip(url='https://microdata.worldbank.org/index.php/catalog/3016/download/42079', extract_to="/app/data/raw"):
  os.makedirs(extract_to, exist_ok=True)
  local_zip_path = os.path.join(extract_to, "temp.zip")

  # Download the file
  with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(local_zip_path, "wb") as f:
      for chunk in r.iter_content(chunk_size=8192):
        f.write(chunk)

  # Unzip the file
  with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

  # Remove the zip file
  os.remove(local_zip_path)

# Example usage:
# download_and_unzip("https://example.com/data.zip")