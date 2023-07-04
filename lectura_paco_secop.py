import os
import requests
import pandas as pd

def download_and_save_csv(url, filename, folder='data'):
    response = requests.get(url)
    response.raise_for_status()

    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(os.path.join(folder, filename), 'wb') as f:
        f.write(response.content)

    print(f"Archivo '{filename}' guardado en la carpeta '{folder}'.")

if __name__ == "__main__":
    url = 'https://paco7public7info7prod.blob.core.windows.net/paco-pulic-info/SECOP_II_Cleaned.csv'
    filename = 'SECOP_II_Cleaned.csv'

    download_and_save_csv(url, filename)
