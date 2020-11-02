from google_drive_downloader import GoogleDriveDownloader as gdd
from sys import platform

'''----------------------'''
'''Check for Environments'''
'''----------------------'''

# Download DeepCrawl environment if it does not exist
if platform == 'linux' or platform == 'linux2':
    gdd.download_file_from_google_drive(file_id='19BlS0W7nwhrBS4E25BMdf96ICDsp7jts',
                                        dest_path='envs/DeepCrawl-Transformer.zip', showsize=True,
                                        unzip=True)

    gdd.download_file_from_google_drive(file_id='13NLxzd7VFQPx0sC5L4nZFhlRLBGkTI2i',
                                        dest_path='envs/DeepCrawl-Dense-Embedding.zip', showsize=True,
                                        unzip=True)