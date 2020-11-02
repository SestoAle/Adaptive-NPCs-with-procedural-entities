from google_drive_downloader import GoogleDriveDownloader as gdd
from sys import platform

'''----------------------'''
'''Check for Environments'''
'''----------------------'''

# Download DeepCrawl environment if it does not exist
if platform == 'linux' or platform == 'linux2':
    gdd.download_file_from_google_drive(file_id='1qT7JUF-72V-ov0yttJghQ87VGLBhIyDx',
                                        dest_path='envs/envs.zip', showsize=True,
                                        unzip=True)
else:
    gdd.download_file_from_google_drive(file_id='1-0fraEld9CSRV7vacAnddqpVdUkTWXvn',
                                        dest_path='envs/envs.zip', showsize=True,
                                        unzip=True)