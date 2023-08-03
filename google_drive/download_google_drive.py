from __future__ import print_function
from fileinput import filename

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import *

SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']

def get_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            print(flow)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    service = build('drive', 'v3', credentials=creds)
    return service

def get_ID_by_name(service, file_name):
    page_token = None
    while True:
        response = service.files().list(q="name contains '{}'".format(file_name),
                                            spaces='drive',
                                            fields='nextPageToken, files(id, name)',
                                            pageToken=page_token).execute()
        for file in response.get('files', []):
            return file.get('id')

def download_from_google_drive(service, file_id, file_name):
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(file_name, 'wb') 
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print ("Download %d%%." % int(status.progress() * 100))
    except HttpError as error:
        print(f'An error occurred: {error}')

service = get_service()
download_from_google_drive(service, get_ID_by_name(service,'res.csv'), 'res.csv')