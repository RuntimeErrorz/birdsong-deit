from __future__ import print_function
from fileinput import filename

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']


def upload_google_drive(file_name, file_path, folder_name):
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
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    try:
        service = build('drive', 'v3', credentials=creds)
        folder_id = None
        query = "mimeType='application/vnd.google-apps.folder' and trashed=false and name='" + folder_name + "'"
        results = service.files().list(pageSize=1, q=query,
                                       fields="files(id, name)").execute()
        folders = results.get('files', [])
        if folders:
            folder_id = folders[0]['id']
        file_metadata = {'name': file_name, 'parents': [folder_id]}
        media = MediaFileUpload(file_path, mimetype='text/plain')
        file = service.files().create(body=file_metadata,
                                      media_body=media,
                                      fields='id').execute()
        print('File ID:{}'.format(file.get('id')))
    except HttpError as error:
        print(f'An error occurred: {error}')

upload_google_drive('code.py','files/predict.py','birdclef-2021')
