# drive_access.py
import io
from django.conf import settings
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

class GoogleDriveAccess:
    def __init__(self):
        self.service = self._authenticate()
        self.models_folder_id = self._get_folder_id(settings.GOOGLE_DRIVE_MODELS_FOLDER)
        # Get the base folder ID for datasets
        self.dataset_base_id = self._get_folder_id('skin-ds', parent_id=self.models_folder_id)
    
    def _authenticate(self):
        """Authenticate with Google Drive using service account credentials"""
        creds = service_account.Credentials.from_service_account_file(
            settings.GOOGLE_DRIVE_CREDENTIALS_FILE,
            scopes=['https://www.googleapis.com/auth/drive']
        )
        return build('drive', 'v3', credentials=creds)
    
    def _get_folder_id(self, folder_name, parent_id=None):
        """Get folder ID by name, optionally under a parent folder"""
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        if parent_id:
            query += f" and '{parent_id}' in parents"
            
        response = self.service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name)'
        ).execute()
        
        if not response.get('files', []):
            raise ValueError(f"Folder '{folder_name}' not found in Google Drive")
        
        return response['files'][0]['id']
    
    def get_dataset_files(self, category):
        """Get all image files for a specific category"""
        try:
            # Get category folder ID
            category_id = self._get_folder_id(category, parent_id=self._get_folder_id('train', parent_id=self.dataset_base_id))
            
            # Get all files in category folder
            results = self.service.files().list(
                q=f"'{category_id}' in parents and mimeType!='application/vnd.google-apps.folder'",
                pageSize=1000,
                fields="files(id, name)"
            ).execute()
            
            return results.get('files', [])
        except Exception as e:
            print(f"Error accessing category {category}: {e}")
            return []
    
    def get_file_stream_by_id(self, file_id):
        """Get file content by ID"""
        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
        
        fh.seek(0)
        return fh
    
    def upload_file(self, file_name, file_content):
        """Upload file to Google Drive"""
        file_metadata = {
            'name': file_name,
            'parents': [self.models_folder_id]
        }
        
        media = MediaIoBaseUpload(
            io.BytesIO(file_content),
            mimetype='application/octet-stream'
        )
        
        self.service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
    
    def list_files(self):
        """List all files in the configured folder"""
        results = self.service.files().list(
            q=f"'{self.models_folder_id}' in parents",
            pageSize=100,
            fields="files(id, name, mimeType)"
        ).execute()
        return results.get('files', [])