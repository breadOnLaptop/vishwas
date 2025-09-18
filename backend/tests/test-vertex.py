from vertexai import init
from vertexai.generative_models import GenerativeModel
from app.core.config import settings
from google.oauth2 import service_account

# Load credentials explicitly
creds = service_account.Credentials.from_service_account_file(
    settings.GOOGLE_APPLICATION_CREDENTIALS
)

# Init Vertex AI
init(project=settings.GCP_PROJECT, location=settings.GCP_REGION, credentials=creds)

# Use Gemini model
model = GenerativeModel("gemini-2.5-pro")

response = model.generate_content("Say hello in one sentence.")
print(response.text)
