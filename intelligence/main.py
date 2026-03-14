import os
import sys
import grpc
import time
import logging
from concurrent import futures
from pathlib import Path
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] Brain: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add necessary paths for modular imports
# Add the 'gen_proto' directory to sys.path to resolve imports correctly
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'gen_proto'))

# Load environment
ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=ROOT_ENV)

# Delayed imports to ensure sys.path is updated
from service import IntelligenceService
from v1 import intelligence_pb2_grpc

def serve():
    port = os.getenv("INTELLIGENCE_PORT", "50051")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Add our modular service to the server
    intelligence_pb2_grpc.add_IntelligenceServiceServicer_to_server(IntelligenceService(), server)
    
    server.add_insecure_port(f'[::]:{port}')
    logger.info(f"--- Brain Server (Python) Starting ---")
    logger.info(f"Listening on port: {port}")
    
    server.start()
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        server.stop(0)

if __name__ == '__main__':
    serve()
