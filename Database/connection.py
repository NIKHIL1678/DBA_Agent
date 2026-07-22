import os
import logging
from typing import Optional
from urllib.parse import quote_plus
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from dotenv import load_dotenv



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DatabaseManager:
    """
    Strictly manages the MySQL connection pool. 
    Execution and schema extraction are handled elsewhere by agent tools.
    """
    _instance: Optional['DatabaseManager'] = None
    engine: Optional[Engine] = None

    def __new__(cls) -> 'DatabaseManager':
        """Singleton pattern to ensure only one connection pool exists."""
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initializes the SQLAlchemy engine."""
        db_user = os.getenv("DB_USER", "root")
        db_password = os.getenv("DB_PASSWORD", "")
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "3306")
        db_name = os.getenv("DB_NAME", "dba_agent_test")

        if not db_password:
            logger.warning("DB_PASSWORD is empty. Check that .env is present and loaded correctly.")

        # URL-encode user/password so special characters (@, :, /, etc.)
        # in credentials don't get misparsed as URL separators.
        # e.g. password "Nikhil@2005" would otherwise break the host portion
        # of the connection string.
        safe_user = quote_plus(db_user)
        safe_password = quote_plus(db_password)

        uri = f"mysql+pymysql://{safe_user}:{safe_password}@{db_host}:{db_port}/{db_name}"

        try:
            # pool_recycle prevents "MySQL server has gone away" timeouts
            self.engine = create_engine(uri, pool_recycle=3600, echo=False)
            logger.info(f"Database engine connected to {db_host}:{db_port}/{db_name}")
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise

def get_db_engine() -> Engine:
    """Helper function to quickly grab the initialized engine."""
    engine = DatabaseManager().engine
    if engine is None:
        raise RuntimeError("Database engine is not initialized. Check your connection settings.")
    return engine