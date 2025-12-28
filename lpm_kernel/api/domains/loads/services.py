from lpm_kernel.api.common.script_executor import logger
from lpm_kernel.models.load import Load
from lpm_kernel.common.repository.database_session import DatabaseSession
from typing import Optional

class LoadService:
    @staticmethod
    def get_current_upload_name() -> Optional[str]:
        """
        Get the current upload name
        
        Returns:
            str: Returns the upload name if found, otherwise None
        """
        try:
            with DatabaseSession.session() as session:
                # Get the latest record
                latest_load = session.query(Load).order_by(Load.created_at.desc()).first()
                return latest_load.name if latest_load else None
        except Exception as e:
            logger.error(f"Error getting current upload name: {str(e)}", exc_info=True)
            return None
    
    # Get the current upload instance description
    @staticmethod
    def get_current_upload_description() -> Optional[str]:
        """
        Get the current upload description
        
        Returns:
            str: Returns the upload description if found, otherwise None
        """
        try:
            with DatabaseSession.session() as session:
                # Get the latest record
                latest_load = session.query(Load).order_by(Load.created_at.desc()).first()
                return latest_load.description if latest_load else None
        except Exception as e:
            print(f"Error getting current upload description: {str(e)}")
            return None
