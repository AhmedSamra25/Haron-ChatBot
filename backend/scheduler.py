"""
Scheduler Service for Business bel Arabi RAG System

This module provides automatic scheduling for data ingestion and vector database updates.
Updates occur every 3 days to keep the knowledge base fresh with latest content.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional
import schedule
import time
from threading import Thread
import json
import os

from .data_ingester import EnhancedDataIngester
from .utils import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SchedulerService:
    """
    Handles automatic data updates for the RAG system
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.data_ingester = EnhancedDataIngester()
        self.is_running = False
        self.last_update = None
        self.update_interval_days = 3
        self.status_file = "scheduler_status.json"
        
    def load_status(self) -> dict:
        """Load scheduler status from file"""
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load scheduler status: {e}")
        
        return {
            "last_update": None,
            "total_updates": 0,
            "last_status": "never_run"
        }
    
    def save_status(self, status: str, error: Optional[str] = None):
        """Save scheduler status to file"""
        status_data = {
            "last_update": datetime.now().isoformat(),
            "last_status": status,
            "total_updates": self.load_status().get("total_updates", 0) + (1 if status == "success" else 0),
            "error": error
        }
        
        try:
            with open(self.status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save scheduler status: {e}")
    
    async def update_vector_database(self) -> bool:
        """
        Update the vector database with fresh content
        Returns True if successful, False otherwise
        """
        logger.info("Starting scheduled vector database update...")
        start_time = datetime.now()
        
        try:
            # Update podcast content
            logger.info("Ingesting podcast content...")
            podcast_result = await self.data_ingester.ingest_podcast_data()
            
            if not podcast_result.get("success", False):
                logger.warning(f"Podcast ingestion failed: {podcast_result.get('error', 'Unknown error')}")
            else:
                logger.info(f"Podcast ingestion completed: {podcast_result.get('message', 'Success')}")
            
            # Update WordPress content
            logger.info("Ingesting WordPress content...")
            wp_result = await self.data_ingester.ingest_wordpress_data()
            
            if not wp_result.get("success", False):
                logger.warning(f"WordPress ingestion failed: {wp_result.get('error', 'Unknown error')}")
            else:
                logger.info(f"WordPress ingestion completed: {wp_result.get('message', 'Success')}")
            
            # Check if at least one source succeeded
            if podcast_result.get("success", False) or wp_result.get("success", False):
                duration = datetime.now() - start_time
                logger.info(f"Vector database update completed successfully in {duration.total_seconds():.2f} seconds")
                self.save_status("success")
                return True
            else:
                logger.error("Both podcast and WordPress ingestion failed")
                self.save_status("failed", "Both data sources failed")
                return False
                
        except Exception as e:
            duration = datetime.now() - start_time
            logger.error(f"Vector database update failed after {duration.total_seconds():.2f} seconds: {str(e)}")
            self.save_status("error", str(e))
            return False
    
    def should_update(self) -> bool:
        """Check if an update is needed based on last update time"""
        status = self.load_status()
        if not status.get("last_update"):
            logger.info("No previous update found, scheduling immediate update")
            return True
        
        try:
            last_update = datetime.fromisoformat(status["last_update"])
            time_since_update = datetime.now() - last_update
            
            if time_since_update >= timedelta(days=self.update_interval_days):
                logger.info(f"Update needed: {time_since_update.days} days since last update")
                return True
            else:
                days_remaining = self.update_interval_days - time_since_update.days
                logger.info(f"No update needed: {days_remaining} days until next update")
                return False
                
        except Exception as e:
            logger.warning(f"Could not parse last update time: {e}, scheduling update")
            return True
    
    async def run_scheduled_update(self):
        """Run the scheduled update task"""
        if not self.should_update():
            return
        
        logger.info("Running scheduled vector database update...")
        success = await self.update_vector_database()
        
        if success:
            logger.info("Scheduled update completed successfully")
        else:
            logger.error("Scheduled update failed")
    
    def schedule_jobs(self):
        """Set up the scheduled jobs"""
        # Schedule daily check at 2 AM
        schedule.every().day.at("02:00").do(
            lambda: asyncio.create_task(self.run_scheduled_update())
        )
        
        # Schedule immediate check every hour for missed updates
        schedule.every().hour.do(
            lambda: asyncio.create_task(self.run_scheduled_update()) if self.should_update() else None
        )
        
        logger.info(f"Scheduler configured to update every {self.update_interval_days} days")
        logger.info("Daily check scheduled at 2:00 AM")
    
    def run_scheduler(self):
        """Run the scheduler in a separate thread"""
        def scheduler_thread():
            self.schedule_jobs()
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        if not self.is_running:
            self.is_running = True
            thread = Thread(target=scheduler_thread, daemon=True)
            thread.start()
            logger.info("Scheduler service started")
        else:
            logger.warning("Scheduler is already running")
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        self.is_running = False
        schedule.clear()
        logger.info("Scheduler service stopped")
    
    def get_status(self) -> dict:
        """Get current scheduler status"""
        status = self.load_status()
        
        next_update = None
        if status.get("last_update"):
            try:
                last_update = datetime.fromisoformat(status["last_update"])
                next_update = (last_update + timedelta(days=self.update_interval_days)).isoformat()
            except:
                pass
        
        return {
            "is_running": self.is_running,
            "update_interval_days": self.update_interval_days,
            "last_update": status.get("last_update"),
            "next_scheduled_update": next_update,
            "total_updates": status.get("total_updates", 0),
            "last_status": status.get("last_status", "never_run"),
            "last_error": status.get("error"),
            "should_update_now": self.should_update()
        }
    
    async def force_update(self) -> dict:
        """Force an immediate update regardless of schedule"""
        logger.info("Forcing immediate vector database update...")
        success = await self.update_vector_database()
        
        return {
            "success": success,
            "message": "Force update completed successfully" if success else "Force update failed",
            "timestamp": datetime.now().isoformat()
        }

# Global scheduler instance
scheduler_service = SchedulerService()

def get_scheduler() -> SchedulerService:
    """Get the global scheduler instance"""
    return scheduler_service
