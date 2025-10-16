"""Tests for scheduler functionality."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timedelta

from app.core.scheduler import SchedulerService


class TestSchedulerService:
    """Tests for scheduler service."""
    
    @pytest.fixture
    def scheduler_service(self):
        with patch('app.core.scheduler.redis.from_url'):
            return SchedulerService()
    
    @pytest.mark.asyncio
    async def test_initialize(self, scheduler_service):
        """Test scheduler initialization."""
        with patch('app.core.scheduler.redis.from_url') as mock_redis, \
             patch('app.core.scheduler.Queue') as mock_queue, \
             patch('app.core.scheduler.AsyncIOScheduler') as mock_scheduler_class:
            
            # Setup mocks
            mock_redis_client = AsyncMock()
            mock_redis_client.ping.return_value = True
            mock_redis.return_value = mock_redis_client
            
            mock_scheduler = Mock()
            mock_scheduler_class.return_value = mock_scheduler
            
            # Initialize
            await scheduler_service.initialize()
            
            # Verify initialization
            assert scheduler_service.redis_client is not None
            assert scheduler_service.queue is not None
            assert scheduler_service.scheduler is not None
            mock_scheduler.add_job.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_scheduler(self, scheduler_service):
        """Test starting the scheduler."""
        mock_scheduler = Mock()
        mock_scheduler.running = False
        scheduler_service.scheduler = mock_scheduler
        
        with patch.object(scheduler_service, 'initialize') as mock_init:
            await scheduler_service.start()
            
            mock_init.assert_called_once()
            mock_scheduler.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_scheduler(self, scheduler_service):
        """Test stopping the scheduler."""
        mock_scheduler = Mock()
        mock_scheduler.running = True
        scheduler_service.scheduler = mock_scheduler
        
        mock_redis_client = AsyncMock()
        scheduler_service.redis_client = mock_redis_client
        
        await scheduler_service.stop()
        
        mock_scheduler.shutdown.assert_called_once()
        mock_redis_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_trigger_manual_ingestion(self, scheduler_service):
        """Test triggering manual ingestion."""
        mock_job = Mock()
        mock_job.id = "manual_job_123"
        
        mock_queue = Mock()
        mock_queue.enqueue.return_value = mock_job
        scheduler_service.queue = mock_queue
        
        job_id = await scheduler_service.trigger_manual_ingestion()
        
        assert job_id == "manual_job_123"
        mock_queue.enqueue.assert_called_once()
    
    def test_get_next_scheduled_run(self, scheduler_service):
        """Test getting next scheduled run time."""
        # Test with scheduler and job
        mock_job = Mock()
        mock_job.next_run_time = datetime.now() + timedelta(days=1)
        
        mock_scheduler = Mock()
        mock_scheduler.get_job.return_value = mock_job
        scheduler_service.scheduler = mock_scheduler
        
        next_run = scheduler_service.get_next_scheduled_run()
        
        assert next_run == mock_job.next_run_time
        mock_scheduler.get_job.assert_called_with("recurring_ingestion")
    
    def test_get_next_scheduled_run_no_scheduler(self, scheduler_service):
        """Test getting next run time with no scheduler."""
        scheduler_service.scheduler = None
        
        next_run = scheduler_service.get_next_scheduled_run()
        
        assert next_run is None
    
    @pytest.mark.asyncio
    async def test_enqueue_ingestion_job(self, scheduler_service):
        """Test enqueuing ingestion job."""
        mock_job = Mock()
        mock_job.id = "ingestion_123"
        
        mock_queue = Mock()
        mock_queue.enqueue.return_value = mock_job
        scheduler_service.queue = mock_queue
        
        # Test the private method
        await scheduler_service._enqueue_ingestion_job()
        
        mock_queue.enqueue.assert_called_once()
        # Verify the job was enqueued with correct parameters
        call_args = mock_queue.enqueue.call_args
        assert call_args[1]['job_timeout'] == "2h"
    
    @pytest.mark.asyncio
    async def test_enqueue_ingestion_job_no_queue(self, scheduler_service):
        """Test enqueuing when queue is not initialized."""
        scheduler_service.queue = None
        
        # Should not raise an exception, just log an error
        await scheduler_service._enqueue_ingestion_job()
    
    @pytest.mark.asyncio 
    async def test_scheduler_job_scheduling(self, scheduler_service):
        """Test that jobs are scheduled with correct intervals."""
        with patch('app.core.scheduler.redis.from_url') as mock_redis, \
             patch('app.core.scheduler.Queue') as mock_queue, \
             patch('app.core.scheduler.AsyncIOScheduler') as mock_scheduler_class, \
             patch('app.core.scheduler.settings') as mock_settings:
            
            # Setup settings
            mock_settings.ingest_interval_days = 5  # Custom interval
            mock_settings.redis_url = "redis://test"
            
            # Setup mocks
            mock_redis_client = AsyncMock()
            mock_redis_client.ping.return_value = True
            mock_redis.return_value = mock_redis_client
            
            mock_scheduler = Mock()
            mock_scheduler_class.return_value = mock_scheduler
            
            # Initialize
            await scheduler_service.initialize()
            
            # Verify job was scheduled with correct interval
            mock_scheduler.add_job.assert_called_once()
            call_args = mock_scheduler.add_job.call_args
            
            # Check that IntervalTrigger was used with correct days
            trigger = call_args[1]['trigger']
            assert hasattr(trigger, 'interval')
    
    @pytest.mark.asyncio
    async def test_scheduler_error_handling(self, scheduler_service):
        """Test scheduler error handling."""
        with patch('app.core.scheduler.redis.from_url') as mock_redis:
            # Simulate Redis connection failure
            mock_redis.side_effect = Exception("Redis connection failed")
            
            with pytest.raises(Exception):
                await scheduler_service.initialize()
