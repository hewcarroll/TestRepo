"""
Audit Logger

Comprehensive audit trail and compliance logging system for regulatory compliance.
"""

import asyncio
import structlog
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from decimal import Decimal
from threading import Event, Lock
from typing import Dict, List, Optional, Any
from pathlib import Path
import gzip
import shutil


@dataclass
class AuditRecord:
    """Represents a single audit trail record."""
    timestamp: datetime
    event_type: str
    category: str  # 'trade', 'compliance', 'violation', 'system'
    severity: str  # 'info', 'warning', 'error', 'critical'
    data: Dict[str, Any]
    source: str  # Component that generated the record
    record_id: str = field(default_factory=lambda: str(datetime.now(timezone.utc).timestamp()))


class AuditLogger:
    """
    Comprehensive audit logging system for regulatory compliance.
    
    Provides immutable audit trails for:
    - All trading activities
    - Compliance decisions and violations
    - System events and errors
    - Regulatory reporting data
    """
    
    def __init__(self, shutdown_event: Optional[Event] = None):
        self.shutdown_event = shutdown_event or Event()
        self.logger = structlog.get_logger("audit_logger")
        
        # Audit configuration
        self._audit_dir = Path("audit_logs")
        self._max_file_size = 100 * 1024 * 1024  # 100MB per file
        self._retention_days = 2555  # 7 years for regulatory compliance
        self._compress_threshold = 24 * 60 * 60  # Compress after 24 hours
        
        # In-memory audit buffer
        self._audit_buffer: List[AuditRecord] = []
        self._buffer_lock = Lock()
        self._current_file: Optional[str] = None
        self._current_file_size = 0
        
        # Statistics
        self._records_written = 0
        self._violations_logged = 0
        self._trades_logged = 0
        
        self.logger.info("Audit logger initialized",
                        audit_dir=str(self._audit_dir),
                        retention_days=self._retention_days)
    
    async def initialize(self):
        """Initialize audit logger and create necessary directories."""
        try:
            # Create audit directory structure
            self._audit_dir.mkdir(exist_ok=True)
            (self._audit_dir / "daily").mkdir(exist_ok=True)
            (self._audit_dir / "violations").mkdir(exist_ok=True)
            (self._audit_dir / "trades").mkdir(exist_ok=True)
            (self._audit_dir / "compliance").mkdir(exist_ok=True)
            
            # Start background tasks
            asyncio.create_task(self._flush_buffer_loop())
            asyncio.create_task(self._cleanup_old_logs())
            
            self.logger.info("Audit logger initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize audit logger", error=str(e))
            raise
    
    async def shutdown(self):
        """Shutdown audit logger and flush all pending records."""
        try:
            self.logger.info("Shutting down audit logger")
            
            # Flush any remaining records
            await self._flush_buffer()
            
            # Close current file
            if self._current_file:
                await self._close_current_file()
            
            self.logger.info("Audit logger shutdown complete",
                           total_records=self._records_written,
                           violations_logged=self._violations_logged,
                           trades_logged=self._trades_logged)
        except Exception as e:
            self.logger.error("Error during audit logger shutdown", error=str(e))
    
    async def log_trade_activity(self, trade_data: Dict[str, Any]):
        """Log trading activity for audit trail."""
        
        record = AuditRecord(
            timestamp=datetime.now(timezone.utc),
            event_type='trade_execution',
            category='trade',
            severity='info',
            data=trade_data,
            source='trading_engine'
        )
        
        await self._queue_record(record)
        self._trades_logged += 1
        
        self.logger.debug("Trade activity logged",
                         symbol=trade_data.get('symbol'),
                         action=trade_data.get('action'))
    
    async def log_compliance_decision(self, decision_data: Dict[str, Any]):
        """Log compliance decision for audit trail."""
        
        # Determine severity based on decision
        severity = 'info'
        if not decision_data.get('approved', True):
            severity = 'warning'
        
        record = AuditRecord(
            timestamp=datetime.now(timezone.utc),
            event_type='compliance_decision',
            category='compliance',
            severity=severity,
            data=decision_data,
            source='compliance_engine'
        )
        
        await self._queue_record(record)
        
        self.logger.debug("Compliance decision logged",
                         approved=decision_data.get('approved'),
                         symbol=decision_data.get('symbol'))
    
    async def log_compliance_violation(self, violation_data: Dict[str, Any]):
        """Log compliance violation for audit trail."""
        
        record = AuditRecord(
            timestamp=datetime.now(timezone.utc),
            event_type='compliance_violation',
            category='violation',
            severity='error',
            data=violation_data,
            source='compliance_engine'
        )
        
        await self._queue_record(record)
        self._violations_logged += 1
        
        self.logger.warning("Compliance violation logged",
                           violation_type=violation_data.get('type'),
                           symbol=violation_data.get('symbol'))
    
    async def log_system_event(self, event_type: str, event_data: Dict[str, Any], 
                              severity: str = 'info'):
        """Log system events for audit trail."""
        
        record = AuditRecord(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            category='system',
            severity=severity,
            data=event_data,
            source='system'
        )
        
        await self._queue_record(record)
        
        self.logger.debug("System event logged",
                         event_type=event_type,
                         severity=severity)
    
    async def _queue_record(self, record: AuditRecord):
        """Queue audit record for writing."""
        with self._buffer_lock:
            self._audit_buffer.append(record)
            
            # Auto-flush if buffer gets too large
            if len(self._audit_buffer) >= 1000:
                asyncio.create_task(self._flush_buffer())
    
    async def _flush_buffer_loop(self):
        """Background task to periodically flush audit buffer."""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(30)  # Flush every 30 seconds
                
                if self._audit_buffer:
                    await self._flush_buffer()
                    
            except Exception as e:
                self.logger.error("Error in audit buffer flush loop", error=str(e))
    
    async def _flush_buffer(self):
        """Flush audit buffer to disk."""
        if not self._audit_buffer:
            return
        
        with self._buffer_lock:
            records_to_write = self._audit_buffer.copy()
            self._audit_buffer.clear()
        
        if not records_to_write:
            return
        
        try:
            # Group records by category for separate files
            records_by_category = {
                'trade': [],
                'compliance': [],
                'violation': [],
                'system': []
            }
            
            for record in records_to_write:
                if record.category in records_by_category:
                    records_by_category[record.category].append(record)
            
            # Write to appropriate files
            for category, records in records_by_category.items():
                if records:
                    await self._write_category_records(category, records)
            
            self._records_written += len(records_to_write)
            
        except Exception as e:
            self.logger.error("Error flushing audit buffer", error=str(e))
            
            # Put records back in buffer for retry
            with self._buffer_lock:
                self._audit_buffer.extend(records_to_write)
    
    async def _write_category_records(self, category: str, records: List[AuditRecord]):
        """Write records to category-specific file."""
        
        if not records:
            return
        
        # Get current date for file naming
        today = date.today()
        
        # Determine file path based on category
        if category == 'violation':
            file_path = self._audit_dir / "violations" / f"violations_{today.isoformat()}.jsonl"
        elif category == 'trade':
            file_path = self._audit_dir / "trades" / f"trades_{today.isoformat()}.jsonl"
        elif category == 'compliance':
            file_path = self._audit_dir / "compliance" / f"compliance_{today.isoformat()}.jsonl"
        else:
            file_path = self._audit_dir / "daily" / f"system_{today.isoformat()}.jsonl"
        
        # Check if we need a new file
        if await self._should_create_new_file(file_path):
            await self._close_current_file()
            self._current_file = str(file_path)
            self._current_file_size = 0
        
        # Write records
        with open(file_path, 'a', encoding='utf-8') as f:
            for record in records:
                json_line = json.dumps({
                    'timestamp': record.timestamp.isoformat(),
                    'event_type': record.event_type,
                    'category': record.category,
                    'severity': record.severity,
                    'data': record.data,
                    'source': record.source,
                    'record_id': record.record_id
                }, default=str) + '\n'
                
                f.write(json_line)
                self._current_file_size += len(json_line.encode('utf-8'))
        
        # Compress old files if needed
        await self._compress_old_files()
    
    async def _should_create_new_file(self, file_path: Path) -> bool:
        """Check if we should create a new file."""
        if not file_path.exists():
            return True
        
        # Check file size
        if self._current_file != str(file_path):
            return True
            
        return file_path.stat().st_size >= self._max_file_size
    
    async def _close_current_file(self):
        """Close current audit file."""
        # File will be automatically closed when we stop writing to it
        self._current_file = None
        self._current_file_size = 0
    
    async def _compress_old_files(self):
        """Compress audit files older than threshold."""
        try:
            cutoff_time = datetime.now(timezone.utc).timestamp() - self._compress_threshold
            
            for log_dir in ['daily', 'violations', 'trades', 'compliance']:
                dir_path = self._audit_dir / log_dir
                
                for file_path in dir_path.glob("*.jsonl"):
                    if file_path.stat().st_mtime < cutoff_time:
                        # Compress the file
                        compressed_path = file_path.with_suffix('.jsonl.gz')
                        
                        if not compressed_path.exists():
                            with open(file_path, 'rb') as f_in:
                                with gzip.open(compressed_path, 'wb') as f_out:
                                    shutil.copyfileobj(f_in, f_out)
                            
                            # Remove original file after successful compression
                            file_path.unlink()
                            
        except Exception as e:
            self.logger.error("Error compressing audit files", error=str(e))
    
    async def _cleanup_old_logs(self):
        """Background task to clean up old audit logs."""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(24 * 60 * 60)  # Run daily
                
                await self._cleanup_expired_logs()
                    
            except Exception as e:
                self.logger.error("Error in audit cleanup task", error=str(e))
    
    async def _cleanup_expired_logs(self):
        """Remove audit logs older than retention period."""
        try:
            cutoff_date = date.today() - timedelta(days=self._retention_days)
            
            for log_dir in ['daily', 'violations', 'trades', 'compliance']:
                dir_path = self._audit_dir / log_dir
                
                for file_path in dir_path.glob("*.jsonl*"):
                    # Extract date from filename
                    try:
                        date_str = file_path.stem.split('_')[-1]
                        file_date = datetime.fromisoformat(date_str).date()
                        
                        if file_date < cutoff_date:
                            file_path.unlink()
                            self.logger.info("Removed expired audit file", file=str(file_path))
                            
                    except (ValueError, IndexError):
                        # Skip files that don't match expected naming pattern
                        continue
                        
        except Exception as e:
            self.logger.error("Error cleaning up expired logs", error=str(e))
    
    def get_audit_summary(self) -> Dict[str, Any]:
        """Get audit logger summary and statistics."""
        
        return {
            'total_records': self._records_written,
            'violations_logged': self._violations_logged,
            'trades_logged': self._trades_logged,
            'buffer_size': len(self._audit_buffer),
            'current_file': self._current_file,
            'current_file_size': self._current_file_size,
            'retention_days': self._retention_days,
            'audit_directory': str(self._audit_dir)
        }
    
    def export_audit_data(self, start_date: date, end_date: date, 
                          categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Export audit data for specified date range.
        
        Args:
            start_date: Start date for export
            end_date: End date for export
            categories: Categories to export (None for all)
            
        Returns:
            List of audit records
        """
        exported_records = []
        
        try:
            # Determine which directories to search
            search_dirs = []
            if not categories or 'trade' in categories:
                search_dirs.append('trades')
            if not categories or 'violation' in categories:
                search_dirs.append('violations')
            if not categories or 'compliance' in categories:
                search_dirs.append('compliance')
            if not categories or 'system' in categories:
                search_dirs.append('daily')
            
            for log_dir in search_dirs:
                dir_path = self._audit_dir / log_dir
                
                for file_path in dir_path.glob("*.jsonl*"):
                    # Check if file is in date range
                    try:
                        date_str = file_path.stem.split('_')[-1]
                        file_date = datetime.fromisoformat(date_str).date()
                        
                        if start_date <= file_date <= end_date:
                            # Read and parse file
                            if file_path.suffix == '.gz':
                                import gzip
                                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                                    for line in f:
                                        try:
                                            record = json.loads(line)
                                            exported_records.append(record)
                                        except json.JSONDecodeError:
                                            continue
                            else:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    for line in f:
                                        try:
                                            record = json.loads(line)
                                            exported_records.append(record)
                                        except json.JSONDecodeError:
                                            continue
                                            
                    except (ValueError, IndexError):
                        continue
            
            # Sort by timestamp
            exported_records.sort(key=lambda x: x.get('timestamp', ''))
            
            return exported_records
            
        except Exception as e:
            self.logger.error("Error exporting audit data", error=str(e))
            return []
