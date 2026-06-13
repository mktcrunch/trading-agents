"""
Base agent class for all agents in the system
Provides common functionality for signal generation, execution, and monitoring
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
from src import config
from src.logger import setup_logger

logger = setup_logger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents
    Each agent handles a specific responsibility in the trading system
    """

    def __init__(self, system: str = "baseline"):
        """
        Initialize base agent

        Args:
            system: "baseline" or "internal"
        """
        self.system = system
        self.logger = logger
        self.created_at = datetime.now()

    @abstractmethod
    async def execute(self) -> bool:
        """
        Main execution method for the agent
        Implemented by subclasses
        """
        pass

    def log_action(
        self,
        action: str,
        data: Dict[str, Any] = None,
        event_type: str = "agent_action",
        status: str = "ok",
    ):
        """Log agent action and record to audit trail."""
        self.logger.info(f"[{self.system.upper()}] {action}")
        if data:
            self.logger.debug(f"  Data: {data}")
        if config.AUDIT_ENABLED:
            from src.audit import record_event
            record_event(
                event_type=event_type,
                action=action,
                system=self.system,
                agent=self.__class__.__name__,
                status=status,
                payload=data or {},
            )

    def log_risk_rejection(self, reason: str, data: Dict[str, Any] = None):
        """Record an expected risk-limit rejection (not a system failure)."""
        self.logger.info(f"[{self.system.upper()}] {reason}")
        if config.AUDIT_ENABLED:
            from src.audit import record_event
            record_event(
                event_type="risk_rejected",
                action=reason,
                system=self.system,
                agent=self.__class__.__name__,
                status="ok",
                payload=data or {},
            )

    def log_error(self, error: str, exception: Exception = None):
        """Log agent error"""
        self.logger.error(f"[{self.system.upper()}] {error}")
        if exception:
            self.logger.error(f"  Exception: {exception}")
        if config.AUDIT_ENABLED:
            from src.audit import record_event
            record_event(
                event_type="error",
                action=error,
                system=self.system,
                agent=self.__class__.__name__,
                status="error",
                payload={"exception": str(exception) if exception else None},
            )
