"""
Anti-Scam Sentinel Python SDK
A simple client library for interacting with the Anti-Scam Sentinel API.

Usage:
    from anti_scam_sdk import AntiScamClient
    
    # Initialize client
    client = AntiScamClient(
        base_url="http://localhost:8000",
        api_key="your-api-key"  # Optional if server doesn't require auth
    )
    
    # Send a message
    response = client.send_message(
        session_id="my-session-123",
        message="Your SBI account will be blocked!"
    )
    
    # Check if scam detected
    if response.is_scam:
        print(f"Scam detected! Type: {response.forensics.scam_type}")
        print(f"UPI IDs found: {response.extracted_entities.upi_ids}")
"""

import httpx
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio


@dataclass
class ValidatedUPI:
    """Validated UPI ID with bank provider info"""
    upi_id: str
    bank_provider: Optional[str] = None
    provider_type: Optional[str] = None
    verified: bool = False
    confidence: float = 0.0


@dataclass
class BankAccount:
    """Extracted bank account details"""
    account_number: str
    ifsc_code: Optional[str] = None
    bank_name: Optional[str] = None


@dataclass
class ExtractedEntities:
    """Intelligence extracted from scam messages"""
    upi_ids: List[ValidatedUPI]
    bank_accounts: List[BankAccount]
    phone_numbers: List[str]
    urls: List[str]
    emails: List[str]
    amounts: List[str]
    intel_completeness_score: float = 0.0


@dataclass
class Forensics:
    """Forensics data from scam detection"""
    scam_type: str
    threat_level: str
    detected_indicators: List[str]
    persona_used: Optional[str] = None
    scammer_frustration: str = "none"
    intel_quality: str = "low"


@dataclass 
class ResponseMetadata:
    """Metadata about the response"""
    phase: str
    turn_count: int
    latency_ms: int
    persona: Optional[str] = None
    llm_used: str = "template"
    processing_async: bool = False


@dataclass
class AgentResponse:
    """Complete response from the Anti-Scam Agent"""
    session_id: str
    is_scam: bool
    confidence_score: float
    agent_response: str
    extracted_entities: ExtractedEntities
    forensics: Forensics
    metadata: ResponseMetadata


@dataclass
class HealthStatus:
    """API health status"""
    status: str
    version: str
    timestamp: str
    api_key_required: bool
    components: Dict[str, str]


class AntiScamClient:
    """
    Python client for the Anti-Scam Sentinel API.
    
    Args:
        base_url: Base URL of the API (default: http://localhost:8000)
        api_key: Optional API key for authentication (X-API-Key header)
        timeout: Request timeout in seconds (default: 30)
    
    Example:
        >>> client = AntiScamClient("http://localhost:8000", api_key="your-key")
        >>> health = client.health_check()
        >>> print(health.status)
        'healthy'
    """
    
    def __init__(
        self, 
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 30.0
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self._async_client = None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        return headers
    
    def _parse_response(self, data: Dict) -> AgentResponse:
        """Parse API response into AgentResponse dataclass"""
        entities_data = data.get("extracted_entities", {})
        
        # Parse UPI IDs
        upi_ids = []
        for upi in entities_data.get("upi_ids", []):
            if isinstance(upi, dict):
                upi_ids.append(ValidatedUPI(**upi))
            else:
                upi_ids.append(ValidatedUPI(upi_id=str(upi)))
        
        # Parse bank accounts
        bank_accounts = []
        for acc in entities_data.get("bank_accounts", []):
            if isinstance(acc, dict):
                bank_accounts.append(BankAccount(
                    account_number=acc.get("account_number", ""),
                    ifsc_code=acc.get("ifsc_code"),
                    bank_name=acc.get("bank_name")
                ))
        
        extracted_entities = ExtractedEntities(
            upi_ids=upi_ids,
            bank_accounts=bank_accounts,
            phone_numbers=entities_data.get("phone_numbers", []),
            urls=entities_data.get("urls", []),
            emails=entities_data.get("emails", []),
            amounts=entities_data.get("amounts", []),
            intel_completeness_score=entities_data.get("intel_completeness_score", 0.0)
        )
        
        forensics_data = data.get("forensics", {})
        forensics = Forensics(
            scam_type=forensics_data.get("scam_type", "unknown"),
            threat_level=forensics_data.get("threat_level", "low"),
            detected_indicators=forensics_data.get("detected_indicators", []),
            persona_used=forensics_data.get("persona_used"),
            scammer_frustration=forensics_data.get("scammer_frustration", "none"),
            intel_quality=forensics_data.get("intel_quality", "low")
        )
        
        metadata_data = data.get("metadata", {})
        metadata = ResponseMetadata(
            phase=metadata_data.get("phase", "unknown"),
            turn_count=metadata_data.get("turn_count", 0),
            latency_ms=metadata_data.get("latency_ms", 0),
            persona=metadata_data.get("persona"),
            llm_used=metadata_data.get("llm_used", "template"),
            processing_async=metadata_data.get("processing_async", False)
        )
        
        return AgentResponse(
            session_id=data.get("session_id", ""),
            is_scam=data.get("is_scam", False),
            confidence_score=data.get("confidence_score", 0.0),
            agent_response=data.get("agent_response", ""),
            extracted_entities=extracted_entities,
            forensics=forensics,
            metadata=metadata
        )
    
    # =========================================================================
    # Synchronous Methods
    # =========================================================================
    
    def health_check(self) -> HealthStatus:
        """
        Check API health status.
        
        Returns:
            HealthStatus with API status, version, and component health
        
        Example:
            >>> health = client.health_check()
            >>> print(health.status, health.version)
            'healthy' '2.0.0'
        """
        response = self._client.get(
            f"{self.base_url}/health",
            headers=self._get_headers()
        )
        response.raise_for_status()
        data = response.json()
        return HealthStatus(
            status=data.get("status", "unknown"),
            version=data.get("version", "unknown"),
            timestamp=data.get("timestamp", ""),
            api_key_required=data.get("api_key_required", False),
            components=data.get("components", {})
        )
    
    def send_message(
        self, 
        session_id: str, 
        message: str,
        timestamp: Optional[str] = None
    ) -> AgentResponse:
        """
        Send a scammer message and get agent response.
        
        Args:
            session_id: Unique session identifier
            message: The scammer's message
            timestamp: Optional ISO timestamp (defaults to now)
        
        Returns:
            AgentResponse with detection results, intelligence, and agent reply
        
        Example:
            >>> response = client.send_message(
            ...     session_id="session-123",
            ...     message="Your account will be blocked!"
            ... )
            >>> print(response.is_scam, response.agent_response)
            True "Oh dear, what should I do?"
        """
        payload = {
            "session_id": session_id,
            "message": message,
            "timestamp": timestamp or datetime.now().isoformat()
        }
        
        response = self._client.post(
            f"{self.base_url}/message",
            headers=self._get_headers(),
            json=payload
        )
        response.raise_for_status()
        return self._parse_response(response.json())
    
    def get_session(self, session_id: str) -> Dict[str, Any]:
        """
        Get session details.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Dictionary with session state, history, and intelligence
        """
        response = self._client.get(
            f"{self.base_url}/session/{session_id}",
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_analytics(self, session_id: str) -> Dict[str, Any]:
        """
        Get detailed analytics for a session.
        
        Args:
            session_id: Session identifier
        
        Returns:
            Analytics including timeline, indicators, and intelligence score
        """
        response = self._client.get(
            f"{self.base_url}/analytics/{session_id}",
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get API performance metrics"""
        response = self._client.get(
            f"{self.base_url}/metrics",
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def lookup_profile(
        self,
        upi: Optional[str] = None,
        phone: Optional[str] = None,
        account: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Look up a scammer profile by identifier.
        
        Args:
            upi: UPI ID to search for
            phone: Phone number to search for
            account: Bank account number to search for
        
        Returns:
            Profile match result with profile data if found
        """
        params = {}
        if upi:
            params["upi"] = upi
        if phone:
            params["phone"] = phone
        if account:
            params["account"] = account
        
        response = self._client.get(
            f"{self.base_url}/profile/lookup",
            headers=self._get_headers(),
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    def close(self):
        """Close the HTTP client"""
        self._client.close()
        if self._async_client:
            asyncio.get_event_loop().run_until_complete(self._async_client.aclose())
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    # =========================================================================
    # Async Methods
    # =========================================================================
    
    async def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client"""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self.timeout)
        return self._async_client
    
    async def async_health_check(self) -> HealthStatus:
        """Async version of health_check"""
        client = await self._get_async_client()
        response = await client.get(
            f"{self.base_url}/health",
            headers=self._get_headers()
        )
        response.raise_for_status()
        data = response.json()
        return HealthStatus(
            status=data.get("status", "unknown"),
            version=data.get("version", "unknown"),
            timestamp=data.get("timestamp", ""),
            api_key_required=data.get("api_key_required", False),
            components=data.get("components", {})
        )
    
    async def async_send_message(
        self, 
        session_id: str, 
        message: str,
        timestamp: Optional[str] = None
    ) -> AgentResponse:
        """Async version of send_message"""
        client = await self._get_async_client()
        payload = {
            "session_id": session_id,
            "message": message,
            "timestamp": timestamp or datetime.now().isoformat()
        }
        
        response = await client.post(
            f"{self.base_url}/message",
            headers=self._get_headers(),
            json=payload
        )
        response.raise_for_status()
        return self._parse_response(response.json())
    
    async def async_close(self):
        """Async close the HTTP clients"""
        self._client.close()
        if self._async_client:
            await self._async_client.aclose()


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("Anti-Scam Sentinel SDK Test")
    print("=" * 50)
    
    client = AntiScamClient("http://localhost:8000")
    
    try:
        # Health check
        health = client.health_check()
        print(f"✅ API Status: {health.status} (v{health.version})")
        print(f"   API Key Required: {health.api_key_required}")
        
        # Send test message
        print("\n📤 Sending test message...")
        response = client.send_message(
            session_id="sdk-test-" + str(int(datetime.now().timestamp())),
            message="Your SBI account will be blocked within 24 hours due to KYC update pending."
        )
        
        print(f"\n🔍 Detection Results:")
        print(f"   Is Scam: {response.is_scam}")
        print(f"   Confidence: {response.confidence_score:.1%}")
        print(f"   Scam Type: {response.forensics.scam_type}")
        print(f"   Threat Level: {response.forensics.threat_level}")
        
        print(f"\n💬 Agent Response:")
        print(f"   {response.agent_response}")
        
        print(f"\n📊 Intelligence Score: {response.extracted_entities.intel_completeness_score}/100")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        client.close()
