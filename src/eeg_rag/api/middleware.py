"""
Production-grade middleware for FastAPI.

Includes authentication, rate limiting, and telemetry.
"""

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Optional, Callable
import time
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import jwt
import os

logger = logging.getLogger(__name__)


# ============== Authentication Middleware ==============

# ---------------------------------------------------------------------------
# ID           : api.middleware.JWTAuthMiddleware
# Requirement  : `JWTAuthMiddleware` class shall be instantiable and expose the documented interface
# Purpose      : JWT token-based authentication middleware
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate JWTAuthMiddleware with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class JWTAuthMiddleware(BaseHTTPMiddleware):
    """
    JWT token-based authentication middleware.
    
    Validates Bearer tokens in Authorization header.
    Public endpoints bypass authentication.
    """
    
    PUBLIC_PATHS = {"/", "/health", "/docs", "/openapi.json", "/redoc"}
    
    # ---------------------------------------------------------------------------
    # ID           : api.middleware.JWTAuthMiddleware.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : app; secret_key: Optional[str] (default=None)
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __init__(self, app, secret_key: Optional[str] = None):
        super().__init__(app)
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY", "dev-secret-change-in-prod")
        self.algorithm = "HS256"
    
    # ---------------------------------------------------------------------------
    # ID           : api.middleware.JWTAuthMiddleware.dispatch
    # Requirement  : `dispatch` shall process request with JWT validation
    # Purpose      : Process request with JWT validation
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : request: Request; call_next: Callable
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def dispatch(self, request: Request, call_next: Callable):
        """Process request with JWT validation."""
        
        # Skip auth for public paths
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)
        
        # Extract token
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing or invalid authorization header"}
            )
        
        token = auth_header.replace("Bearer ", "")
        
        try:
            # Decode and validate token
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Add user info to request state
            request.state.user_id = payload.get("sub")
            request.state.username = payload.get("username")
            request.state.roles = payload.get("roles", [])
            
            logger.debug(f"Authenticated user: {request.state.username}")
            
        except jwt.ExpiredSignatureError:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Token has expired"}
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid token"}
            )
        
        response = await call_next(request)
        return response


# ---------------------------------------------------------------------------
# ID           : api.middleware.create_access_token
# Requirement  : `create_access_token` shall generate JWT access token
# Purpose      : Generate JWT access token
# Rationale    : Implements domain-specific logic per system design; see referenced specs
# Inputs       : user_id: str; username: str; roles: list; expires_in_hours: int (default=24)
# Outputs      : str
# Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
# Postcond.    : Return value satisfies documented output type and range
# Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
# Side Effects : May update instance state or perform I/O; see body
# Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
# Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
# Constraints  : Synchronous — must not block event loop
# Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
def create_access_token(user_id: str, username: str, roles: list, 
                       expires_in_hours: int = 24) -> str:
    """
    Generate JWT access token.
    
    Args:
        user_id: Unique user identifier
        username: Username for display
        roles: List of user roles
        expires_in_hours: Token expiration time
    
    Returns:
        Encoded JWT token
    """
    secret_key = os.getenv("JWT_SECRET_KEY", "dev-secret-change-in-prod")
    
    payload = {
        "sub": user_id,
        "username": username,
        "roles": roles,
        "exp": datetime.utcnow() + timedelta(hours=expires_in_hours),
        "iat": datetime.utcnow()
    }
    
    token = jwt.encode(payload, secret_key, algorithm="HS256")
    return token


# ============== Rate Limiting Middleware ==============

# ---------------------------------------------------------------------------
# ID           : api.middleware.RateLimitMiddleware
# Requirement  : `RateLimitMiddleware` class shall be instantiable and expose the documented interface
# Purpose      : Token bucket rate limiting per user/IP
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate RateLimitMiddleware with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Token bucket rate limiting per user/IP.
    
    Limits:
    - Free tier: 10 requests/minute, 100/hour
    - Premium tier: 100 requests/minute, 1000/hour
    """
    
    # ---------------------------------------------------------------------------
    # ID           : api.middleware.RateLimitMiddleware.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : app
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __init__(self, app):
        super().__init__(app)
        self.requests: Dict[str, list] = defaultdict(list)
        self.limits = {
            "free": {"minute": 10, "hour": 100},
            "premium": {"minute": 100, "hour": 1000},
            "admin": {"minute": 1000, "hour": 10000}
        }
    
    # ---------------------------------------------------------------------------
    # ID           : api.middleware.RateLimitMiddleware.dispatch
    # Requirement  : `dispatch` shall apply rate limiting
    # Purpose      : Apply rate limiting
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : request: Request; call_next: Callable
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def dispatch(self, request: Request, call_next: Callable):
        """Apply rate limiting."""
        
        # Identify user (authenticated) or IP (anonymous)
        user_key = getattr(request.state, "user_id", None) or request.client.host
        
        # Get user tier from roles
        user_roles = getattr(request.state, "roles", [])
        tier = "admin" if "admin" in user_roles else "premium" if "premium" in user_roles else "free"
        
        # Check rate limits
        now = time.time()
        request_times = self.requests[user_key]
        
        # Remove old requests
        request_times[:] = [t for t in request_times if now - t < 3600]  # Keep last hour
        
        # Count requests in windows
        minute_count = sum(1 for t in request_times if now - t < 60)
        hour_count = len(request_times)
        
        # Check limits
        limits = self.limits[tier]
        if minute_count >= limits["minute"]:
            logger.warning(f"Rate limit exceeded: {user_key} ({tier}) - minute")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": f"Rate limit exceeded: {limits['minute']} requests per minute",
                    "retry_after": 60 - (now - max(request_times))
                }
            )
        
        if hour_count >= limits["hour"]:
            logger.warning(f"Rate limit exceeded: {user_key} ({tier}) - hour")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": f"Rate limit exceeded: {limits['hour']} requests per hour",
                    "retry_after": 3600 - (now - min(request_times))
                }
            )
        
        # Record request
        request_times.append(now)
        
        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit-Minute"] = str(limits["minute"])
        response.headers["X-RateLimit-Remaining-Minute"] = str(limits["minute"] - minute_count - 1)
        response.headers["X-RateLimit-Limit-Hour"] = str(limits["hour"])
        response.headers["X-RateLimit-Remaining-Hour"] = str(limits["hour"] - hour_count - 1)
        
        return response


# ============== Request Telemetry Middleware ==============

# ---------------------------------------------------------------------------
# ID           : api.middleware.TelemetryMiddleware
# Requirement  : `TelemetryMiddleware` class shall be instantiable and expose the documented interface
# Purpose      : Request telemetry and performance tracking
# Rationale    : Object-oriented encapsulation isolates state and enforces invariants
# Inputs       : Constructor arguments — see __init__ signature
# Outputs      : N/A (class definition)
# Precond.     : All imported dependencies must be available at import time
# Postcond.    : Instance attributes initialised as documented; invariants hold
# Assumptions  : Python runtime ≥ 3.9; package dependencies installed
# Side Effects : May allocate heap memory; __init__ may open connections or load models
# Fail Modes   : ImportError if dependency missing; TypeError for invalid constructor args
# Err Handling : Constructor raises on invalid args; see __init__ body
# Constraints  : Thread-safety not guaranteed unless explicitly documented
# Verification : Instantiate TelemetryMiddleware with valid args; assert attribute types and values
# References   : EEG-RAG system design specification; see module docstring
# ---------------------------------------------------------------------------
class TelemetryMiddleware(BaseHTTPMiddleware):
    """
    Request telemetry and performance tracking.
    
    Records:
    - Request duration
    - Status codes
    - User agents
    - Error rates
    """
    
    # ---------------------------------------------------------------------------
    # ID           : api.middleware.TelemetryMiddleware.__init__
    # Requirement  : `__init__` shall execute as specified
    # Purpose      :   init  
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : app
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def __init__(self, app):
        super().__init__(app)
        self.metrics = {
            "total_requests": 0,
            "total_errors": 0,
            "response_times": [],
            "status_codes": defaultdict(int)
        }
    
    # ---------------------------------------------------------------------------
    # ID           : api.middleware.TelemetryMiddleware.dispatch
    # Requirement  : `dispatch` shall track request metrics
    # Purpose      : Track request metrics
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : request: Request; call_next: Callable
    # Outputs      : Implicitly None or see body
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Must be awaited (async)
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    async def dispatch(self, request: Request, call_next: Callable):
        """Track request metrics."""
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics["total_requests"] += 1
            self.metrics["response_times"].append(duration)
            self.metrics["status_codes"][response.status_code] += 1
            
            # Keep only recent response times (last 1000)
            if len(self.metrics["response_times"]) > 1000:
                self.metrics["response_times"] = self.metrics["response_times"][-1000:]
            
            # Add telemetry headers
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            response.headers["X-Request-ID"] = str(request.state.request_id) if hasattr(request.state, "request_id") else "unknown"
            
            # Log slow requests
            if duration > 2.0:
                logger.warning(
                    f"Slow request: {request.method} {request.url.path} - {duration:.2f}s"
                )
            
            return response
            
        except Exception as e:
            self.metrics["total_errors"] += 1
            logger.error(f"Request error: {e}")
            raise
    
    # ---------------------------------------------------------------------------
    # ID           : api.middleware.TelemetryMiddleware.get_metrics
    # Requirement  : `get_metrics` shall get current metrics snapshot
    # Purpose      : Get current metrics snapshot
    # Rationale    : Implements domain-specific logic per system design; see referenced specs
    # Inputs       : None
    # Outputs      : Dict
    # Precond.     : Owning object properly initialised (if method); inputs within documented valid ranges
    # Postcond.    : Return value satisfies documented output type and range
    # Assumptions  : Python runtime ≥ 3.9; inputs are well-typed at call site
    # Side Effects : May update instance state or perform I/O; see body
    # Fail Modes   : Invalid inputs raise ValueError/TypeError; I/O failures raise OSError or subclass
    # Err Handling : Validates critical inputs at boundary; propagates unexpected exceptions
    # Constraints  : Synchronous — must not block event loop
    # Verification : Unit test with representative, boundary, and invalid inputs; assert return satisfies postcondition
    # References   : EEG-RAG system design specification; see module docstring
    # ---------------------------------------------------------------------------
    def get_metrics(self) -> Dict:
        """Get current metrics snapshot."""
        response_times = self.metrics["response_times"]
        
        return {
            "total_requests": self.metrics["total_requests"],
            "total_errors": self.metrics["total_errors"],
            "error_rate": self.metrics["total_errors"] / max(self.metrics["total_requests"], 1),
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "p95_response_time": sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0,
            "p99_response_time": sorted(response_times)[int(len(response_times) * 0.99)] if response_times else 0,
            "status_codes": dict(self.metrics["status_codes"])
        }
