"""
Configuration management for EEG-RAG system.

This module handles all configuration loading, validation, and environment
variable management. It ensures that the system fails gracefully when
configuration is invalid or missing.

REQ-001: Configuration must be loaded from environment variables
REQ-002: All paths must be validated before use
REQ-003: API keys must never be logged or exposed
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """
    Central configuration class for EEG-RAG system.
    
    All configuration values are loaded from environment variables with
    sensible defaults. Validates configuration on initialization to fail
    fast if critical values are missing.
    
    Attributes:
        openai_api_key: OpenAI API key (CRITICAL - must be set)
        openai_model: Model to use for generation (default: gpt-3.5-turbo)
        openai_max_tokens: Maximum tokens for generation (default: 1000)
        openai_temperature: Temperature for generation (default: 0.2, range: 0-1)
        data_raw_dir: Directory for raw data files
        data_processed_dir: Directory for processed data
        data_embeddings_dir: Directory for embeddings and indices
        faiss_index_path: Path to FAISS index file
        faiss_metadata_path: Path to chunk metadata file
        embedding_model: HuggingFace model name for embeddings
        embedding_dim: Dimension of embedding vectors
        default_top_k: Default number of results to retrieve
        min_similarity_score: Minimum similarity score threshold (0-1)
        chunk_size: Size of text chunks in tokens
        chunk_overlap: Overlap between chunks in tokens
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        max_workers: Maximum number of parallel workers
        batch_size: Batch size for processing
        enable_gpu: Whether to use GPU if available
        debug: Enable debug mode with verbose logging
        profile_performance: Enable performance profiling
    """
    
    # OpenAI Configuration (CRITICAL)
    openai_api_key: str = ""
    openai_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int = 1000
    openai_temperature: float = 0.2
    
    # Data Paths
    data_raw_dir: Path = field(default_factory=lambda: Path("data/raw"))
    data_processed_dir: Path = field(default_factory=lambda: Path("data/processed"))
    data_embeddings_dir: Path = field(default_factory=lambda: Path("data/embeddings"))
    
    # FAISS Configuration
    faiss_index_path: Path = field(
        default_factory=lambda: Path("data/embeddings/faiss_index.bin")
    )
    faiss_metadata_path: Path = field(
        default_factory=lambda: Path("data/embeddings/chunk_metadata.jsonl")
    )
    
    # Embedding Configuration
    embedding_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    embedding_dim: int = 768
    
    # Retrieval Configuration
    default_top_k: int = 10
    min_similarity_score: float = 0.5
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Neo4j Configuration (Optional)
    neo4j_uri: Optional[str] = None
    neo4j_user: Optional[str] = None
    neo4j_password: Optional[str] = None
    neo4j_database: str = "eeg-rag"
    
    # Redis Configuration (Optional)
    redis_host: Optional[str] = None
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    cache_ttl: int = 3600  # Cache TTL in seconds
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    
    # PubMed Configuration
    pubmed_api_key: Optional[str] = None
    pubmed_email: Optional[str] = None
    pubmed_rate_limit: int = 3  # Requests per second
    
    # Performance Configuration
    max_workers: int = 4
    batch_size: int = 32
    enable_gpu: bool = False
    
    # Development Settings
    debug: bool = False
    profile_performance: bool = False
    
    def __post_init__(self) -> None:
        """
        Validate configuration after initialization.
        
        REQ-004: Configuration validation must occur at startup
        REQ-005: Invalid configuration must raise descriptive errors
        
        Raises:
            ValueError: If critical configuration is missing or invalid
        """
        self._validate_critical_config()
        self._validate_numeric_bounds()
        self._validate_paths()
        
        if self.debug:
            logger.info("Configuration loaded successfully")
            logger.debug(f"Embedding model: {self.embedding_model}")
            logger.debug(f"Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
    
    def _validate_critical_config(self) -> None:
        """
        Validate critical configuration that must be present.
        
        REQ-006: OpenAI API key must be set for generation
        
        Raises:
            ValueError: If critical configuration is missing
        """
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable. "
                "Get your key from https://platform.openai.com/api-keys"
            )
        
        # Validate API key format (should start with 'sk-')
        if not self.openai_api_key.startswith("sk-"):
            logger.warning(
                "OpenAI API key format appears invalid. "
                "Valid keys typically start with 'sk-'"
            )
    
    def _validate_numeric_bounds(self) -> None:
        """
        Validate that numeric configuration values are within valid ranges.
        
        REQ-007: Numeric configuration must be within valid bounds
        REQ-008: Temperature must be between 0 and 1
        REQ-009: Token counts must be positive
        
        Raises:
            ValueError: If numeric values are out of bounds
        """
        # Validate temperature (0-1 range for most LLMs)
        if not 0 <= self.openai_temperature <= 1:
            raise ValueError(
                f"openai_temperature must be between 0 and 1, got {self.openai_temperature}"
            )
        
        # Validate token counts
        if self.openai_max_tokens <= 0:
            raise ValueError(
                f"openai_max_tokens must be positive, got {self.openai_max_tokens}"
            )
        
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        
        if self.chunk_overlap < 0:
            raise ValueError(
                f"chunk_overlap must be non-negative, got {self.chunk_overlap}"
            )
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        
        # Validate similarity score (0-1 range)
        if not 0 <= self.min_similarity_score <= 1:
            raise ValueError(
                f"min_similarity_score must be between 0 and 1, "
                f"got {self.min_similarity_score}"
            )
        
        # Validate top_k
        if self.default_top_k <= 0:
            raise ValueError(
                f"default_top_k must be positive, got {self.default_top_k}"
            )
        
        # Validate worker and batch size
        if self.max_workers <= 0:
            raise ValueError(f"max_workers must be positive, got {self.max_workers}")
        
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        # Validate cache TTL (must be positive, measured in seconds)
        if self.cache_ttl <= 0:
            raise ValueError(
                f"cache_ttl must be positive (seconds), got {self.cache_ttl}"
            )
        
        # Validate PubMed rate limit
        if self.pubmed_rate_limit <= 0:
            raise ValueError(
                f"pubmed_rate_limit must be positive, got {self.pubmed_rate_limit}"
            )
    
    def _validate_paths(self) -> None:
        """
        Validate and create necessary directories.
        
        REQ-010: Required directories must exist or be created
        REQ-011: Path validation must occur before use
        
        Creates directories if they don't exist. Logs warnings for missing
        index files (which is expected on first run).
        """
        # Create data directories if they don't exist
        for directory in [
            self.data_raw_dir,
            self.data_processed_dir,
            self.data_embeddings_dir,
        ]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {directory}")
            except Exception as exc:
                raise ValueError(
                    f"Failed to create directory {directory}: {exc}"
                ) from exc
        
        # Create log directory if log_file is specified
        if self.log_file:
            try:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                raise ValueError(
                    f"Failed to create log directory {self.log_file.parent}: {exc}"
                ) from exc
    
    @classmethod
    def from_env(cls, env_file: Optional[str] = ".env") -> "Config":
        """
        Create configuration from environment variables.
        
        Loads environment variables from .env file if it exists, then creates
        a Config instance with values from environment variables, using
        defaults where not specified.
        
        REQ-012: Configuration must support loading from .env files
        REQ-013: Missing .env file should not cause failure
        
        Args:
            env_file: Path to .env file (default: ".env" in current directory)
        
        Returns:
            Config instance with values loaded from environment
        
        Example:
            >>> config = Config.from_env()
            >>> print(config.embedding_model)
            'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
        """
        # Load .env file if it exists (don't fail if missing)
        if env_file and os.path.exists(env_file):
            load_dotenv(env_file)
            logger.info(f"Loaded environment from {env_file}")
        else:
            logger.debug(f"No .env file found at {env_file}, using environment variables")
        
        # Helper function to safely get and convert environment variables
        def get_env(key: str, default: Any, convert_fn=str) -> Any:
            """Get environment variable with type conversion and default."""
            value = os.getenv(key)
            if value is None:
                return default
            try:
                return convert_fn(value)
            except (ValueError, TypeError) as exc:
                logger.warning(
                    f"Failed to convert {key}={value} to {convert_fn.__name__}, "
                    f"using default {default}: {exc}"
                )
                return default
        
        # Helper for boolean conversion
        def str_to_bool(value: str) -> bool:
            """Convert string to boolean."""
            return value.lower() in ("true", "1", "yes", "on")
        
        # Helper for Path conversion
        def str_to_path(value: str) -> Path:
            """Convert string to Path."""
            return Path(value)
        
        # Build configuration from environment
        return cls(
            # OpenAI
            openai_api_key=get_env("OPENAI_API_KEY", ""),
            openai_model=get_env("OPENAI_MODEL", "gpt-3.5-turbo"),
            openai_max_tokens=get_env("OPENAI_MAX_TOKENS", 1000, int),
            openai_temperature=get_env("OPENAI_TEMPERATURE", 0.2, float),
            
            # Data Paths
            data_raw_dir=get_env("DATA_RAW_DIR", "data/raw", str_to_path),
            data_processed_dir=get_env("DATA_PROCESSED_DIR", "data/processed", str_to_path),
            data_embeddings_dir=get_env("DATA_EMBEDDINGS_DIR", "data/embeddings", str_to_path),
            
            # FAISS
            faiss_index_path=get_env(
                "FAISS_INDEX_PATH", "data/embeddings/faiss_index.bin", str_to_path
            ),
            faiss_metadata_path=get_env(
                "FAISS_METADATA_PATH", "data/embeddings/chunk_metadata.jsonl", str_to_path
            ),
            
            # Embeddings
            embedding_model=get_env(
                "EMBEDDING_MODEL",
                "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            ),
            embedding_dim=get_env("EMBEDDING_DIM", 768, int),
            
            # Retrieval
            default_top_k=get_env("DEFAULT_TOP_K", 10, int),
            min_similarity_score=get_env("MIN_SIMILARITY_SCORE", 0.5, float),
            chunk_size=get_env("CHUNK_SIZE", 512, int),
            chunk_overlap=get_env("CHUNK_OVERLAP", 50, int),
            
            # Neo4j (optional)
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_user=os.getenv("NEO4J_USER"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            neo4j_database=get_env("NEO4J_DATABASE", "eeg-rag"),
            
            # Redis (optional)
            redis_host=os.getenv("REDIS_HOST"),
            redis_port=get_env("REDIS_PORT", 6379, int),
            redis_db=get_env("REDIS_DB", 0, int),
            redis_password=os.getenv("REDIS_PASSWORD"),
            cache_ttl=get_env("CACHE_TTL", 3600, int),
            
            # Logging
            log_level=get_env("LOG_LEVEL", "INFO"),
            log_file=get_env("LOG_FILE", None, str_to_path) if os.getenv("LOG_FILE") else None,
            
            # PubMed
            pubmed_api_key=os.getenv("PUBMED_API_KEY"),
            pubmed_email=os.getenv("PUBMED_EMAIL"),
            pubmed_rate_limit=get_env("PUBMED_RATE_LIMIT", 3, int),
            
            # Performance
            max_workers=get_env("MAX_WORKERS", 4, int),
            batch_size=get_env("BATCH_SIZE", 32, int),
            enable_gpu=get_env("ENABLE_GPU", False, str_to_bool),
            
            # Development
            debug=get_env("DEBUG", False, str_to_bool),
            profile_performance=get_env("PROFILE_PERFORMANCE", False, str_to_bool),
        )
    
    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        REQ-014: Configuration must be serializable for logging
        REQ-015: Secrets must never be included in logs
        
        Args:
            include_secrets: Whether to include API keys (default: False)
        
        Returns:
            Dictionary representation of configuration
        """
        config_dict = {
            "openai_model": self.openai_model,
            "openai_max_tokens": self.openai_max_tokens,
            "openai_temperature": self.openai_temperature,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "default_top_k": self.default_top_k,
            "min_similarity_score": self.min_similarity_score,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "log_level": self.log_level,
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "enable_gpu": self.enable_gpu,
        }
        
        if include_secrets:
            config_dict["openai_api_key"] = self.openai_api_key
            config_dict["neo4j_password"] = self.neo4j_password
            config_dict["redis_password"] = self.redis_password
        else:
            config_dict["openai_api_key"] = "***REDACTED***"
            config_dict["neo4j_password"] = "***REDACTED***"
            config_dict["redis_password"] = "***REDACTED***"
        
        return config_dict
