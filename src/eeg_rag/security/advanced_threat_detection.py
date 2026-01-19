"""
Advanced Threat Detection System

This module provides enhanced security threat detection capabilities
beyond the basic dataset scanner, including behavioral analysis, 
anomaly detection, and real-time threat monitoring.

Features:
- Behavioral anomaly detection
- Real-time threat monitoring
- Advanced pattern recognition
- Security event correlation
- Automated threat response
"""

import logging
import time
import re
import hashlib
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
import json

from eeg_rag.security.dataset_scanner import ThreatLevel, ThreatType
from eeg_rag.utils.common_utils import (
    validate_non_empty_string,
    validate_positive_number, 
    format_error_message
)


@dataclass
class SecurityEvent:
    """Security event for threat tracking"""
    event_id: str
    threat_type: ThreatType
    threat_level: ThreatLevel
    source: str
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "event_id": self.event_id,
            "threat_type": self.threat_type.value,
            "threat_level": self.threat_level.value,
            "source": self.source,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "resolved": self.resolved
        }


@dataclass
class BehavioralPattern:
    """Behavioral pattern for anomaly detection"""
    pattern_id: str
    pattern_type: str
    baseline_frequency: float
    current_frequency: float
    anomaly_threshold: float = 3.0  # Standard deviations
    
    @property
    def is_anomalous(self) -> bool:
        """Check if current frequency is anomalous"""
        if self.baseline_frequency == 0:
            return self.current_frequency > 0
        
        deviation = abs(self.current_frequency - self.baseline_frequency)
        threshold = self.baseline_frequency * (self.anomaly_threshold / 100)
        return deviation > threshold


class AdvancedThreatDetector:
    """
    Advanced threat detection system with behavioral analysis
    and real-time monitoring capabilities
    """
    
    def __init__(
        self,
        monitoring_window_hours: float = 24.0,
        anomaly_threshold: float = 3.0,
        max_events_history: int = 10000,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize advanced threat detector
        
        Args:
            monitoring_window_hours: Time window for behavioral analysis
            anomaly_threshold: Threshold for anomaly detection (std deviations)
            max_events_history: Maximum security events to keep
            logger: Logger instance
        """
        self.monitoring_window_hours = validate_positive_number(
            monitoring_window_hours, "monitoring_window_hours"
        )
        self.anomaly_threshold = validate_positive_number(
            anomaly_threshold, "anomaly_threshold"
        )
        self.max_events_history = validate_positive_number(
            max_events_history, "max_events_history", min_value=100
        )
        
        self.logger = logger or logging.getLogger("eeg_rag.security.advanced_threat")
        
        # Event storage and tracking
        self.security_events: deque = deque(maxlen=self.max_events_history)
        self.threat_patterns: Dict[str, BehavioralPattern] = {}
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_stats = {
            "events_detected": 0,
            "threats_blocked": 0,
            "false_positives": 0,
            "monitoring_start_time": None
        }
        
        # Known attack patterns
        self.attack_signatures = self._load_attack_signatures()
        
        self.logger.info(
            f"AdvancedThreatDetector initialized: "
            f"window={self.monitoring_window_hours}h, "
            f"threshold={self.anomaly_threshold}"
        )
    
    def _load_attack_signatures(self) -> Dict[str, List[str]]:
        """Load known attack signature patterns"""
        return {
            "prompt_injection": [
                r"ignore\s+previous\s+instructions",
                r"system\s+prompt\s+override",
                r"\\x[0-9a-fA-F]{2}",  # Hex encoding
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"data:text/html",
                r"\{\{.*\}\}",  # Template injection
            ],
            "data_exfiltration": [
                r"SELECT\s+\*\s+FROM",
                r"UNION\s+SELECT",
                r"DROP\s+TABLE",
                r"INSERT\s+INTO",
                r"UPDATE\s+SET",
                r"DELETE\s+FROM",
                r"exec\s*\(",
                r"eval\s*\(",
            ],
            "malicious_payload": [
                r"<iframe[^>]*>",
                r"<object[^>]*>",
                r"<embed[^>]*>",
                r"file://",
                r"ftp://",
                r"\\\\[a-zA-Z0-9\\.]+\\",  # UNC paths
            ]
        }
    
    def start_monitoring(self) -> None:
        """Start real-time threat monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_stats["monitoring_start_time"] = datetime.now()
            self.logger.info("Started real-time threat monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop real-time threat monitoring"""
        if self.monitoring_active:
            self.monitoring_active = False
            duration = datetime.now() - self.monitoring_stats["monitoring_start_time"]
            self.logger.info(f"Stopped threat monitoring after {duration}")
    
    def detect_threats(
        self, 
        content: str, 
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[SecurityEvent]:
        """
        Detect threats in content using advanced pattern matching
        
        Args:
            content: Content to analyze
            source: Source of the content
            metadata: Additional metadata
            
        Returns:
            List of detected security events
        """
        content = validate_non_empty_string(content, "content")
        source = validate_non_empty_string(source, "source")
        metadata = metadata or {}
        
        threats = []
        
        try:
            # Check against known attack signatures
            for attack_type, patterns in self.attack_signatures.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
                    for match in matches:
                        event = self._create_security_event(
                            threat_type=ThreatType.PROMPT_INJECTION if "prompt" in attack_type else ThreatType.XSS_ATTACK,
                            threat_level=ThreatLevel.HIGH,
                            source=source,
                            description=f"Detected {attack_type} pattern: {pattern}",
                            metadata={
                                **metadata,
                                "matched_pattern": pattern,
                                "match_position": match.start(),
                                "match_text": match.group()[:100]  # First 100 chars
                            }
                        )
                        threats.append(event)
            
            # Behavioral anomaly detection
            behavioral_threats = self._detect_behavioral_anomalies(content, source, metadata)
            threats.extend(behavioral_threats)
            
            # Update behavioral patterns
            self._update_behavioral_patterns(content, source)
            
            # Store detected events
            for threat in threats:
                self.security_events.append(threat)
                self.monitoring_stats["events_detected"] += 1
                
                if threat.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                    self.monitoring_stats["threats_blocked"] += 1
            
            if threats:
                self.logger.warning(
                    f"Detected {len(threats)} threats in content from {source}"
                )
            
            return threats
            
        except Exception as e:
            error_msg = format_error_message("detect threats", e, {"source": source})
            self.logger.error(error_msg)
            return []
    
    def _create_security_event(
        self,
        threat_type: ThreatType,
        threat_level: ThreatLevel,
        source: str,
        description: str,
        metadata: Dict[str, Any]
    ) -> SecurityEvent:
        """Create a security event"""
        event_id = hashlib.sha256(
            f"{threat_type.value}{source}{description}{time.time()}".encode()
        ).hexdigest()[:16]
        
        return SecurityEvent(
            event_id=event_id,
            threat_type=threat_type,
            threat_level=threat_level,
            source=source,
            description=description,
            metadata=metadata
        )
    
    def _detect_behavioral_anomalies(
        self, 
        content: str, 
        source: str, 
        metadata: Dict[str, Any]
    ) -> List[SecurityEvent]:
        """Detect behavioral anomalies in content patterns"""
        anomalies = []
        
        # Analyze content characteristics
        content_stats = {
            "length": len(content),
            "word_count": len(content.split()),
            "line_count": content.count('\n'),
            "special_char_ratio": len(re.findall(r'[^a-zA-Z0-9\s]', content)) / len(content) if content else 0,
            "uppercase_ratio": len(re.findall(r'[A-Z]', content)) / len(content) if content else 0,
            "numeric_ratio": len(re.findall(r'[0-9]', content)) / len(content) if content else 0,
        }
        
        # Check for anomalous patterns
        if content_stats["special_char_ratio"] > 0.3:  # >30% special characters
            anomalies.append(self._create_security_event(
                threat_type=ThreatType.SUSPICIOUS_METADATA,
                threat_level=ThreatLevel.MEDIUM,
                source=source,
                description="High ratio of special characters detected",
                metadata={**metadata, "special_char_ratio": content_stats["special_char_ratio"]}
            ))
        
        if content_stats["length"] > 50000:  # Very long content
            anomalies.append(self._create_security_event(
                threat_type=ThreatType.PROMPT_INJECTION,
                threat_level=ThreatLevel.MEDIUM,
                source=source,
                description="Unusually long content detected",
                metadata={**metadata, "content_length": content_stats["length"]}
            ))
        
        # Check for repeating patterns (potential flooding/DoS)
        if self._detect_repetitive_patterns(content):
            anomalies.append(self._create_security_event(
                threat_type=ThreatType.SUSPICIOUS_METADATA,
                threat_level=ThreatLevel.MEDIUM,
                source=source,
                description="Repetitive patterns detected (possible DoS attempt)",
                metadata={**metadata, "pattern_type": "repetitive"}
            ))
        
        return anomalies
    
    def _detect_repetitive_patterns(self, content: str) -> bool:
        """Detect repetitive patterns that might indicate flooding"""
        if len(content) < 100:
            return False
        
        # Look for repeated substrings
        words = content.split()
        if len(words) < 10:
            return False
        
        word_counts = defaultdict(int)
        for word in words:
            if len(word) > 3:  # Ignore short words
                word_counts[word] += 1
        
        # Check if any word appears more than 50% of the time
        max_count = max(word_counts.values()) if word_counts else 0
        return max_count > len(words) * 0.5
    
    def _update_behavioral_patterns(self, content: str, source: str) -> None:
        """Update behavioral patterns for anomaly detection"""
        pattern_id = f"{source}_content_length"
        content_length = len(content)
        
        if pattern_id in self.threat_patterns:
            pattern = self.threat_patterns[pattern_id]
            # Simple moving average update
            pattern.current_frequency = (pattern.current_frequency * 0.9) + (content_length * 0.1)
        else:
            self.threat_patterns[pattern_id] = BehavioralPattern(
                pattern_id=pattern_id,
                pattern_type="content_length",
                baseline_frequency=content_length,
                current_frequency=content_length
            )
    
    def analyze_threat_landscape(self) -> Dict[str, Any]:
        """
        Analyze the current threat landscape based on collected events
        
        Returns:
            Dictionary with threat analysis
        """
        if not self.security_events:
            return {
                "total_events": 0,
                "threat_summary": {},
                "timeline": [],
                "recommendations": []
            }
        
        # Analyze threat types
        threat_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        source_counts = defaultdict(int)
        
        recent_events = [
            event for event in self.security_events
            if event.timestamp > datetime.now() - timedelta(hours=self.monitoring_window_hours)
        ]
        
        for event in recent_events:
            threat_counts[event.threat_type.value] += 1
            severity_counts[event.threat_level.value] += 1
            source_counts[event.source] += 1
        
        # Generate timeline
        timeline = []
        for i in range(24):  # Last 24 hours
            hour_start = datetime.now() - timedelta(hours=23-i)
            hour_end = hour_start + timedelta(hours=1)
            
            hour_events = [
                event for event in recent_events
                if hour_start <= event.timestamp < hour_end
            ]
            
            timeline.append({
                "hour": hour_start.strftime("%H:00"),
                "events": len(hour_events),
                "high_severity": len([e for e in hour_events if e.threat_level == ThreatLevel.HIGH])
            })
        
        # Generate recommendations
        recommendations = []
        
        if threat_counts:
            most_common_threat = max(threat_counts.items(), key=lambda x: x[1])
            recommendations.append(
                f"Most common threat: {most_common_threat[0]} ({most_common_threat[1]} occurrences)"
            )
        
        if severity_counts.get("high", 0) > 10:
            recommendations.append("High number of severe threats detected - review security policies")
        
        if len(source_counts) == 1:
            recommendations.append("All threats from single source - possible targeted attack")
        
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "monitoring_window_hours": self.monitoring_window_hours,
            "total_events": len(recent_events),
            "threat_summary": dict(threat_counts),
            "severity_summary": dict(severity_counts),
            "top_sources": dict(sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            "timeline": timeline,
            "recommendations": recommendations,
            "monitoring_stats": self.monitoring_stats.copy()
        }
    
    def get_threat_report(
        self, 
        include_resolved: bool = False,
        threat_level_filter: Optional[ThreatLevel] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive threat report
        
        Args:
            include_resolved: Include resolved threats
            threat_level_filter: Filter by threat level
            
        Returns:
            Dictionary with threat report
        """
        filtered_events = []
        
        for event in self.security_events:
            # Filter by resolution status
            if not include_resolved and event.resolved:
                continue
            
            # Filter by threat level
            if threat_level_filter and event.threat_level != threat_level_filter:
                continue
            
            filtered_events.append(event)
        
        # Sort by timestamp (newest first)
        filtered_events.sort(key=lambda e: e.timestamp, reverse=True)
        
        return {
            "report_timestamp": datetime.now().isoformat(),
            "total_events": len(filtered_events),
            "events": [event.to_dict() for event in filtered_events[:100]],  # Last 100 events
            "threat_landscape": self.analyze_threat_landscape()
        }
    
    def resolve_threat(self, event_id: str, resolution_note: str = "") -> bool:
        """
        Mark a threat as resolved
        
        Args:
            event_id: ID of the security event
            resolution_note: Note about the resolution
            
        Returns:
            True if threat was found and resolved
        """
        event_id = validate_non_empty_string(event_id, "event_id")
        
        for event in self.security_events:
            if event.event_id == event_id:
                event.resolved = True
                event.metadata["resolution_note"] = resolution_note
                event.metadata["resolved_timestamp"] = datetime.now().isoformat()
                
                self.logger.info(f"Resolved security event {event_id}: {resolution_note}")
                return True
        
        return False
    
    def export_threat_intelligence(self, filepath: Path) -> None:
        """
        Export threat intelligence data
        
        Args:
            filepath: Path to export file
        """
        intelligence_data = {
            "export_timestamp": datetime.now().isoformat(),
            "threat_detector_version": "1.0",
            "attack_signatures": self.attack_signatures,
            "behavioral_patterns": {
                pid: {
                    "pattern_id": pattern.pattern_id,
                    "pattern_type": pattern.pattern_type,
                    "baseline_frequency": pattern.baseline_frequency,
                    "current_frequency": pattern.current_frequency,
                    "is_anomalous": pattern.is_anomalous
                }
                for pid, pattern in self.threat_patterns.items()
            },
            "recent_events": [
                event.to_dict() for event in list(self.security_events)[-1000:]
            ],
            "threat_landscape": self.analyze_threat_landscape()
        }
        
        with open(filepath, 'w') as f:
            json.dump(intelligence_data, f, indent=2)
        
        self.logger.info(f"Exported threat intelligence to {filepath}")


class SecurityOrchestrator:
    """
    Orchestrates multiple security components for comprehensive protection
    """
    
    def __init__(self):
        """Initialize security orchestrator"""
        self.threat_detector = AdvancedThreatDetector()
        self.logger = logging.getLogger("eeg_rag.security.orchestrator")
        
        # Security policies
        self.policies = {
            "auto_block_critical": True,
            "quarantine_suspicious": True,
            "alert_on_medium_threats": True,
            "max_threat_events_per_hour": 100
        }
    
    def comprehensive_security_scan(
        self, 
        content: str, 
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive security scanning
        
        Args:
            content: Content to scan
            source: Source of content
            metadata: Additional metadata
            
        Returns:
            Security scan results
        """
        scan_start = time.time()
        
        # Advanced threat detection
        threats = self.threat_detector.detect_threats(content, source, metadata)
        
        # Determine overall security status
        max_threat_level = ThreatLevel.SAFE
        if threats:
            threat_levels = [threat.threat_level for threat in threats]
            if ThreatLevel.CRITICAL in threat_levels:
                max_threat_level = ThreatLevel.CRITICAL
            elif ThreatLevel.HIGH in threat_levels:
                max_threat_level = ThreatLevel.HIGH
            elif ThreatLevel.MEDIUM in threat_levels:
                max_threat_level = ThreatLevel.MEDIUM
            elif ThreatLevel.LOW in threat_levels:
                max_threat_level = ThreatLevel.LOW
        
        # Apply security policies
        action_taken = self._apply_security_policies(threats, max_threat_level)
        
        scan_duration = time.time() - scan_start
        
        return {
            "scan_timestamp": datetime.now().isoformat(),
            "source": source,
            "threats_detected": len(threats),
            "max_threat_level": max_threat_level.value,
            "threats": [threat.to_dict() for threat in threats],
            "action_taken": action_taken,
            "scan_duration_ms": scan_duration * 1000,
            "scan_passed": max_threat_level in [ThreatLevel.SAFE, ThreatLevel.LOW]
        }
    
    def _apply_security_policies(
        self, 
        threats: List[SecurityEvent], 
        max_threat_level: ThreatLevel
    ) -> str:
        """Apply security policies based on detected threats"""
        if not threats:
            return "none"
        
        if max_threat_level == ThreatLevel.CRITICAL and self.policies["auto_block_critical"]:
            return "blocked"
        elif max_threat_level == ThreatLevel.HIGH and self.policies["quarantine_suspicious"]:
            return "quarantined"
        elif max_threat_level == ThreatLevel.MEDIUM and self.policies["alert_on_medium_threats"]:
            return "alerted"
        else:
            return "logged"


# Export public interface
__all__ = [
    "SecurityEvent",
    "BehavioralPattern", 
    "AdvancedThreatDetector",
    "SecurityOrchestrator"
]
