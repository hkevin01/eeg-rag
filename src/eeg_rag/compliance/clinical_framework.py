"""
Clinical vs Research EEG Framework

Handles different requirements for clinical and research EEG systems:
- Clinical: 250+ electrode nodes (high-density clinical montages)
- Research: 128+1 reference (standard research configurations)
- Regulatory compliance (HIPAA, FDA, CE marking)
- Clinical workflow integration

Requirements:
- REQ-CLIN-001: Support clinical EEG configurations (250+ nodes)
- REQ-CLIN-002: Support research EEG configurations (128+ nodes)
- REQ-CLIN-003: HIPAA compliance for clinical data
- REQ-CLIN-004: FDA/CE marking readiness for clinical deployment
- REQ-CLIN-005: Audit trail for clinical decision support
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class EEGSystemType(Enum):
    """Type of EEG system"""
    CLINICAL = "clinical"  # 250+ nodes, clinical montages
    RESEARCH = "research"  # 128+1 reference, research montages
    HYBRID = "hybrid"  # Supports both


class MontageType(Enum):
    """EEG montage configurations"""
    # Clinical montages
    CLINICAL_STANDARD = "10-20"  # Standard 10-20 system
    CLINICAL_HIGH_DENSITY = "10-10"  # 10-10 system (128+ electrodes)
    CLINICAL_ULTRA_HIGH_DENSITY = "10-5"  # 10-5 system (250+ electrodes)
    
    # Research montages
    RESEARCH_32 = "32-channel"  # 32 electrodes + reference
    RESEARCH_64 = "64-channel"  # 64 electrodes + reference
    RESEARCH_128 = "128-channel"  # 128 electrodes + reference
    RESEARCH_256 = "256-channel"  # 256 electrodes + reference
    
    # Specialized
    EPILEPSY_MONITORING = "epilepsy"  # Epilepsy monitoring units
    ICU_MONITORING = "icu"  # ICU continuous EEG monitoring
    SLEEP_LAB = "sleep"  # Sleep laboratory
    INTRAOPERATIVE = "intraop"  # Intraoperative monitoring


class RegulatoryFramework(Enum):
    """Regulatory frameworks"""
    HIPAA = "hipaa"  # US healthcare data protection
    GDPR = "gdpr"  # EU data protection
    FDA_510K = "fda_510k"  # FDA medical device clearance
    FDA_PMA = "fda_pma"  # FDA premarket approval
    CE_MARK = "ce_mark"  # European Conformity marking
    ISO_13485 = "iso_13485"  # Medical device quality management
    IEC_60601 = "iec_60601"  # Medical electrical equipment safety


@dataclass
class EEGConfiguration:
    """
    EEG system configuration
    
    Attributes:
        system_type: Clinical, research, or hybrid
        montage_type: Electrode montage configuration
        num_electrodes: Number of electrodes
        reference_type: Reference electrode configuration
        sampling_rate: Sampling rate in Hz
        regulatory_compliance: List of applicable regulations
        clinical_use: Whether system is used for clinical care
        research_use: Whether system is used for research
    """
    system_type: EEGSystemType
    montage_type: MontageType
    num_electrodes: int
    reference_type: str  # e.g., "Cz", "average", "linked ears"
    sampling_rate: int  # Hz
    regulatory_compliance: List[RegulatoryFramework]
    clinical_use: bool = False
    research_use: bool = False
    
    def validate(self) -> List[str]:
        """
        Validate configuration
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Clinical systems must have enough electrodes
        if self.system_type == EEGSystemType.CLINICAL:
            if self.num_electrodes < 19:  # Minimum 10-20 system
                errors.append(f"Clinical systems require ≥19 electrodes (found {self.num_electrodes})")
            
            if self.clinical_use and not any(
                reg in [RegulatoryFramework.HIPAA, RegulatoryFramework.GDPR]
                for reg in self.regulatory_compliance
            ):
                errors.append("Clinical use requires HIPAA or GDPR compliance")
        
        # Research systems typically 128+1
        if self.system_type == EEGSystemType.RESEARCH:
            if self.num_electrodes < 32:
                errors.append(f"Research systems typically have ≥32 electrodes (found {self.num_electrodes})")
        
        # Sampling rate validation
        if self.sampling_rate < 250:
            errors.append(f"Sampling rate should be ≥250 Hz for clinical use (found {self.sampling_rate})")
        
        return errors
    
    def is_clinical_grade(self) -> bool:
        """Check if configuration meets clinical-grade requirements"""
        return (
            self.clinical_use and
            self.num_electrodes >= 19 and
            self.sampling_rate >= 250 and
            any(reg in [RegulatoryFramework.HIPAA, RegulatoryFramework.FDA_510K, RegulatoryFramework.CE_MARK]
                for reg in self.regulatory_compliance)
        )


@dataclass
class ClinicalWorkflow:
    """
    Clinical workflow configuration
    
    Defines how EEG-RAG integrates with clinical systems
    """
    workflow_name: str
    description: str
    use_case: str  # e.g., "epilepsy diagnosis", "sleep staging", "ICU monitoring"
    required_montage: MontageType
    required_electrodes: int
    regulatory_requirements: List[RegulatoryFramework]
    integration_points: List[str]  # e.g., "EMR", "PACS", "lab system"
    approval_workflow: bool  # Whether clinical approval is required
    audit_logging: bool  # Whether audit logs are required
    
    def get_integration_requirements(self) -> Dict[str, Any]:
        """Get requirements for clinical system integration"""
        return {
            'workflow_name': self.workflow_name,
            'use_case': self.use_case,
            'regulatory_compliance': [r.value for r in self.regulatory_requirements],
            'integration_points': self.integration_points,
            'requires_approval': self.approval_workflow,
            'requires_audit': self.audit_logging,
            'electrode_requirements': {
                'montage': self.required_montage.value,
                'minimum_electrodes': self.required_electrodes
            }
        }


class ClinicalComplianceFramework:
    """
    Clinical compliance and integration framework
    
    Manages regulatory compliance, clinical workflows, and system integration
    for both clinical and research EEG-RAG deployments.
    
    Usage:
        # Clinical deployment (epilepsy monitoring unit)
        framework = ClinicalComplianceFramework()
        
        config = EEGConfiguration(
            system_type=EEGSystemType.CLINICAL,
            montage_type=MontageType.CLINICAL_HIGH_DENSITY,
            num_electrodes=256,
            reference_type="average",
            sampling_rate=500,
            regulatory_compliance=[
                RegulatoryFramework.HIPAA,
                RegulatoryFramework.FDA_510K,
                RegulatoryFramework.CE_MARK
            ],
            clinical_use=True,
            research_use=False
        )
        
        workflow = framework.get_workflow("epilepsy_monitoring")
        validation = framework.validate_clinical_deployment(config, workflow)
        
        if validation['compliant']:
            print("✅ Ready for clinical deployment")
        else:
            print(f"❌ Compliance issues: {validation['issues']}")
    """
    
    def __init__(self):
        """Initialize clinical compliance framework"""
        self.workflows = self._load_workflows()
        self.configurations = {}
    
    def _load_workflows(self) -> Dict[str, ClinicalWorkflow]:
        """Load predefined clinical workflows"""
        return {
            'epilepsy_monitoring': ClinicalWorkflow(
                workflow_name="Epilepsy Monitoring Unit",
                description="Continuous EEG monitoring for seizure detection",
                use_case="epilepsy_diagnosis",
                required_montage=MontageType.EPILEPSY_MONITORING,
                required_electrodes=256,  # High-density for source localization
                regulatory_requirements=[
                    RegulatoryFramework.HIPAA,
                    RegulatoryFramework.FDA_510K,
                    RegulatoryFramework.CE_MARK
                ],
                integration_points=["EMR", "video_monitoring", "nurse_station"],
                approval_workflow=True,
                audit_logging=True
            ),
            'icu_monitoring': ClinicalWorkflow(
                workflow_name="ICU Continuous EEG Monitoring",
                description="Continuous EEG for critically ill patients",
                use_case="icu_monitoring",
                required_montage=MontageType.ICU_MONITORING,
                required_electrodes=256,  # High-density preferred
                regulatory_requirements=[
                    RegulatoryFramework.HIPAA,
                    RegulatoryFramework.FDA_510K,
                    RegulatoryFramework.IEC_60601
                ],
                integration_points=["EMR", "ICU_dashboard", "alert_system"],
                approval_workflow=True,
                audit_logging=True
            ),
            'sleep_lab': ClinicalWorkflow(
                workflow_name="Sleep Laboratory",
                description="Polysomnography and sleep staging",
                use_case="sleep_analysis",
                required_montage=MontageType.SLEEP_LAB,
                required_electrodes=32,  # Standard sleep montage
                regulatory_requirements=[
                    RegulatoryFramework.HIPAA,
                    RegulatoryFramework.FDA_510K
                ],
                integration_points=["EMR", "sleep_scoring_software"],
                approval_workflow=True,
                audit_logging=True
            ),
            'research_standard': ClinicalWorkflow(
                workflow_name="Research EEG",
                description="Standard research EEG recording",
                use_case="research",
                required_montage=MontageType.RESEARCH_128,
                required_electrodes=128,  # 128+1 reference
                regulatory_requirements=[
                    RegulatoryFramework.HIPAA,  # If using patient data
                    RegulatoryFramework.GDPR  # If in EU
                ],
                integration_points=["research_database", "analysis_software"],
                approval_workflow=False,  # Research doesn't require clinical approval
                audit_logging=True
            ),
            'cognitive_research': ClinicalWorkflow(
                workflow_name="Cognitive Neuroscience Research",
                description="ERP and cognitive EEG research",
                use_case="cognitive_research",
                required_montage=MontageType.RESEARCH_64,
                required_electrodes=64,
                regulatory_requirements=[
                    RegulatoryFramework.HIPAA  # If using patient data
                ],
                integration_points=["research_database", "stimulus_software"],
                approval_workflow=False,
                audit_logging=False
            )
        }
    
    def get_workflow(self, workflow_name: str) -> Optional[ClinicalWorkflow]:
        """
        Get predefined clinical workflow
        
        Args:
            workflow_name: Name of workflow
            
        Returns:
            ClinicalWorkflow or None if not found
        """
        return self.workflows.get(workflow_name)
    
    def validate_clinical_deployment(
        self,
        config: EEGConfiguration,
        workflow: ClinicalWorkflow
    ) -> Dict[str, Any]:
        """
        Validate configuration for clinical deployment
        
        Args:
            config: EEG configuration
            workflow: Clinical workflow
            
        Returns:
            Validation result with compliance status
        """
        issues = []
        
        # Validate configuration
        config_errors = config.validate()
        if config_errors:
            issues.extend(config_errors)
        
        # Check electrode requirements
        if config.num_electrodes < workflow.required_electrodes:
            issues.append(
                f"Workflow requires ≥{workflow.required_electrodes} electrodes, "
                f"config has {config.num_electrodes}"
            )
        
        # Check regulatory compliance
        for req_regulation in workflow.regulatory_requirements:
            if req_regulation not in config.regulatory_compliance:
                issues.append(f"Missing required regulation: {req_regulation.value}")
        
        # Check clinical use flag
        if workflow.approval_workflow and not config.clinical_use:
            issues.append("Clinical workflow requires clinical_use=True in configuration")
        
        return {
            'compliant': len(issues) == 0,
            'issues': issues,
            'workflow': workflow.workflow_name,
            'configuration_valid': len(config_errors) == 0,
            'regulatory_complete': all(
                reg in config.regulatory_compliance
                for reg in workflow.regulatory_requirements
            )
        }
    
    def get_integration_guide(self, workflow_name: str) -> Optional[Dict[str, Any]]:
        """
        Get integration guide for clinical system
        
        Args:
            workflow_name: Name of workflow
            
        Returns:
            Integration requirements and guide
        """
        workflow = self.get_workflow(workflow_name)
        if not workflow:
            return None
        
        return {
            'workflow': workflow.get_integration_requirements(),
            'setup_steps': [
                "1. Verify regulatory compliance documentation",
                "2. Configure EEG system with required montage",
                "3. Integrate with clinical systems (EMR, etc.)",
                "4. Enable audit logging",
                f"5. {'Set up clinical approval workflow' if workflow.approval_workflow else 'Configure research protocols'}",
                "6. Train clinical staff on system use",
                "7. Validate with test cases",
                "8. Go-live with monitoring"
            ],
            'regulatory_documents': [
                reg.value for reg in workflow.regulatory_requirements
            ],
            'integration_points': workflow.integration_points,
            'testing_requirements': [
                "Functional testing",
                "Integration testing",
                "Regulatory compliance testing",
                "User acceptance testing",
                "Performance testing"
            ]
        }
    
    def compare_clinical_vs_research(self) -> Dict[str, Any]:
        """
        Compare clinical vs research requirements
        
        Returns:
            Comparison matrix
        """
        return {
            'clinical': {
                'typical_electrodes': '250+ (high-density)',
                'montage': '10-5 system (ultra-high density)',
                'regulatory': 'HIPAA, FDA 510(k), CE Mark',
                'integration': 'EMR, PACS, clinical dashboards',
                'approval_required': True,
                'audit_logging': 'Required',
                'use_case': 'Diagnosis, treatment planning, monitoring',
                'data_retention': '7+ years (legal requirements)',
                'patient_consent': 'Required for clinical care'
            },
            'research': {
                'typical_electrodes': '128+1 reference',
                'montage': '10-10 or 10-20 system',
                'regulatory': 'HIPAA/GDPR (if using patient data), IRB approval',
                'integration': 'Research databases, analysis software',
                'approval_required': False,
                'audit_logging': 'Recommended',
                'use_case': 'Scientific research, publications, algorithm development',
                'data_retention': 'Per IRB protocol (typically 3-7 years)',
                'patient_consent': 'Required for research participation'
            },
            'key_differences': [
                'Clinical requires FDA/CE marking approval',
                'Clinical requires tighter integration with healthcare systems',
                'Clinical has stricter audit and approval workflows',
                'Research has more flexibility in electrode configurations',
                'Clinical requires higher data security and retention standards'
            ],
            'adoption_barriers_clinical': [
                'High regulatory burden (FDA clearance process)',
                'Integration with existing clinical systems',
                'Training requirements for clinical staff',
                'Liability and malpractice considerations',
                'Cost of clinical-grade hardware and validation'
            ],
            'adoption_strategy': [
                'Phase 1: Research deployment (lower regulatory burden)',
                'Phase 2: Clinical validation studies',
                'Phase 3: FDA 510(k) submission (if applicable)',
                'Phase 4: Clinical pilot programs',
                'Phase 5: Full clinical deployment'
            ]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get framework statistics"""
        return {
            'total_workflows': len(self.workflows),
            'clinical_workflows': sum(
                1 for w in self.workflows.values()
                if w.approval_workflow
            ),
            'research_workflows': sum(
                1 for w in self.workflows.values()
                if not w.approval_workflow
            ),
            'regulatory_frameworks': len(RegulatoryFramework),
            'montage_types': len(MontageType)
        }
