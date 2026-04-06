"""
ClinicalTrials.gov client for EEG/neurology trial data.

Uses the ClinicalTrials.gov REST API v2 to retrieve trials
relevant to EEG-based neurology: epilepsy, BCI, sleep, and
cognitive/psychiatric applications.

API docs: https://clinicaltrials.gov/data-api/api
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class ClinicalTrial:
    """Structured clinical trial record from ClinicalTrials.gov."""

    nct_id: str                         # NCT number (primary key)
    title: str
    brief_summary: str
    detailed_description: str
    conditions: List[str]
    interventions: List[str]
    status: str                         # Recruiting, Completed, etc.
    phase: Optional[str]                # Phase 1/2/3/4/N/A
    study_type: str                     # Interventional / Observational
    start_date: Optional[datetime]
    completion_date: Optional[datetime]
    enrollment: Optional[int]
    sponsors: List[str]
    locations: List[str]
    primary_outcomes: List[str]
    secondary_outcomes: List[str]
    eligibility_criteria: str
    mesh_terms: List[str]
    keywords: List[str]
    eeg_relevant: bool = False          # Flagged by EEG keyword scan
    eeg_methods_mentioned: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to unified document dict for ingestion pipeline."""
        return {
            "nct_id": self.nct_id,
            "title": self.title,
            "abstract": self.brief_summary,
            "detailed_description": self.detailed_description,
            "conditions": self.conditions,
            "interventions": self.interventions,
            "status": self.status,
            "phase": self.phase,
            "study_type": self.study_type,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "completion_date": (
                self.completion_date.isoformat() if self.completion_date else None
            ),
            "enrollment": self.enrollment,
            "sponsors": self.sponsors,
            "locations": self.locations,
            "primary_outcomes": self.primary_outcomes,
            "secondary_outcomes": self.secondary_outcomes,
            "eligibility_criteria": self.eligibility_criteria,
            "mesh_terms": self.mesh_terms,
            "keywords": self.keywords,
            "eeg_relevant": self.eeg_relevant,
            "eeg_methods_mentioned": self.eeg_methods_mentioned,
            "source": "clinicaltrials",
        }


class ClinicalTrialsClient:
    """
    Async client for ClinicalTrials.gov REST API v2.

    Focuses on EEG-relevant clinical trials in:
    - Epilepsy and seizure disorders
    - Brain-computer interfaces (BCI) and neuroprosthetics
    - Sleep disorders (EEG-diagnosed)
    - Cognitive and psychiatric disorders with EEG endpoints
    - Neurofeedback and biofeedback protocols

    Rate limit: Reasonable use; no strict published limit.
    """

    BASE_URL = "https://clinicaltrials.gov/api/v2"

    # EEG-relevant search terms for ClinicalTrials.gov queries
    EEG_SEARCH_QUERIES: List[str] = [
        "electroencephalography",
        "EEG biomarker",
        "EEG monitoring epilepsy",
        "brain-computer interface",
        "neurofeedback",
        "EEG sleep",
        "event-related potential",
        "EEG seizure detection",
        "quantitative EEG",
        "EEG biofeedback",
        "motor imagery BCI",
        "EEG emotion recognition",
        "EEG cognitive",
    ]

    # Keywords that flag a trial as EEG-relevant
    EEG_METHOD_KEYWORDS: List[str] = [
        "electroencephalograph",
        r"\beeg\b",
        "quantitative EEG",
        "event-related potential",
        r"\berp\b",
        "brain-computer interface",
        r"\bbci\b",
        "neurofeedback",
        "brain oscillation",
        "brain wave",
        "sleep EEG",
        "polysomnography",
        "brain mapping",
        "source localization",
        "P300",
        "N400",
        "steady-state visually evoked",
        "SSVEP",
        "motor imagery",
        "EEG electrode",
        "scalp electrode",
    ]

    def __init__(
        self,
        page_size: int = 100,
        timeout: float = 30.0,
    ):
        self.page_size = page_size
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self) -> "ClinicalTrialsClient":
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._session:
            await self._session.close()
            self._session = None
        return False

    # ------------------------------------------------------------------
    # Public search interface
    # ------------------------------------------------------------------

    async def search_eeg_trials(
        self,
        max_results: int = 500,
        status_filter: Optional[List[str]] = None,
    ) -> AsyncIterator[ClinicalTrial]:
        """
        Iterate over EEG-relevant clinical trials.

        Args:
            max_results: Maximum total results to yield.
            status_filter: Restrict to these status values, e.g.
                           ["RECRUITING", "COMPLETED"]. None = all.

        Yields:
            ClinicalTrial records flagged as EEG-relevant.
        """
        seen: set[str] = set()
        yielded = 0

        for query in self.EEG_SEARCH_QUERIES:
            if yielded >= max_results:
                break
            async for trial in self._paginate_search(
                query=query,
                status_filter=status_filter,
            ):
                if yielded >= max_results:
                    break
                if trial.nct_id in seen:
                    continue
                seen.add(trial.nct_id)
                if trial.eeg_relevant:
                    yield trial
                    yielded += 1

    async def fetch_by_nct_id(self, nct_id: str) -> Optional[ClinicalTrial]:
        """Fetch a single trial by NCT identifier."""
        url = f"{self.BASE_URL}/studies/{nct_id}"
        data = await self._get_json(url, params={"format": "json"})
        if not data:
            return None
        return self._parse_study(data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _paginate_search(
        self,
        query: str,
        status_filter: Optional[List[str]],
    ) -> AsyncIterator[ClinicalTrial]:
        params: Dict[str, Any] = {
            "query.term": query,
            "pageSize": str(self.page_size),
            "format": "json",
            "fields": (
                "NCTId,BriefTitle,BriefSummary,DetailedDescription,"
                "Condition,InterventionName,OverallStatus,Phase,"
                "StudyType,StartDate,CompletionDate,EnrollmentCount,"
                "LeadSponsorName,LocationFacility,"
                "PrimaryOutcomeDescription,SecondaryOutcomeDescription,"
                "EligibilityCriteria,MeshTerm,Keyword"
            ),
        }
        if status_filter:
            params["filter.overallStatus"] = ",".join(status_filter)

        next_page_token: Optional[str] = None

        while True:
            if next_page_token:
                params["pageToken"] = next_page_token
            else:
                params.pop("pageToken", None)

            data = await self._get_json(f"{self.BASE_URL}/studies", params=params)
            if not data:
                break

            studies = data.get("studies", [])
            for study_data in studies:
                trial = self._parse_study(study_data)
                if trial:
                    yield trial

            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break

            await asyncio.sleep(0.34)  # ~3 req/s polite rate

    def _parse_study(self, data: Dict[str, Any]) -> Optional[ClinicalTrial]:
        """Parse a ClinicalTrials API v2 study record."""
        try:
            proto = data.get("protocolSection", {})
            id_mod = proto.get("identificationModule", {})
            desc_mod = proto.get("descriptionModule", {})
            cond_mod = proto.get("conditionsModule", {})
            design_mod = proto.get("designModule", {})
            status_mod = proto.get("statusModule", {})
            sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
            outcomes_mod = proto.get("outcomesModule", {})
            eligibility_mod = proto.get("eligibilityModule", {})
            contacts_mod = proto.get("contactsLocationsModule", {})

            nct_id = id_mod.get("nctId", "")
            if not nct_id:
                return None

            # Dates
            start_date = self._parse_date(
                status_mod.get("startDateStruct", {}).get("date")
            )
            completion_date = self._parse_date(
                status_mod.get("primaryCompletionDateStruct", {}).get("date")
                or status_mod.get("completionDateStruct", {}).get("date")
            )

            # Enrollment
            enrollment_info = design_mod.get("enrollmentInfo", {})
            try:
                enrollment = int(enrollment_info.get("count", 0)) or None
            except (TypeError, ValueError):
                enrollment = None

            # Interventions
            interventions = [
                iv.get("name", "")
                for iv in data.get("protocolSection", {})
                .get("armsInterventionsModule", {})
                .get("interventions", [])
            ]

            # Outcomes
            primary_outcomes = [
                o.get("description", o.get("measure", ""))
                for o in outcomes_mod.get("primaryOutcomes", [])
            ]
            secondary_outcomes = [
                o.get("description", o.get("measure", ""))
                for o in outcomes_mod.get("secondaryOutcomes", [])
            ]

            # Sponsors
            sponsors = [sponsor_mod.get("leadSponsor", {}).get("name", "")]
            sponsors += [
                c.get("name", "")
                for c in sponsor_mod.get("collaborators", [])
            ]

            # Locations
            locations = list(
                {
                    loc.get("facility", {}).get("name", "")
                    for loc in contacts_mod.get("locations", [])
                    if loc.get("facility", {}).get("name")
                }
            )

            # Conditions + MeSH
            conditions = cond_mod.get("conditions", [])
            mesh_terms = cond_mod.get("keywords", [])  # often empty
            keywords = cond_mod.get("keywords", [])

            brief_summary = desc_mod.get("briefSummary", "")
            detailed_description = desc_mod.get("detailedDescription", "")
            eligibility_criteria = eligibility_mod.get("eligibilityCriteria", "")

            # Detect EEG relevance
            full_text = " ".join(
                [
                    brief_summary,
                    detailed_description,
                    " ".join(conditions),
                    eligibility_criteria,
                    " ".join(primary_outcomes),
                ]
            ).lower()

            import re
            eeg_methods: List[str] = []
            for kw in self.EEG_METHOD_KEYWORDS:
                if re.search(kw, full_text, re.IGNORECASE):
                    eeg_methods.append(kw)

            return ClinicalTrial(
                nct_id=nct_id,
                title=id_mod.get("briefTitle", ""),
                brief_summary=brief_summary,
                detailed_description=detailed_description,
                conditions=conditions,
                interventions=interventions,
                status=status_mod.get("overallStatus", "UNKNOWN"),
                phase=design_mod.get("phases", [None])[0]
                if design_mod.get("phases")
                else None,
                study_type=design_mod.get("studyType", ""),
                start_date=start_date,
                completion_date=completion_date,
                enrollment=enrollment,
                sponsors=[s for s in sponsors if s],
                locations=locations,
                primary_outcomes=primary_outcomes,
                secondary_outcomes=secondary_outcomes,
                eligibility_criteria=eligibility_criteria,
                mesh_terms=mesh_terms,
                keywords=keywords,
                eeg_relevant=bool(eeg_methods),
                eeg_methods_mentioned=eeg_methods,
            )

        except Exception as exc:
            logger.warning("Failed to parse study record: %s", exc)
            return None

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
        """Parse date strings like 'January 2021' or '2021-01-15'."""
        if not date_str:
            return None
        for fmt in ("%Y-%m-%d", "%B %Y", "%Y"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None

    async def _get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Perform a GET request and return parsed JSON."""
        if not self._session:
            raise RuntimeError("Client must be used as async context manager")
        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json(content_type=None)
                logger.warning(
                    "ClinicalTrials.gov HTTP %d for %s", resp.status, url
                )
                return None
        except Exception as exc:
            logger.error("ClinicalTrials.gov request error: %s", exc)
            return None
