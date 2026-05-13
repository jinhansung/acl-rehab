"""
Pydantic v2 data models for ACL Rehab Companion.

Invariants enforced here:
- No Optional field on a required clinical attribute — use explicit enums instead.
- JournalEntry carries only ciphertext; plaintext never appears in any model.
- ConsentRecord.data_sent_hash must be set before a RehabPlan references it.
"""
import hashlib
from datetime import date as Date, datetime
from enum import Enum
from typing import Annotated, Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── Enumerations ──────────────────────────────────────────────────────────────

class GraftType(str, Enum):
    PATELLAR_TENDON = "patellar_tendon"
    HAMSTRING = "hamstring"
    QUADRICEPS = "quadriceps"
    ALLOGRAFT = "allograft"
    OTHER = "other"

    def display(self) -> str:
        return self.value.replace("_", " ").title()


class WeightBearingStatus(str, Enum):
    NON_WEIGHT_BEARING = "non_weight_bearing"
    TOUCH_DOWN = "touch_down"
    PARTIAL = "partial"
    FULL_WITH_CRUTCHES = "full_with_crutches"
    FULL = "full"

    def display(self) -> str:
        return self.value.replace("_", " ").title()


class MeniscalRepair(str, Enum):
    NONE = "none"
    MEDIAL = "medial"
    LATERAL = "lateral"
    BOTH = "both"

    def display(self) -> str:
        return self.value.title()


class Protocol(str, Enum):
    MOON = "MOON"
    DELAWARE_OSLO = "Delaware-Oslo"
    ASPETAR = "Aspetar"


class SwellingLevel(str, Enum):
    NONE = "None"
    MILD = "Mild"
    MODERATE = "Moderate"
    SEVERE = "Severe"


class ConsentType(str, Enum):
    PLAN_GENERATION = "plan_generation"
    WEEKLY_SUMMARY = "weekly_summary"


# ── Core models ───────────────────────────────────────────────────────────────

class PatientProfile(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: Optional[int] = None
    # name is stored locally only; never passed to any API
    name: str = Field(min_length=1, max_length=120)
    side: str = Field(pattern=r"^(Left|Right)$")
    graft_type: GraftType
    surgery_date: Date
    weight_bearing_status: WeightBearingStatus
    meniscal_repair: MeniscalRepair
    # Patient's own words — stored verbatim, never anonymized away
    stated_goal_text: str = Field(min_length=5, max_length=1000)
    protocol: Protocol
    pt_code: Optional[str] = Field(default=None, max_length=64)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("surgery_date")
    @classmethod
    def surgery_date_not_future(cls, v: Date) -> Date:
        if v > Date.today():
            raise ValueError("Surgery date cannot be in the future.")
        return v

    @property
    def weeks_post_op(self) -> int:
        return max(1, (Date.today() - self.surgery_date).days // 7 + 1)


class BaselineMeasurements(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: Optional[int] = None
    patient_id: int
    measured_at: datetime = Field(default_factory=datetime.utcnow)
    # ROM in degrees; None = not yet measured
    knee_flexion_rom: Optional[float] = Field(default=None, ge=0.0, le=160.0)
    knee_extension_rom: Optional[float] = Field(default=None, ge=-30.0, le=20.0)
    # Strength in Newton-metres
    quad_strength_nm: Optional[float] = Field(default=None, ge=0.0)
    hamstring_strength_nm: Optional[float] = Field(default=None, ge=0.0)
    # Single-leg squat depth in cm
    single_leg_squat_depth_cm: Optional[float] = Field(default=None, ge=0.0)
    pain_at_rest: int = Field(ge=0, le=10)
    pain_with_activity: int = Field(ge=0, le=10)
    notes: str = ""


class ConsentRecord(BaseModel):
    """Records explicit patient consent before any cloud API call is made."""
    model_config = ConfigDict(use_enum_values=True)

    id: Optional[int] = None
    patient_id: int
    consented_at: datetime = Field(default_factory=datetime.utcnow)
    consent_type: ConsentType
    model_used: str
    # SHA-256 of the anonymised payload actually sent — audit trail
    data_sent_hash: str = Field(min_length=64, max_length=64)
    revoked_at: Optional[datetime] = None

    @classmethod
    def make_hash(cls, payload: str) -> str:
        return hashlib.sha256(payload.encode()).hexdigest()

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None


class PlanReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class RehabPlan(BaseModel):
    """Generated exercise plan — only created after ConsentRecord exists."""
    model_config = ConfigDict(use_enum_values=True)

    id: Optional[int] = None
    patient_id: int
    consent_record_id: int          # FK — plan without consent is invalid
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    protocol: Protocol
    week_start: int = Field(ge=1)
    week_end: int = Field(ge=1)
    # List of {name, sets, reps, cues, rationale, rag_source_id, rag_excerpt} dicts
    exercises: list[dict[str, Any]] = Field(default_factory=list)
    model_used: str
    rag_sources: list[str] = Field(default_factory=list)   # chunk IDs cited
    week_summary: str = ""                                  # patient-facing after PT edits
    pt_flag_notes: str = ""                                 # clinical concerns for PT
    goal_protocol_conflicts: list[dict[str, Any]] = Field(default_factory=list)
    # PT review fields
    review_status: PlanReviewStatus = PlanReviewStatus.PENDING
    reviewed_at: Optional[datetime] = None
    pt_review_notes: str = ""

    @model_validator(mode="after")
    def week_range_valid(self) -> "RehabPlan":
        if self.week_end < self.week_start:
            raise ValueError("week_end must be >= week_start")
        return self


class SessionRecord(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: Optional[int] = None
    patient_id: int
    date: Date = Field(default_factory=Date.today)
    week_number: int = Field(ge=1)
    pain_score: int = Field(ge=0, le=10)
    swelling: SwellingLevel
    giving_way: bool
    exercises_completed: list[str] = Field(default_factory=list)
    exercises_skipped: list[str] = Field(default_factory=list)
    session_notes: str = ""
    duration_minutes: Optional[Annotated[int, Field(ge=1, le=480)]] = None

    def is_this_week(self) -> bool:
        return (Date.today() - self.date).days < 7


class Measurement(BaseModel):
    """Single time-series data point for any tracked metric."""
    model_config = ConfigDict(use_enum_values=True)

    id: Optional[int] = None
    patient_id: int
    session_id: Optional[int] = None
    measured_at: datetime = Field(default_factory=datetime.utcnow)
    metric: str = Field(min_length=1, max_length=80)
    value: float
    unit: str = Field(min_length=1, max_length=20)

    @field_validator("metric")
    @classmethod
    def metric_lowercase(cls, v: str) -> str:
        return v.strip().lower()


class JournalEntry(BaseModel):
    """
    Encrypted journal entry.
    PRIVACY: ciphertext only — plaintext is never stored in any model field.
    No server call is made with journal data. See data/journal.py.
    """
    id: Optional[int] = None
    patient_id: int
    date: Date = Field(default_factory=Date.today)
    # Fernet ciphertext bytes — decrypted only in data/journal.py
    ciphertext: bytes

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RedFlagEvent(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: Optional[int] = None
    patient_id: int
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    # Human-readable flag strings from agent/red_flags.py
    flags: list[str] = Field(default_factory=list)
    pain_score: int = Field(ge=0, le=10)
    swelling: SwellingLevel
    giving_way: bool
    reviewed_by_pt: bool = False
    escalated: bool = False
    resolved_at: Optional[datetime] = None

    @field_validator("flags")
    @classmethod
    def flags_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("RedFlagEvent must have at least one flag.")
        return v

    @property
    def is_open(self) -> bool:
        return not self.reviewed_by_pt and self.resolved_at is None
