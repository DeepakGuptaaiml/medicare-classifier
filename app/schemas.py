from pydantic import BaseModel, Field


class ClaimFeatures(BaseModel):
    data_set: str = Field(..., description="Dataset code (WC, GL, AU, PF)")
    pay_cat: str = Field(..., description="Payment category")
    pay_code: int = Field(..., ge=0, description="Payment code")
    pay_type: str = Field(..., description="Payment type (SYS, MAN, HIS)")
    paid_1: float = Field(..., ge=0, description="Indemnity paid amount")
    paid_3: float = Field(..., ge=0, description="Medical paid amount")
    amount: float = Field(..., gt=0, description="Payment amount")
    proc_unit: int = Field(..., ge=0, description="Processing unit")
    cont_num: int = Field(..., ge=0, description="Contract number")
    claim_open: int = Field(..., ge=0, le=1, description="1 if claim open, 0 if closed")
    date_v1m_xmit_flag: int = Field(0, ge=0, le=1, description="1 if already transmitted to V1M")
    is_us_claimant: int = Field(1, ge=0, le=1, description="1 if US claimant")
    orm_threshold_met: int = Field(0, ge=0, le=1, description="1 if paid_3 > 750")
    tpoc_threshold_met: int = Field(0, ge=0, le=1, description="1 if paid_1 > 0")
    is_wc: int = Field(0, ge=0, le=1, description="1 if workers comp claim")
    pay_code_bucket: str = Field(..., description="TPOC, ORM, or SETTLEMENT")
    is_excluded_coverage: int = Field(0, ge=0, le=1)
    is_excluded_line: int = Field(0, ge=0, le=1)
    days_open: float = Field(0.0, ge=0, description="Days claim was open")
    age_at_event: float = Field(..., ge=0, le=120, description="Claimant age at date of event")


class PredictionResponse(BaseModel):
    is_medicare_reportable: int
    probability: float
    label: str
    model_name: str
    target: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None


class ModelInfoResponse(BaseModel):
    model_name: str
    sampling_strategy: str
    target: str
    feature_count: int
    metrics_test: dict


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Medicare policy question")
    max_chunks: int = Field(3, ge=1, le=10, description="Number of context chunks to retrieve")


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    chunks_used: list[str]
    model_used: str
    processing_time_ms: float


class RAGStatusResponse(BaseModel):
    status: str
    documents_loaded: int
    vector_store_ready: bool
    llm_ready: bool
