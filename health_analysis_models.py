from datetime import date
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


MAX_RECORDS = 60
MAX_SYMPTOMS_PER_RECORD = 20
MAX_STRING_LENGTH = 200
MAX_ARRAY_ITEMS = 20
OBSERVATION_FORBIDDEN_PHRASES = (
    "未提供",
    "没有提供",
    "缺少",
    "数据不足",
    "无法判断",
    "不能判断",
    "不能确定",
    "尚未记录",
    "记录不完整",
    "软件没有",
    "字段",
    "经量等级定义",
    "症状严重程度缺失",
    "完整出血天数缺失",
)
LimitedText = Annotated[str, StringConstraints(min_length=1, max_length=MAX_STRING_LENGTH)]
ResultText = Annotated[str, StringConstraints(min_length=1, max_length=2000)]


class PeriodRecordInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: date
    blood: int = Field(ge=0, le=5)
    color: str = Field(default="", max_length=MAX_STRING_LENGTH)
    symptoms: list[LimitedText] = Field(default_factory=list, max_length=MAX_SYMPTOMS_PER_RECORD)

    @field_validator("symptoms")
    @classmethod
    def validate_symptoms(cls, value: list[str]) -> list[str]:
        if any(not item.strip() or len(item) > MAX_STRING_LENGTH for item in value):
            raise ValueError("症状内容不能为空且长度不能超过限制")
        return value


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    regularity_score: int = Field(ge=0, le=100)
    trend_description: str = Field(default="", max_length=MAX_STRING_LENGTH)
    warnings: list[LimitedText] = Field(default_factory=list, max_length=MAX_ARRAY_ITEMS)
    enough_data: bool
    confidence: str = Field(default="", max_length=MAX_STRING_LENGTH)
    predicted_start: date | None = None
    low_bound: date | None = None
    high_bound: date | None = None
    predicted_ovulation: date | None = None


class HealthAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int | None = Field(default=None, ge=10, le=100)
    phase: str | None = Field(default=None, max_length=MAX_STRING_LENGTH)
    records: list[PeriodRecordInput] = Field(default_factory=list, max_length=MAX_RECORDS)
    prediction: PredictionInput | None = None
    pet_name: str = Field(default="露娜", max_length=32)
    care_feedback: list[str] = Field(default_factory=list, max_length=20)


class HealthAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: ResultText
    health_observations: list[ResultText] = Field(max_length=MAX_ARRAY_ITEMS)
    food_suggestions: list[ResultText] = Field(max_length=MAX_ARRAY_ITEMS)
    things_to_do: list[ResultText] = Field(max_length=MAX_ARRAY_ITEMS)
    things_to_limit: list[ResultText] = Field(max_length=MAX_ARRAY_ITEMS)
    symptom_care: list[ResultText] = Field(max_length=MAX_ARRAY_ITEMS)
    attention_points: list[ResultText] = Field(max_length=MAX_ARRAY_ITEMS)
    medical_red_flags: list[ResultText] = Field(max_length=MAX_ARRAY_ITEMS)
    cycle_overview: ResultText
    suggestions: list[ResultText] = Field(max_length=MAX_ARRAY_ITEMS)
    medical_notice: str = Field(min_length=1, max_length=1000)
    data_limitations: list[ResultText] = Field(max_length=2)

    @field_validator("health_observations")
    @classmethod
    def validate_health_observations(cls, value: list[str]) -> list[str]:
        if any(
            phrase in item
            for item in value
            for phrase in OBSERVATION_FORBIDDEN_PHRASES
        ):
            raise ValueError("health_observations 只能包含基于记录的正向观察")
        return value


def request_payload(request: HealthAnalysisRequest) -> dict[str, Any]:
    return request.model_dump(mode="json", exclude_none=True)