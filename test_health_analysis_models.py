import pytest
from pydantic import ValidationError

from health_analysis_models import HealthAnalysisResponse
from elex_service import _prompt, analyze_health
from health_analysis_models import HealthAnalysisRequest


def complete_response():
    return {
        "summary": "记录显示近期整体状态可继续观察。",
        "health_observations": ["基于已提交记录的观察。"],
        "food_suggestions": ["适量饮水，并均衡摄入蛋白质和蔬菜。"],
        "things_to_do": ["规律休息并进行适度活动。"],
        "things_to_limit": ["尽量减少熬夜和过度劳累。"],
        "symptom_care": ["针对已记录的不适，可先休息并观察变化。"],
        "attention_points": ["继续观察症状是否持续或加重。"],
        "medical_red_flags": ["如疼痛严重、持续加重或影响活动，请咨询医生。"],
        "cycle_overview": "周期暂时可参考现有记录，下次预测仅供参考。",
        "suggestions": ["继续保持记录。"],
        "data_limitations": [],
        "medical_notice": "本分析仅用于健康记录和一般性信息参考，不能替代医疗诊断。",
    }


def test_all_new_fields_validate():
    result = HealthAnalysisResponse.model_validate(complete_response())
    assert result.health_observations == ["基于已提交记录的观察。"]
    assert result.data_limitations == []


def test_missing_field_and_wrong_array_type_rejected():
    missing = complete_response()
    del missing["cycle_overview"]
    with pytest.raises(ValidationError):
        HealthAnalysisResponse.model_validate(missing)
    wrong = complete_response()
    wrong["food_suggestions"] = "适量饮水"
    with pytest.raises(ValidationError):
        HealthAnalysisResponse.model_validate(wrong)


@pytest.mark.parametrize(
    "observation",
    [
        "未提供完整出血天数。",
        "缺少症状严重程度字段。",
        "数据不足，无法判断变化。",
        "尚未记录完整出血天数。",
    ],
)
def test_health_observations_reject_data_limitation_language(observation):
    response = complete_response()
    response["health_observations"] = [observation]
    with pytest.raises(ValidationError):
        HealthAnalysisResponse.model_validate(response)


def test_health_observations_accept_empty_array():
    response = complete_response()
    response["health_observations"] = []
    result = HealthAnalysisResponse.model_validate(response)
    assert result.health_observations == []


def test_prompt_separates_observations_from_limitations():
    prompt = _prompt(HealthAnalysisRequest(records=[]))
    assert "只写正向观察" in prompt
    assert "若没有可靠正向观察必须返回 []" in prompt
    assert "症状严重程度缺失" in prompt


def test_insufficient_data_response_has_all_fields():
    import asyncio

    request = HealthAnalysisRequest(records=[])
    result = asyncio.run(analyze_health(request))
    assert result.data_limitations
    assert result.health_observations == []
    assert result.food_suggestions
    assert result.things_to_do
    assert "已经填写" in result.data_limitations[0]
    assert result.cycle_overview.endswith("仅供参考。")