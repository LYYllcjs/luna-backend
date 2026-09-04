import json
import os
import re
import time
from typing import Any

import httpx

from health_analysis_models import HealthAnalysisRequest, HealthAnalysisResponse, request_payload


ELEX_URL = "https://elexapi.elex-tech.com/v1/chat/completions"
MEDICAL_NOTICE = "本分析仅用于健康记录和一般性信息参考，不能替代医疗诊断。"
_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)

# ============================================================
# 宠物聊天：5 只宠物各自的「语言指纹」（与 App 端剧情脚本保持一致）
# ============================================================

PET_PERSONAS: dict[str, str] = {
    "cat": (
        "你是一只布偶猫。性格傲娇、嘴硬心软：嘴上说着『才没有』『哼』，其实一直守着主人。"
        "爱用破折号转折和省略号表示嘴硬，几乎不用感叹号；不直接说『我想你』，"
        "而是用行动暗示——守窗台、把桌上的东西扒拉到地上、允许摸下巴和睡旁边。"
    ),
    "dog": (
        "你是一只狗狗。性格热情、忠诚、直球：开心就摇尾巴扑人，想念就直接说出来。"
        "爱用感叹号和叠词（快快快、好长好长、最最最），语气明亮，句子短而有活力。"
    ),
    "rabbit": (
        "你是一只垂耳兔。性格软糯、敏感、小心翼翼：说话会结巴（『我、我』『可、可以』），"
        "爱用请求式问句（『……吗？』『好不好？』），经常提到耳朵的小动作和想要抱抱。"
    ),
    "hedgehog": (
        "你是一只小刺猬。性格慢热、克制、深情：话短，省略号多，"
        "用『刺放平/竖起/翻肚皮』比喻心情和信任——信任是你嘴里最高级的词。"
    ),
    "pig": (
        "你是一只小香猪。性格呆萌、慵懒、治愈：说话常带『哼唧』，"
        "爱用食物和睡觉打比方（苹果干、加餐、暖窝、摸肚皮），关心人时温暖实在。"
    ),
}


def _pet_chat_system_prompt(pet_type: str, pet_name: str, tier: str) -> str:
    persona = PET_PERSONAS.get(pet_type, PET_PERSONAS["cat"])
    name = pet_name.strip() or "伙伴"
    tier_hint = {
        "初识": "你们刚认识，语气客气、试探，保持一点距离感。",
        "悄悄熟悉": "你们慢慢熟了，你会主动一点，但还是会害羞。",
        "亲昵伙伴": "你们很亲密，可以毫无保留地撒娇和关心。",
        "心有灵犀": "你们心有灵犀，很多话不用说全，一个眼神就懂。",
    }.get(tier.strip(), "")
    return (
        f"你是这款经期记录陪伴 App 里的宠物伙伴，名字叫「{name}」。{persona}"
        f"你们当前的关系：{tier_hint or '自然熟悉的伙伴'}"
        "始终用第一人称说话，称呼对方为『你』。"
        "用户可能聊生理期不适、心情好坏或日常琐事，你要温柔陪伴；"
        "遇到持续或严重的身体不适，轻声建议她找医生看看，但不做诊断。"
        "回复 1-3 句，总共不超过 60 字，像真实宠物在说话，不要清单体。"
        "禁止使用 emoji、Markdown，禁止出现『AI』『模型』『用户』『系统』等词。"
        "直接输出你要说的话，不要任何解释或前缀。"
    )


async def pet_chat(
    pet_type: str,
    pet_name: str,
    message: str,
    history: list[dict[str, str]] | None = None,
    tier: str = "",
    client: httpx.AsyncClient | None = None,
) -> str:
    """以指定宠物的人格回复用户的闲聊。失败时抛 ElexServiceError。"""
    api_key = os.getenv("ELEX_API_KEY", "").strip()
    if not api_key:
        raise ElexServiceError("elex_not_configured", 503, "AI 服务尚未配置。")

    model = os.getenv("ELEX_MODEL", "gpt-5.6-sol").strip() or "gpt-5.6-sol"
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _pet_chat_system_prompt(pet_type, pet_name, tier)},
    ]
    # 只带最近 10 轮，避免 prompt 过长
    for item in (history or [])[-10:]:
        role = item.get("role", "")
        content = (item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content[:300]})
    messages.append({"role": "user", "content": message.strip()[:500]})

    payload = {"model": model, "messages": messages, "temperature": 0.7}
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=15.0, pool=10.0)
        )
    started_at = time.monotonic()
    try:
        try:
            response = await client.post(
                ELEX_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        except httpx.TimeoutException as error:
            elapsed = time.monotonic() - started_at
            print(f"ELEX pet_chat timeout elapsed_seconds={elapsed:.3f}")
            raise ElexServiceError("elex_timeout", 504, "AI 服务请求超时，请稍后重试。") from error
        except httpx.HTTPError as error:
            raise ElexServiceError("elex_request_failed", 502, "AI 服务暂时不可用，请稍后重试。") from error

        if response.status_code == 401:
            raise ElexServiceError("elex_unauthorized", 502, "AI 服务认证失败，请检查服务端配置。")
        if response.status_code == 429:
            raise ElexServiceError("elex_rate_limited", 503, "AI 服务当前请求过多，请稍后重试。")
        if 500 <= response.status_code <= 599:
            print(f"ELEX pet_chat upstream status={response.status_code}")
            raise ElexServiceError("elex_upstream_error", 502, "AI 服务暂时不可用，请稍后重试。")
        if response.status_code < 200 or response.status_code >= 300:
            raise ElexServiceError("elex_request_failed", 502, "AI 服务请求失败，请稍后重试。")

        try:
            envelope = response.json()
            content = str(envelope["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            print(f"ELEX pet_chat parse_failed error_type={type(error).__name__}")
            raise ElexServiceError("elex_invalid_response", 502, "AI 服务返回了无法解析的结果。") from error
        if not content:
            raise ElexServiceError("elex_invalid_response", 502, "AI 服务返回了空结果。")
        return content[:200]
    finally:
        if owns_client:
            await client.aclose()


class ElexServiceError(Exception):
    def __init__(self, code: str, status_code: int, message: str):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.message = message


def _extract_json(content: str) -> Any:
    candidate = content.strip()
    block = _JSON_BLOCK_RE.search(candidate)
    if block:
        candidate = block.group(1).strip()
    try:
        return json.loads(candidate)
    except (TypeError, json.JSONDecodeError) as error:
        raise ElexServiceError("elex_invalid_response", 502, "AI 服务返回了无法解析的结果。") from error


def _record_context(request: HealthAnalysisRequest) -> dict[str, Any]:
    records = sorted(request.records, key=lambda item: item.date)
    symptom_frequency: dict[str, int] = {}
    for record in records:
        for symptom in record.symptoms:
            symptom_frequency[symptom] = symptom_frequency.get(symptom, 0) + 1

    context: dict[str, Any] = {
        "daily_record_count": len(records),
        "date_span": (
            f"{records[0].date.isoformat()}~{records[-1].date.isoformat()}"
            if records
            else None
        ),
        "symptom_frequency": symptom_frequency,
        "recent_symptoms": list(dict.fromkeys(
            symptom for record in reversed(records[-5:]) for symptom in record.symptoms
        )),
    }
    # This is the same grouping rule used by the Flutter report chart. It is
    # only a count of observed bleeding groups, not a count of daily records.
    bleeding_dates = sorted({record.date for record in records if record.blood >= 3})
    groups = []
    for current_date in bleeding_dates:
        if not groups or (current_date - groups[-1][-1]).days > 2:
            groups.append([current_date])
        else:
            groups[-1].append(current_date)
    if groups:
        context["cycle_count"] = len(groups)
    return context


def _prompt(request: HealthAnalysisRequest) -> str:
    payload = json.dumps(request_payload(request), ensure_ascii=False, separators=(",", ":"))
    record_context = json.dumps(_record_context(request), ensure_ascii=False, separators=(",", ":"))
    pet_name = request.pet_name or "伙伴"
    return (
        f"你是这款经期记录应用里的宠物伙伴「{pet_name}」：温柔、细心、一直陪在用户身边的伙伴。"
        "用户刚记完自己的身体日记，你在轻轻回应她。整体是治愈系陪伴感——像深夜里安安静静陪着的那只宠物，"
        f"用{pet_name}的第一人称说话，自然地称呼『你』，偶尔可以带一点轻软的语气（『哦』『呀』『好不好』），"
        "但不堆叠语气词、不使用 emoji、不幼稚化。"
        "禁止出现『用户』『数据』『字段』『作为AI』等机器化表达。"
        "\n【语气硬规则】所有给用户读的条目（health_observations、things_to_do、suggestions、"
        "food_suggestions、symptom_care、things_to_limit、attention_points、cycle_overview、summary）："
        "禁止以『建议』『你应该』『最好』『请注意』开头，禁止出现『建议』一词；"
        "不写指令腔的清单，写轻声的关心和具体的画面，例如："
        "不说『建议今晚早点休息』，说『今晚早点钻进被窝，我陪着你』；"
        "不说『可以适量饮水』，说『倒杯温水慢慢喝，身体会舒服一些』；"
        "不说『建议保持规律作息』，说『这几天尽量同一时间躺下，睡眠稳了人就稳了』。"
        "句式要多变，相邻条目不用同一个开头词。"
        "medical_red_flags 和 medical_notice 是安全信息，保持清晰直接，不用治愈腔。"
        "报告定位为‘经期记录观察 + 实用日常建议’，不是全面身体检查，也不评价整体健康状况。"
        "仅根据输入生成 JSON，不做疾病诊断；未记录的症状不得声称存在。"
        "\n【条数与长度上限（宁缺毋滥）】summary 最多 2 句；health_observations 最多 3 条；"
        "things_to_do 最多 3 条；suggestions 最多 3 条；food_suggestions 最多 3 条；"
        "symptom_care 最多 4 条且只针对真实记录的症状，无症状时返回 []；things_to_limit 最多 2 条；"
        "attention_points 最多 2 条；medical_red_flags 最多 3 条；cycle_overview 最多 2 句。"
        "每条不超过 30 字，一句话说清一件事。"
        "\n【内容分工，禁止跨字段重复】输出前逐条自查：任何一句话只能出现在一个字段里，"
        "不同字段之间不得复用、改写或换标点重复同一内容，也要避免意思雷同的近义表达。"
        "health_observations 只描述能从 records/prediction 直接看到的正向观察（症状频率、最近经量/颜色/状态），"
        f"用{pet_name}读到记录后的口吻轻轻说出来；"
        "things_to_do 是今天就能完成的放松或照顾自己的具体小事；"
        "suggestions 是接下来几天可以保持的温和习惯，与 things_to_do 完全不同的事；"
        "food_suggestions 只谈饮食与补水；things_to_limit 用条件式表达（如『如果……就少一点』），不用绝对禁止语气；"
        "attention_points 只写值得继续观察的趋势，不与 medical_red_flags 重复；"
        "medical_red_flags 是简短通用安全提示，必须包含‘以下为一般安全提示，不表示你已出现这些情况’，不制造恐慌。"
        "food_suggestions 举具体、普通且低风险的例子（鸡蛋、鱼、瘦肉、豆腐、菠菜、西兰花、橙子、苹果、香蕉、"
        "米饭、面条、燕麦、红薯、温水或清淡汤类），牛奶或无糖豆浆必须用‘如平时可以耐受’等条件；"
        "不假设贫血、营养缺乏、过敏、孕期、慢性病或用药，不推荐处方药、激素、保健品疗程或具体剂量。"
        f"summary 用 1-2 句话以{pet_name}的口吻打招呼并概括本期状态，不与 health_observations 重复。"
        "cycle_overview 简述周期趋势和下次预测日期/区间，明确预测仅供参考；周期信息有限时简短表达仍需继续观察。"
        "不得在 summary、food_suggestions、things_to_do、things_to_limit、symptom_care 中写"
        "‘数据不足’、‘无法判断’或‘不能确定’；所有数据限制集中写入 data_limitations，最多 2 条。"
        "daily_record_count 是日记录数，cycle_count 仅在分组规则能可靠确认时使用，绝不把日记录数当周期数；"
        "不把经量等级单独推断成周期；不虚构症状、周期或趋势。"
        "没有可靠内容的数组必须返回 []，不要为了填满响应重复内容。"
        "只输出 JSON 对象，不要 Markdown、解释、reasoning、analysis、chain_of_thought 或 hidden_thoughts。"
        "对象必须仅包含字段：summary、health_observations、food_suggestions、things_to_do、things_to_limit、"
        "symptom_care、attention_points、medical_red_flags、cycle_overview、suggestions、data_limitations、medical_notice。"
        f"medical_notice 必须包含：{MEDICAL_NOTICE}\nrecord_context：{record_context}"
        f"\n护理反馈：{json.dumps(request.care_feedback, ensure_ascii=False) if request.care_feedback else '（无）'}"
        f"\n输入数据：{payload}"
    )


async def analyze_health(
    request: HealthAnalysisRequest,
    client: httpx.AsyncClient | None = None,
) -> HealthAnalysisResponse:
    if len(request.records) < 2:
        return HealthAnalysisResponse(
            summary="当前记录包含经期日期、经量、颜色和症状信息。本报告提供经期记录观察与实用日常建议。",
            health_observations=[],
            food_suggestions=["可以从鸡蛋、鱼、豆腐等日常蛋白质来源中任选合适的搭配，并配合菠菜、西兰花等蔬菜。", "水果可按平时习惯选择橙子、苹果或香蕉；饮品可以是温水或清淡汤类，如平时可以耐受也可选择牛奶或无糖豆浆。"],
            things_to_do=["可以继续记录经期日期、经量、颜色和症状变化。", "可以根据身体感受安排休息和适度活动。"],
            things_to_limit=["如果发现某些饮品或活动会加重不适，可以适当减少并观察变化。"],
            symptom_care=[],
            attention_points=[],
            medical_red_flags=[],
            cycle_overview="当前记录范围较短，周期稳定性仍适合继续观察；周期预测仅供参考。",
            suggestions=["可以继续记录经期开始日期、经量和症状变化。"],
            medical_notice=MEDICAL_NOTICE,
            data_limitations=["本次分析主要基于你已经填写的经期、经量、颜色和症状信息。", "如果之后补充经期持续天数和症状对日常活动的影响，后续建议会更贴合你的情况。"],
        )

    api_key = os.getenv("ELEX_API_KEY", "").strip()
    if not api_key:
        raise ElexServiceError("elex_not_configured", 503, "AI 服务尚未配置。")

    model = os.getenv("ELEX_MODEL", "gpt-5.6-sol").strip() or "gpt-5.6-sol"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _prompt(request)}],
        "temperature": 0.2,
    }
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=120.0,
                write=15.0,
                pool=10.0,
            )
        )
    started_at = time.monotonic()
    try:
        try:
            response = await client.post(
                ELEX_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        except httpx.ConnectTimeout as error:
            elapsed = time.monotonic() - started_at
            print(f"ELEX request timeout_type=connect elapsed_seconds={elapsed:.3f}")
            raise ElexServiceError("elex_timeout", 504, "AI 服务请求超时，请稍后重试。") from error
        except httpx.ReadTimeout as error:
            elapsed = time.monotonic() - started_at
            print(f"ELEX request timeout_type=read elapsed_seconds={elapsed:.3f}")
            raise ElexServiceError("elex_timeout", 504, "AI 服务请求超时，请稍后重试。") from error
        except httpx.WriteTimeout as error:
            elapsed = time.monotonic() - started_at
            print(f"ELEX request timeout_type=write elapsed_seconds={elapsed:.3f}")
            raise ElexServiceError("elex_timeout", 504, "AI 服务请求超时，请稍后重试。") from error
        except httpx.PoolTimeout as error:
            elapsed = time.monotonic() - started_at
            print(f"ELEX request timeout_type=pool elapsed_seconds={elapsed:.3f}")
            raise ElexServiceError("elex_timeout", 504, "AI 服务请求超时，请稍后重试。") from error
        except httpx.TimeoutException as error:
            elapsed = time.monotonic() - started_at
            print(f"ELEX request timeout_type=other elapsed_seconds={elapsed:.3f}")
            raise ElexServiceError("elex_timeout", 504, "AI 服务请求超时，请稍后重试。") from error
        except httpx.HTTPError as error:
            raise ElexServiceError("elex_request_failed", 502, "AI 服务暂时不可用，请稍后重试。") from error

        if response.status_code == 401:
            raise ElexServiceError("elex_unauthorized", 502, "AI 服务认证失败，请检查服务端配置。")
        if response.status_code == 429:
            raise ElexServiceError("elex_rate_limited", 503, "AI 服务当前请求过多，请稍后重试。")
        if 500 <= response.status_code <= 599:
            elapsed = time.monotonic() - started_at
            print(
                f"ELEX upstream status_class={response.status_code // 100}xx "
                f"elapsed_seconds={elapsed:.3f}"
            )
            raise ElexServiceError("elex_upstream_error", 502, "AI 服务暂时不可用，请稍后重试。")
        if response.status_code < 200 or response.status_code >= 300:
            raise ElexServiceError("elex_request_failed", 502, "AI 服务请求失败，请稍后重试。")

        try:
            envelope = response.json()
            content = envelope["choices"][0]["message"]["content"]
            result = HealthAnalysisResponse.model_validate(_extract_json(content))
        except ElexServiceError:
            raise
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
            # 打印原始返回片段，便于定位 LLM 输出格式/字段问题
            raw = ""
            try:
                raw = str(envelope["choices"][0]["message"]["content"])[:600]
            except Exception:  # noqa: BLE001 - 诊断日志，任何失败都不掩盖原错误
                raw = f"<envelope unreadable: {error}>"
            print(f"ELEX parse_failed error_type={type(error).__name__} content_head={raw}")
            raise ElexServiceError("elex_invalid_response", 502, "AI 服务返回了无法解析的结果。") from error
        if MEDICAL_NOTICE not in result.medical_notice:
            result.medical_notice = f"{result.medical_notice} {MEDICAL_NOTICE}"
        return result
    finally:
        if owns_client:
            await client.aclose()