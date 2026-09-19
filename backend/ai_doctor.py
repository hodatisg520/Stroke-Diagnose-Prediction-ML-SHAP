"""Optional Gemini coach plus a local fallback for the STROKEGUARD prototype.

The browser sends this endpoint the existing screening result together with a
small health snapshot. The Gemini key stays on the backend. If no key is set,
the same response shape is produced by local rules so the prototype remains
usable during a demo.
"""

from typing import Any, Dict, List, Optional
import json
import math
import os
import urllib.error
import urllib.request

from pydantic import BaseModel, Field


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()


class AIDoctorRequest(BaseModel):
    """Combined model output and wearable/manual context for the prototype coach."""

    patient_profile: Dict[str, Any] = Field(default_factory=dict)
    prediction: Dict[str, Any] = Field(default_factory=dict)
    health_snapshot: Dict[str, Any] = Field(default_factory=dict)
    history: List[Dict[str, Any]] = Field(default_factory=list)
    source: str = "Manual entry"


def _finite_number(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert user/device values to finite numbers without crashing the advice flow."""

    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _normalise_advice_item(item: Any, fields: List[str]) -> Dict[str, str]:
    """Keep Gemini output renderable even if it omits one optional field."""

    if isinstance(item, dict):
        return {field: str(item.get(field, "")).strip() for field in fields}
    return {fields[0]: str(item).strip(), **{field: "" for field in fields[1:]}}


def _normalise_ai_advice(
    payload: Dict[str, Any],
    mode: str,
    source: str,
    notice: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the stable response shape consumed by the React prototype."""

    signals = payload.get("risk_signals", [])
    recommendations = payload.get("recommendations", [])
    normalised_signals = []
    for signal in signals if isinstance(signals, list) else []:
        item = _normalise_advice_item(signal, ["title", "severity", "message"])
        if item["severity"] not in {"info", "watch", "urgent"}:
            item["severity"] = "info"
        normalised_signals.append(item)

    return {
        "mode": mode,
        "source": source,
        "summary": str(
            payload.get(
                "summary",
                "Review the latest health information with a qualified clinician.",
            )
        ).strip(),
        "risk_signals": normalised_signals[:4],
        "recommendations": [
            _normalise_advice_item(item, ["title", "action", "why"])
            for item in recommendations[:4]
            if item is not None
        ] if isinstance(recommendations, list) else [],
        "next_steps": [
            str(step).strip()
            for step in (
                payload.get("next_steps", [])
                if isinstance(payload.get("next_steps", []), list)
                else []
            )
            if str(step).strip()
        ][:4],
        "disclaimer": str(
            payload.get(
                "disclaimer",
                "This prototype provides educational risk guidance, not a diagnosis or treatment plan.",
            )
        ).strip(),
        **({"notice": notice} if notice else {}),
    }


def _build_demo_advice(data: AIDoctorRequest, notice: Optional[str] = None) -> Dict[str, Any]:
    """Local rule-based fallback so the prototype works without a Gemini key."""

    snapshot = data.health_snapshot or {}
    profile = data.patient_profile or {}
    prediction = data.prediction or {}
    probability = _finite_number(prediction.get("stroke_probability"), 0.0) or 0.0
    systolic = _finite_number(snapshot.get("systolic_bp"))
    diastolic = _finite_number(snapshot.get("diastolic_bp"))
    sleep_hours = _finite_number(snapshot.get("sleep_hours"))
    steps = _finite_number(snapshot.get("daily_steps"))
    active_minutes = _finite_number(snapshot.get("active_minutes"))
    glucose = _finite_number(profile.get("avg_glucose_level"))
    smoking = str(profile.get("smoking_status", "")).lower()

    signals: List[Dict[str, str]] = []
    recommendations: List[Dict[str, str]] = []

    if (systolic is not None and systolic >= 180) or (diastolic is not None and diastolic >= 120):
        signals.append(
            {
                "title": "Very high blood pressure reading",
                "severity": "urgent",
                "message": "Repeat the measurement after resting. If it remains this high or you have concerning symptoms, seek urgent medical help now.",
            }
        )
    elif (systolic is not None and systolic >= 140) or (diastolic is not None and diastolic >= 90):
        signals.append(
            {
                "title": "Blood pressure deserves attention",
                "severity": "watch",
                "message": "This reading is above the usual home target range. Track repeated readings and discuss the pattern with a clinician.",
            }
        )
        recommendations.append(
            {
                "title": "Create a blood-pressure log",
                "action": "Measure at a consistent time after five minutes of rest and record several readings instead of relying on one value.",
                "why": "A trend is more useful than a single measurement for a clinical conversation.",
            }
        )

    if sleep_hours is not None and sleep_hours < 6:
        signals.append(
            {
                "title": "Sleep is running short",
                "severity": "watch",
                "message": f"The latest snapshot shows about {sleep_hours:.1f} hours. Short sleep can make healthy routines harder to maintain.",
            }
        )
        recommendations.append(
            {
                "title": "Protect a regular sleep window",
                "action": "Move toward a consistent bedtime and wake time, reducing late caffeine and screen exposure where practical.",
                "why": "More consistent sleep supports recovery and makes activity and blood-pressure routines easier to sustain.",
            }
        )

    if steps is not None and steps < 5000:
        recommendations.append(
            {
                "title": "Add small movement breaks",
                "action": "Try two or three short, comfortable walks or movement breaks today; increase gradually rather than making a sudden intense change.",
                "why": "Small repeatable actions are a safer way to build daily activity than an abrupt workout jump.",
            }
        )

    if active_minutes is not None and active_minutes < 30:
        signals.append(
            {
                "title": "Low activity in this snapshot",
                "severity": "info",
                "message": "The device sample shows limited active time today. Treat this as a coaching signal, not a diagnosis.",
            }
        )

    if glucose is not None and glucose >= 140:
        signals.append(
            {
                "title": "Glucose reading to review",
                "severity": "watch",
                "message": "This value should be interpreted with its measurement context and your clinician's guidance; do not change medication based on this prototype.",
            }
        )

    if "smokes" in smoking:
        recommendations.append(
            {
                "title": "Choose a quit-support step",
                "action": "Consider a quit plan or speak with a healthcare professional about evidence-based support.",
                "why": "Reducing tobacco exposure is one of the most useful modifiable steps for vascular health.",
            }
        )

    if probability >= 0.60:
        risk_summary = "The screening model currently places this profile in its higher-risk band."
    elif probability >= 0.30:
        risk_summary = "The screening model currently places this profile in its moderate-risk band."
    else:
        risk_summary = "The screening model currently places this profile in its lower-risk band."

    if not signals:
        signals.append(
            {
                "title": "No immediate prototype flag",
                "severity": "info",
                "message": "Keep tracking trends and continue routine preventive care; a normal-looking snapshot does not rule out medical problems.",
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "title": "Keep the routine sustainable",
                "action": "Continue regular movement, balanced meals, adequate sleep, and any care plan already agreed with your clinician.",
                "why": "Consistent habits are more valuable than a one-day optimization.",
            }
        )

    payload = {
        "summary": f"{risk_summary} The connected source is {data.source or 'manual entry'}. Use this as a conversation starter, not a diagnosis.",
        "risk_signals": signals,
        "recommendations": recommendations,
        "next_steps": [
            "Review repeated readings rather than reacting to one isolated value.",
            "Bring unusual trends and the model result to a qualified healthcare professional.",
            "If sudden facial drooping, arm weakness, speech difficulty, severe confusion, or another emergency symptom appears, call local emergency services immediately.",
        ],
    }
    return _normalise_ai_advice(payload, "demo_fallback", data.source, notice)


def _build_gemini_prompt(data: AIDoctorRequest) -> str:
    """Create a constrained prompt for the optional Gemini call."""

    context = {
        "source": data.source,
        "patient_profile": data.patient_profile,
        "prediction": data.prediction,
        "latest_health_snapshot": data.health_snapshot,
        "recent_history": data.history[-7:],
    }
    return f"""
You are a cautious preventive-health assistant inside a student research prototype called STROKEGUARD.

Use the supplied screening result and health telemetry only to give general educational guidance. Do not diagnose stroke, do not claim that a probability is a medical diagnosis, and do not prescribe, stop, or change medication. Do not invent measurements. If a supplied value looks potentially urgent, clearly say to repeat it after resting and seek urgent medical help, especially if concerning symptoms are present. Keep recommendations practical, gentle, and suitable for a general adult audience.

Return ONLY valid JSON with this exact shape and no markdown:
{{
  "summary": "one short paragraph",
  "risk_signals": [{{"title": "short title", "severity": "info|watch|urgent", "message": "plain-language explanation"}}],
  "recommendations": [{{"title": "short title", "action": "specific low-risk action", "why": "brief reason"}}],
  "next_steps": ["short next step"],
  "disclaimer": "one sentence explaining that this is not a diagnosis or treatment plan"
}}

Treat source values as potentially simulated demo data. Mention that context when it affects interpretation. Keep the output concise: at most 4 risk signals, 4 recommendations, and 4 next steps.

Data:
{json.dumps(context, ensure_ascii=False, allow_nan=False)}
""".strip()


def _call_gemini(data: AIDoctorRequest) -> Dict[str, Any]:
    """Call Gemini through its REST API and return a structured JSON response."""

    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured; using local demo guidance.")

    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    request_body = {
        "contents": [{"role": "user", "parts": [{"text": _build_gemini_prompt(data)}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 900,
            "responseFormat": {                 "text": {                     "mimeType": "application/json",                     "schema": {                         "type": "object",                         "properties": {                             "summary": {"type": "string"},                             "risk_signals": {                                 "type": "array",                                 "items": {                                     "type": "object",                                     "properties": {                                         "title": {"type": "string"},                                         "severity": {"type": "string", "enum": ["info", "watch", "urgent"]},                                         "message": {"type": "string"}                                     },                                     "required": ["title", "severity", "message"]                                 }                             },                             "recommendations": {                                 "type": "array",                                 "items": {                                     "type": "object",                                     "properties": {                                         "title": {"type": "string"},                                         "action": {"type": "string"},                                         "why": {"type": "string"}                                     },                                     "required": ["title", "action", "why"]                                 }                             },                             "next_steps": {"type": "array", "items": {"type": "string"}},                             "disclaimer": {"type": "string"}                         },                         "required": ["summary", "risk_signals", "recommendations", "next_steps", "disclaimer"]                     }                 }             },
        },
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            response_body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Do not expose the response body or API key to the browser.
        raise RuntimeError(f"Gemini request failed with HTTP {exc.code}.") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError("Gemini could not be reached right now.") from exc

    candidates = response_body.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no candidate response.")
    parts = candidates[0].get("content", {}).get("parts", [])
    raw_text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
    if not raw_text.strip():
        raise RuntimeError("Gemini returned an empty response.")

    # Be tolerant if a model wraps JSON in a code fence despite the instruction.
    clean_text = raw_text.strip().replace("```json", "").replace("```", "").strip()
    start = clean_text.find("{")
    end = clean_text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError("Gemini returned a non-JSON response.")
    return json.loads(clean_text[start : end + 1])


def generate_ai_advice(data: AIDoctorRequest) -> Dict[str, Any]:
    """Prefer Gemini, then fall back safely so demos do not depend on a live key."""

    try:
        payload = _call_gemini(data)
        return _normalise_ai_advice(payload, "gemini", data.source)
    except Exception as exc:  # The fallback is intentional for a prototype demo.
        if "not configured" in str(exc):
            notice = "Gemini is not configured, so this panel is using local demo guidance."
        else:
            notice = f"Gemini request failed ({str(exc)}); the panel is using local demo guidance."
        return _build_demo_advice(data, notice)
