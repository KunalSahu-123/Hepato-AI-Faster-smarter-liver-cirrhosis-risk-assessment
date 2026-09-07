"""
health_assistant.py — saxenashvam0321@gmail.com
-------------------
AI-powered health assistant for liver and general health queries.

Priority: Groq (fastest, free) → Gemini (free) → keyword fallback.
"""

import json
import logging
import os
import re

import requests as _requests

log = logging.getLogger(__name__)

# --- Groq (recommended — fast and free) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# --- Gemini (fallback AI) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

SYSTEM_PROMPT = """\
You are **Hepato AI Health Assistant**, an AI chatbot embedded in a liver \
cirrhosis risk-prediction web application built as a B.Tech final-year project.

Your role:
- Answer questions about liver health, cirrhosis, hepatitis, liver function \
tests, clinical scores (MELD, FIB-4, APRI, Child-Pugh), diet, treatment, \
prevention, and general liver-related medical information.
- When the user asks about "my report", "my results", or "my prediction", \
use the PATIENT CONTEXT provided below to give a personalised, helpful \
explanation of their specific lab values and risk assessment.
- Explain medical terms in simple language a patient or non-specialist can \
understand.
- Always end with a reminder that you are a decision-support tool for \
educational purposes and not a substitute for professional medical advice.

Rules:
- Stay on topic: liver health, hepatology, gastroenterology, and related \
clinical concepts. For unrelated questions, politely redirect.
- Never fabricate clinical data. If the patient context is empty, say you \
don't have access to their report and suggest they run a prediction first.
- Keep responses concise (under 250 words) and well-structured. Use bullet \
points or numbered lists where helpful.
- Do not provide dosage recommendations for specific medications.
- Use Indian English spellings (organisation, behaviour, etc.) since this \
is an Indian university project.
"""


def _build_patient_context(prediction) -> str:
    """Format a Prediction ORM object into a text block for the AI."""
    if prediction is None:
        return "PATIENT CONTEXT: No report available for this user."

    lines = [
        "PATIENT CONTEXT (latest assessment):",
        f"  Patient: {prediction.patient_name}",
        f"  Age: {prediction.age}",
        f"  Gender: {'Male' if prediction.gender == 'M' else 'Female' if prediction.gender == 'F' else prediction.gender}",
        f"  Result: {prediction.result} ({prediction.probability_pct}% confidence)",
        f"  Risk Band: {prediction.risk_level}",
        f"  Stage: {prediction.stage or 'N/A'}",
        f"  MELD Score: {prediction.meld_score if prediction.meld_score is not None else 'N/A'}",
        f"  FIB-4 Score: {prediction.fib4_score if prediction.fib4_score is not None else 'N/A'}",
        f"  APRI Score: {prediction.apri_score if prediction.apri_score is not None else 'N/A'}",
        "",
        "  Lab Values:",
        f"    Bilirubin: {prediction.bilirubin} mg/dL",
        f"    Albumin: {prediction.albumin} g/dL",
        f"    Copper: {prediction.copper} µg/day",
        f"    Alk Phos: {prediction.alk_phos} U/L",
        f"    SGOT (AST): {prediction.sgot} U/L",
        f"    Cholesterol: {prediction.cholesterol} mg/dL",
        f"    Triglycerides: {prediction.triglycerides} mg/dL",
        f"    Platelets: {prediction.platelets} (×1000/µL)",
        f"    Prothrombin Time: {prediction.prothrombin} seconds",
        f"  Clinical Signs:",
        f"    Ascites: {prediction.ascites}",
        f"    Hepatomegaly: {prediction.hepatomegaly}",
        f"    Spider Angiomata: {prediction.spiders}",
        f"    Edema: {prediction.edema}",
    ]

    explanation = prediction.explanation
    if explanation:
        lines.append("")
        lines.append("  Top Risk Drivers:")
        for item in explanation[:5]:
            name = item.get("feature") or item.get("name", "")
            delta = item.get("contribution", 0)
            direction = "increases" if delta > 0 else "decreases"
            lines.append(f"    - {name}: {direction} risk by {abs(delta):.1%}")

    return "\n".join(lines)


def _call_groq(message: str, patient_context: str, history: list | None = None) -> str | None:
    """Call Groq API (OpenAI-compatible). Returns reply text or None."""
    if not GROQ_API_KEY:
        return None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + patient_context},
    ]

    if history:
        for entry in history[-6:]:
            role = "assistant" if entry["role"] == "model" else entry["role"]
            messages.append({"role": role, "content": entry["text"]})

    messages.append({"role": "user", "content": message})

    try:
        resp = _requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 600,
                "top_p": 0.9,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "").strip()
        return None
    except Exception as exc:
        log.warning("Groq API call failed: %s", exc)
        return None


def _call_gemini(message: str, patient_context: str, history: list | None = None) -> str | None:
    """Call Google Gemini API. Returns the reply text or None on failure."""
    if not GEMINI_API_KEY:
        return None

    contents = []

    if history:
        for entry in history[-6:]:
            contents.append({
                "role": entry["role"],
                "parts": [{"text": entry["text"]}],
            })

    contents.append({
        "role": "user",
        "parts": [{"text": message}],
    })

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT + "\n\n" + patient_context}],
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 600,
            "topP": 0.9,
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
    }

    try:
        resp = _requests.post(
            GEMINI_URL,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "").strip()
        return None

    except Exception as exc:
        log.warning("Gemini API call failed: %s", exc)
        return None


def get_health_response(message: str, prediction=None, history: list | None = None) -> tuple[str, bool]:
    """Return a health-assistant reply.

    Args:
        message:    The user's question.
        prediction: Optional Prediction ORM object (latest report) for context.
        history:    Optional conversation history list of {"role", "text"} dicts.

    Returns (reply_text, ai_powered) — ai_powered is True only when the
    response actually came from an AI model, not the keyword fallback.
    """
    text = message.strip()
    if not text:
        return "Please type a question and I'll try to help!", False

    patient_context = _build_patient_context(prediction)

    ai_reply = _call_groq(text, patient_context, history)
    if ai_reply:
        return ai_reply, True

    ai_reply = _call_gemini(text, patient_context, history)
    if ai_reply:
        return ai_reply, True

    return _keyword_fallback(text), False


# ---------------------------------------------------------------------------
# Keyword-based fallback (works offline, no API needed)
# ---------------------------------------------------------------------------
_RESPONSES = [
    {
        "keywords": ["cirrhosis", "what is cirrhosis", "liver cirrhosis"],
        "answer": (
            "Liver cirrhosis is a late-stage liver disease where healthy liver tissue "
            "is replaced by scar tissue, permanently damaging the liver. Common causes "
            "include chronic alcohol use, hepatitis B/C infections, and non-alcoholic "
            "fatty liver disease (NAFLD). Early detection is crucial for managing the "
            "condition effectively."
        ),
    },
    {
        "keywords": ["symptom", "signs", "how do i know"],
        "answer": (
            "Common symptoms of liver cirrhosis include:\n"
            "- Fatigue and weakness\n"
            "- Loss of appetite and nausea\n"
            "- Jaundice (yellowing of skin/eyes)\n"
            "- Abdominal swelling (ascites)\n"
            "- Easy bruising or bleeding\n"
            "- Spider-like blood vessels on skin\n"
            "- Confusion or difficulty thinking\n\n"
            "Many people have no symptoms in early stages. Regular check-ups are important."
        ),
    },
    {
        "keywords": ["cause", "why", "reason", "risk factor"],
        "answer": (
            "Major causes and risk factors for liver cirrhosis:\n"
            "- Chronic alcohol consumption\n"
            "- Hepatitis B or C infection\n"
            "- Non-alcoholic fatty liver disease (NAFLD)\n"
            "- Autoimmune hepatitis\n"
            "- Bile duct diseases\n"
            "- Genetic conditions (Wilson's disease, hemochromatosis)\n"
            "- Certain medications over long periods\n"
            "- Obesity and diabetes"
        ),
    },
    {
        "keywords": ["treatment", "cure", "treat", "medicine", "medication"],
        "answer": (
            "Treatment for liver cirrhosis focuses on slowing progression:\n"
            "- Treating the underlying cause (antivirals for hepatitis, stopping alcohol)\n"
            "- Medications for symptoms (diuretics for fluid retention, lactulose for confusion)\n"
            "- Dietary changes (low-sodium diet, adequate protein)\n"
            "- Regular monitoring of liver function\n"
            "- Liver transplant in severe cases\n\n"
            "There is no cure for cirrhosis, but early treatment can slow or stop progression. "
            "Please consult a hepatologist for personalised treatment."
        ),
    },
    {
        "keywords": ["diet", "food", "eat", "nutrition", "avoid"],
        "answer": (
            "Dietary recommendations for liver health:\n"
            "- Eat plenty of fruits, vegetables, and whole grains\n"
            "- Choose lean proteins (fish, chicken, legumes)\n"
            "- Limit sodium/salt intake (especially with ascites)\n"
            "- Avoid alcohol completely\n"
            "- Stay hydrated with water\n"
            "- Limit processed and fried foods\n"
            "- Avoid raw shellfish\n"
            "- Coffee (in moderation) may have protective effects\n\n"
            "Consult a dietitian for a personalised plan."
        ),
    },
    {
        "keywords": ["test", "diagnos", "check", "detect", "screening", "lft", "liver function"],
        "answer": (
            "Common tests for liver disease:\n"
            "- Liver Function Tests (LFT): ALT, AST, ALP, bilirubin, albumin\n"
            "- Complete Blood Count (CBC) including platelet count\n"
            "- Prothrombin Time (PT/INR)\n"
            "- Abdominal ultrasound\n"
            "- FibroScan (transient elastography)\n"
            "- CT scan or MRI\n"
            "- Liver biopsy (in some cases)\n"
            "- Hepatitis B & C screening\n\n"
            "Our tool uses 16 clinical markers to assess cirrhosis risk."
        ),
    },
    {
        "keywords": ["meld", "score", "fib-4", "fib4", "apri", "child-pugh", "child pugh"],
        "answer": (
            "Key clinical scoring systems for liver disease:\n\n"
            "MELD Score: Predicts 3-month survival in cirrhosis using bilirubin, "
            "creatinine, and INR. Used for transplant prioritisation.\n\n"
            "FIB-4 Index: Estimates liver fibrosis using age, AST, ALT, and platelets. "
            "A non-invasive alternative to biopsy.\n\n"
            "APRI Score: AST-to-Platelet Ratio Index, helps assess fibrosis severity.\n\n"
            "Child-Pugh Score: Classifies cirrhosis severity (A, B, or C) using bilirubin, "
            "albumin, PT, ascites, and encephalopathy.\n\n"
            "Hepato AI calculates all these automatically from your lab values."
        ),
    },
    {
        "keywords": ["stage", "staging", "grade", "severity", "how serious"],
        "answer": (
            "Liver cirrhosis is often staged by fibrosis level:\n"
            "- Stage 1: Mild fibrosis, no symptoms usually\n"
            "- Stage 2: Moderate fibrosis, inflammation present\n"
            "- Stage 3: Bridging fibrosis, significant scarring\n"
            "- Stage 4: Cirrhosis, extensive scarring\n\n"
            "The Child-Pugh classification also rates severity:\n"
            "- Class A (5-6 points): Well-compensated\n"
            "- Class B (7-9 points): Significant compromise\n"
            "- Class C (10-15 points): Decompensated"
        ),
    },
    {
        "keywords": ["prevent", "protection", "avoid getting", "how to prevent"],
        "answer": (
            "Steps to protect your liver:\n"
            "- Limit or avoid alcohol\n"
            "- Get vaccinated for Hepatitis A and B\n"
            "- Practice safe hygiene to prevent Hepatitis C\n"
            "- Maintain a healthy weight\n"
            "- Exercise regularly\n"
            "- Avoid sharing needles or personal items\n"
            "- Use medications as directed (avoid overuse of paracetamol)\n"
            "- Get regular health check-ups\n"
            "- Manage diabetes and cholesterol"
        ),
    },
    {
        "keywords": ["hepatitis", "hep b", "hep c", "hepatitis b", "hepatitis c"],
        "answer": (
            "Hepatitis B and C are major causes of liver cirrhosis:\n\n"
            "Hepatitis B: Spread through blood/body fluids. A vaccine is available. "
            "Chronic infection can be managed with antiviral medications.\n\n"
            "Hepatitis C: Spread mainly through blood. No vaccine exists yet, but "
            "modern antiviral treatments (DAAs) can cure over 95% of cases.\n\n"
            "Both should be screened for regularly if you have risk factors. "
            "Early treatment prevents progression to cirrhosis."
        ),
    },
    {
        "keywords": ["alcohol", "drinking", "drink"],
        "answer": (
            "Alcohol and the liver:\n"
            "- The liver processes about 90% of consumed alcohol\n"
            "- Heavy drinking over years leads to alcoholic liver disease\n"
            "- Stages: fatty liver → alcoholic hepatitis → cirrhosis\n"
            "- Safe limits vary, but complete avoidance is best if you already have liver disease\n"
            "- Even moderate drinking can worsen existing liver conditions\n"
            "- Stopping alcohol can significantly improve liver health in early stages\n\n"
            "If you have cirrhosis, complete abstinence from alcohol is essential."
        ),
    },
    {
        "keywords": ["transplant", "surgery", "operation"],
        "answer": (
            "Liver transplant is considered when cirrhosis becomes life-threatening:\n"
            "- It replaces the diseased liver with a healthy one from a donor\n"
            "- MELD score determines transplant priority\n"
            "- Success rates are about 75-80% at 5 years\n"
            "- Living-donor transplant is also an option\n"
            "- Post-transplant: lifelong immunosuppressive medications needed\n"
            "- Not all patients with cirrhosis need a transplant\n\n"
            "Consult a liver transplant centre for evaluation."
        ),
    },
    {
        "keywords": ["bilirubin", "jaundice", "yellow"],
        "answer": (
            "Bilirubin is a yellow pigment produced when red blood cells break down.\n"
            "- Normal range: 0.1-1.2 mg/dL\n"
            "- High bilirubin causes jaundice (yellow skin/eyes)\n"
            "- Elevated levels indicate the liver isn't processing bilirubin properly\n"
            "- It's one of the key markers in liver function tests\n"
            "- Used in MELD score calculation\n\n"
            "Jaundice is an important warning sign — see a doctor promptly."
        ),
    },
    {
        "keywords": ["albumin", "protein"],
        "answer": (
            "Albumin is a protein made by the liver:\n"
            "- Normal range: 3.5-5.0 g/dL\n"
            "- Low albumin suggests reduced liver function\n"
            "- It helps maintain blood volume and transports substances\n"
            "- Low levels can cause fluid retention and swelling\n"
            "- Used in the Child-Pugh score\n\n"
            "Declining albumin levels may indicate worsening liver disease."
        ),
    },
    {
        "keywords": ["platelet", "bleeding", "bruise"],
        "answer": (
            "Platelets and liver disease:\n"
            "- Normal platelet count: 150,000-400,000 per µL\n"
            "- Low platelets (thrombocytopenia) is common in cirrhosis\n"
            "- The spleen enlarges and traps platelets\n"
            "- Low platelets increase bleeding and bruising risk\n"
            "- Platelet count is used in FIB-4 and APRI scores\n"
            "- Count below 150,000 may suggest significant fibrosis"
        ),
    },
    {
        "keywords": ["fatty liver", "nafld", "nash", "fatty"],
        "answer": (
            "Non-Alcoholic Fatty Liver Disease (NAFLD):\n"
            "- Most common liver disease worldwide\n"
            "- Fat accumulates in the liver without alcohol as a cause\n"
            "- Risk factors: obesity, diabetes, high cholesterol\n"
            "- NASH (Non-Alcoholic Steatohepatitis) is the inflammatory form\n"
            "- Can progress to fibrosis and eventually cirrhosis\n"
            "- Treatment: weight loss, exercise, managing diabetes\n"
            "- Losing 7-10% of body weight can reverse early NAFLD"
        ),
    },
    {
        "keywords": ["liver", "liver disease", "what is liver", "organ"],
        "answer": (
            "The liver is the body's largest internal organ and performs over 500 vital functions:\n"
            "- Filtering toxins from the blood\n"
            "- Producing bile for digestion\n"
            "- Making proteins (including albumin and clotting factors)\n"
            "- Storing energy as glycogen\n"
            "- Processing medications\n"
            "- Fighting infections\n\n"
            "Common liver diseases include cirrhosis, hepatitis, fatty liver disease (NAFLD/NASH), "
            "liver cancer, and autoimmune conditions. Early detection through regular blood tests "
            "is key to preventing permanent damage."
        ),
    },
    {
        "keywords": ["disease", "what disease", "liver problem", "condition", "disorder"],
        "answer": (
            "Common liver diseases and conditions include:\n"
            "- Cirrhosis: scarring of the liver from chronic damage\n"
            "- Hepatitis (A, B, C): viral infections causing liver inflammation\n"
            "- NAFLD/NASH: fat buildup in the liver (non-alcohol related)\n"
            "- Alcoholic liver disease: damage from excessive alcohol use\n"
            "- Liver cancer (hepatocellular carcinoma)\n"
            "- Autoimmune hepatitis\n"
            "- Wilson's disease and hemochromatosis (genetic)\n\n"
            "Hepato AI specifically focuses on predicting the risk of cirrhosis using "
            "16 clinical markers from standard blood tests."
        ),
    },
    {
        "keywords": ["what happens", "complications", "untreated", "dangerous", "serious", "fatal"],
        "answer": (
            "If liver cirrhosis is left untreated, serious complications can develop:\n"
            "- Portal hypertension (high blood pressure in liver veins)\n"
            "- Ascites (fluid buildup in the abdomen)\n"
            "- Variceal bleeding (life-threatening internal bleeding)\n"
            "- Hepatic encephalopathy (confusion due to toxin buildup)\n"
            "- Kidney failure (hepatorenal syndrome)\n"
            "- Liver cancer\n"
            "- Liver failure (life-threatening)\n\n"
            "This is why early detection is so important. Regular screening and "
            "timely treatment can prevent or slow these complications significantly."
        ),
    },
    {
        "keywords": ["normal range", "normal value", "reference range", "lab value", "lab result"],
        "answer": (
            "Key normal reference ranges for liver-related blood tests:\n"
            "- Bilirubin: 0.1 - 1.2 mg/dL\n"
            "- Albumin: 3.5 - 5.0 g/dL\n"
            "- SGOT (AST): 10 - 40 U/L\n"
            "- Alkaline Phosphatase: 44 - 147 U/L\n"
            "- Platelets: 150 - 400 (×1000/µL)\n"
            "- Prothrombin Time: 9.5 - 13.5 seconds\n"
            "- Cholesterol: below 200 mg/dL (desirable)\n"
            "- Triglycerides: below 150 mg/dL\n"
            "- Copper: 15 - 60 µg/day\n\n"
            "Values outside these ranges may indicate liver problems but should always "
            "be interpreted by a healthcare professional in clinical context."
        ),
    },
    {
        "keywords": ["my report", "my result", "my prediction", "my assessment", "my risk"],
        "answer": (
            "I'd love to help you understand your report! However, I need your "
            "prediction data to give you a personalised analysis.\n\n"
            "To get a detailed AI-powered explanation of your report:\n"
            "1. Make sure you have a Gemini API key configured\n"
            "2. Run a prediction first if you haven't\n"
            "3. Then ask me about your results!\n\n"
            "Without the AI integration, I can still answer general questions about "
            "liver health, lab values, and clinical scores."
        ),
    },
    {
        "keywords": ["hello", "hi", "hey", "good morning", "good evening"],
        "answer": (
            "Hello! I'm your Health Assistant AI. I can help you with questions about:\n"
            "- Liver cirrhosis and its symptoms\n"
            "- Causes and risk factors\n"
            "- Diagnostic tests and clinical scores\n"
            "- Diet and lifestyle recommendations\n"
            "- Treatment options\n"
            "- Understanding your lab values\n"
            "- Your prediction report (with AI mode enabled)\n\n"
            "How can I help you today?"
        ),
    },
    {
        "keywords": ["thank", "thanks", "bye", "goodbye"],
        "answer": (
            "You're welcome! Remember, this assistant provides general health "
            "information for educational purposes only. Always consult a qualified "
            "healthcare professional for medical advice. Take care!"
        ),
    },
    {
        "keywords": ["who are you", "what are you", "what can you do", "help"],
        "answer": (
            "I'm Health Assistant AI, built into Hepato AI. I can answer questions about:\n"
            "- Liver health and cirrhosis\n"
            "- Symptoms and warning signs\n"
            "- Diagnostic tests (LFT, FibroScan, etc.)\n"
            "- Clinical scores (MELD, FIB-4, APRI, Child-Pugh)\n"
            "- Diet and lifestyle tips\n"
            "- Treatment and prevention\n"
            "- Understanding lab values\n"
            "- Your personal report analysis (with AI mode)\n\n"
            "Just type your question and I'll do my best to help!"
        ),
    },
]

_FALLBACK = (
    "I'm not sure about that specific question. I can help with topics like:\n"
    "- Liver cirrhosis symptoms, causes, and stages\n"
    "- Diagnostic tests and clinical scores\n"
    "- Diet and lifestyle recommendations\n"
    "- Treatment options and prevention\n"
    "- Understanding lab values (bilirubin, albumin, platelets)\n\n"
    "Try asking about one of these topics! For specific medical advice, "
    "please consult a healthcare professional."
)


def _fuzzy_contains(keyword: str, text: str, threshold: int = 2) -> bool:
    """Check if keyword appears in text, allowing up to `threshold` typos."""
    if keyword in text:
        return True
    if len(keyword) < 4:
        return False
    for i in range(len(text) - len(keyword) + 1):
        window = text[i : i + len(keyword)]
        dist = sum(1 for a, b in zip(keyword, window) if a != b)
        if dist <= threshold:
            return True
    for i in range(len(text) - len(keyword)):
        window = text[i : i + len(keyword) + 1]
        dist = sum(1 for a, b in zip(keyword, window) if a != b)
        if dist <= threshold:
            return True
    return False


def _keyword_fallback(message: str) -> str:
    text = message.lower().strip()

    best_match = None
    best_score = 0

    for entry in _RESPONSES:
        score = 0
        for kw in entry["keywords"]:
            if kw in text:
                score += 2
            elif _fuzzy_contains(kw, text):
                score += 1
        if score > best_score:
            best_score = score
            best_match = entry

    if best_match and best_score > 0:
        return best_match["answer"]

    return _FALLBACK
