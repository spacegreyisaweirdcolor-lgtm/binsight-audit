
import json
from datetime import date
from html import escape

import pandas as pd
import plotly.express as px
import streamlit as st
from google import genai
from google.genai import types


MODEL_NAME = "gemini-2.5-flash"

MATERIALS = [
    "Plastic",
    "Paper",
    "Cardboard",
    "Metal",
    "Glass",
    "Food waste",
    "Mixed material",
    "Hazardous or special waste",
    "Other",
]

BINS = ["Recycling", "Compost", "Trash", "Special drop-off", "Uncertain"]

DECISIONS = [
    "Likely recycle",
    "Likely compost",
    "Likely trash",
    "Special handling needed",
    "Uncertain, check local rules",
]

PROBLEM_JUDGMENTS = [
    "Likely contamination",
    "Possible contamination",
    "Not diverted",
    "Special handling needed",
    "Mismatch or uncertainty",
    "Needs review",
]

BIN_COLORS = {
    "Recycling": "#0F7B4F",
    "Compost": "#6A8E23",
    "Trash": "#4B5563",
    "Special drop-off": "#B7791F",
    "Uncertain": "#2563EB",
}

BIN_ICONS = {
    "Recycling": "♻️",
    "Compost": "🌱",
    "Trash": "🗑️",
    "Special drop-off": "⚠️",
    "Uncertain": "❓",
}


st.set_page_config(
    page_title="BinSight Audit",
    page_icon="♻️",
    layout="wide",
)


st.markdown(
    """
<style>
    .block-container {
        padding-top: 1.2rem;
        max-width: 1280px;
    }

    .top-hero {
        padding: 1.6rem 1.8rem;
        border-radius: 26px;
        background: linear-gradient(135deg, #0B3D2E 0%, #106B45 48%, #7ACB91 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(16, 107, 69, 0.20);
        margin-bottom: 1rem;
    }

    .top-hero h1 {
        margin: 0;
        font-size: 2.7rem;
        letter-spacing: -0.04rem;
    }

    .top-hero p {
        margin-top: 0.35rem;
        margin-bottom: 0;
        font-size: 1.05rem;
        opacity: 0.95;
        max-width: 900px;
    }

    .result-card {
        border-radius: 28px;
        padding: 1.4rem 1.5rem;
        color: white;
        box-shadow: 0 18px 42px rgba(0, 0, 0, 0.15);
        margin-bottom: 0.85rem;
    }

    .result-card .label {
        font-size: 0.78rem;
        font-weight: 900;
        letter-spacing: 0.08rem;
        text-transform: uppercase;
        opacity: 0.82;
    }

    .result-card .bin {
        font-size: 2.25rem;
        font-weight: 900;
        margin-top: 0.15rem;
        letter-spacing: -0.03rem;
    }

    .result-card .sub {
        margin-top: 0.4rem;
        font-size: 1rem;
        opacity: 0.96;
    }

    .detail-card {
        border: 1px solid #E2ECE5;
        background: white;
        border-radius: 20px;
        padding: 1rem;
        box-shadow: 0 8px 20px rgba(13, 61, 46, 0.04);
        margin-bottom: 0.8rem;
    }

    .detail-title {
        font-size: 0.78rem;
        color: #106B45;
        font-weight: 900;
        letter-spacing: 0.08rem;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E2ECE5;
        border-radius: 18px;
        padding: 0.85rem;
        box-shadow: 0 8px 20px rgba(13, 61, 46, 0.04);
    }

    .stButton > button {
        border-radius: 14px;
        font-weight: 800;
    }
</style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
<div class="top-hero">
    <h1>BinSight Audit</h1>
    <p>Upload a waste photo. Get an AI sorting recommendation. Answer smart follow-up checks. Save the final result into a live audit dashboard.</p>
</div>
""",
    unsafe_allow_html=True,
)


if "rows" not in st.session_state:
    st.session_state.rows = []

if "ai_result" not in st.session_state:
    st.session_state.ai_result = None

if "last_image_signature" not in st.session_state:
    st.session_state.last_image_signature = None


def clean_json_text(text):
    if not text:
        return ""

    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned.replace("```json", "", 1).strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.replace("```", "", 1).strip()

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    return cleaned


def normalize_list(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def safe_confidence(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0

    return max(0.0, min(1.0, number))


def validate_ai_result(result):
    if not isinstance(result, dict):
        result = {}

    material = result.get("material_category", "Other")
    likely_bin = result.get("likely_bin", "Uncertain")
    decision = result.get("decision_label", "Uncertain, check local rules")

    if material not in MATERIALS:
        material = "Other"
    if likely_bin not in BINS:
        likely_bin = "Uncertain"
    if decision not in DECISIONS:
        decision = "Uncertain, check local rules"

    return {
        "item_name": str(result.get("item_name") or "Unclear item"),
        "material_category": material,
        "likely_bin": likely_bin,
        "decision_label": decision,
        "confidence": safe_confidence(result.get("confidence", 0)),
        "plain_english": str(result.get("plain_english") or "This item needs a closer look before sorting."),
        "how_to_dispose": normalize_list(result.get("how_to_dispose")),
        "why": str(result.get("why") or "No explanation provided."),
        "contamination_risks": normalize_list(result.get("contamination_risks")),
        "visible_clues": normalize_list(result.get("visible_clues")),
        "follow_up_questions": normalize_list(result.get("follow_up_questions")),
        "audit_tags": normalize_list(result.get("audit_tags")),
        "local_rule_note": str(result.get("local_rule_note") or "Check local rules before making a final decision."),
        "education_message": str(result.get("education_message") or "When unsure, check before sorting."),
    }


def analyze_image_with_vision(image_bytes, mime_type, api_key, compost_available, glass_accepted):
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a careful school waste sorting assistant for a student waste audit.

Analyze the most visible waste item in the image.

Important distinction:
- item_name should be the object, such as "plastic bottle", "chip bag", "banana peel", "paper cup", or "aluminum can".
- material_category should be the broad material, such as Plastic, Metal, Paper, Food waste, or Mixed material.
- Do not use a vague item_name like "plastic" unless the object truly cannot be identified.

Local audit context:
- Compost available at this site: {compost_available}
- Glass accepted in recycling at this site: {glass_accepted}
- Use general school waste guidance. Local rules can vary.

Return only valid JSON using this exact schema:
{{
  "item_name": string,
  "material_category": one of ["Plastic", "Paper", "Cardboard", "Metal", "Glass", "Food waste", "Mixed material", "Hazardous or special waste", "Other"],
  "likely_bin": one of ["Recycling", "Compost", "Trash", "Special drop-off", "Uncertain"],
  "decision_label": one of ["Likely recycle", "Likely compost", "Likely trash", "Special handling needed", "Uncertain, check local rules"],
  "confidence": number from 0 to 1,
  "plain_english": string,
  "how_to_dispose": list of 2 to 4 short steps,
  "why": string,
  "contamination_risks": list of strings,
  "visible_clues": list of strings,
  "follow_up_questions": list of strings,
  "audit_tags": list of strings,
  "local_rule_note": string,
  "education_message": string
}}

Sorting guidance:
- If it looks like a clean bottle, clean can, clean jar, clean paper, or clean cardboard, likely_bin can be Recycling, but mention local rules.
- If it looks like a chip bag, candy wrapper, pouch, plastic film, or flexible wrapper, likely_bin should usually be Trash, not Recycling.
- If it looks like food waste and compost is available, likely_bin should be Compost.
- If it looks like food waste and compost is not available, likely_bin should be Trash.
- If it looks like a battery, electronic, chemical container, aerosol can, sharp, or medical waste, likely_bin should be Special drop-off.
- If the item is greasy, wet, food-stained, half-full, or mixed-material, explain the contamination risk.
- If the image is unclear, use Uncertain and ask follow-up questions.
- Return JSON only. No markdown.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            prompt,
        ],
        config={"response_mime_type": "application/json"},
    )

    text = clean_json_text(response.text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {
            "item_name": "Unclear item",
            "material_category": "Other",
            "likely_bin": "Uncertain",
            "decision_label": "Uncertain, check local rules",
            "confidence": 0,
            "plain_english": "The image could not be analyzed clearly.",
            "how_to_dispose": [
                "Inspect the item manually.",
                "Check whether it is clean, empty, greasy, or mixed material.",
            ],
            "why": "The model response was not readable JSON.",
            "contamination_risks": [],
            "visible_clues": [],
            "follow_up_questions": ["What object is this?", "Is it clean and empty?"],
            "audit_tags": ["unclear"],
            "local_rule_note": "Check local rules before making a final decision.",
            "education_message": "When unsure, pause and check before sorting.",
        }

    return validate_ai_result(parsed)


def yes(value):
    return value == "Yes"


def no(value):
    return value == "No"


def should_show_plastic_checks(result):
    item = result["item_name"].lower()
    material = result["material_category"]
    bottle_words = ["bottle", "container", "jug", "tub", "cup", "lid", "cap"]
    if material == "Plastic":
        return True
    return any(word in item for word in bottle_words)


def refine_recommendation(result, answers, compost_available, glass_accepted):
    original_bin = result["likely_bin"]
    final_bin = original_bin
    final_label = result["decision_label"]
    reasons = []
    steps = list(result["how_to_dispose"])
    changed = False

    item_name = result["item_name"].lower()
    material = result["material_category"]

    clean_empty = answers["clean_empty"]
    has_food = answers["has_food"]
    has_liquid = answers["has_liquid"]
    greasy = answers["greasy"]
    flexible = answers["flexible"]
    mixed = answers["mixed"]
    special = answers["special"]
    resin_code = answers["resin_code"]
    cap_rule = answers["cap_rule"]
    cap_status = answers["cap_status"]

    if yes(special) or material == "Hazardous or special waste":
        final_bin = "Special drop-off"
        final_label = "Special handling needed"
        reasons.append("special waste risk")
        steps = [
            "Do not place this item in normal trash, recycling, or compost.",
            "Use a school-approved special drop-off box or ask a staff member.",
        ]
        changed = True

    elif material == "Food waste":
        if compost_available == "Yes":
            final_bin = "Compost"
            final_label = "Likely compost"
            reasons.append("food waste and compost is available")
            steps = [
                "Place food waste in compost.",
                "Remove non-compostable packaging first.",
            ]
        else:
            final_bin = "Trash"
            final_label = "Likely trash"
            reasons.append("food waste but compost is not available")
            steps = [
                "Place the food waste in trash.",
                "Do not put loose food waste in recycling.",
            ]
        changed = final_bin != original_bin

    elif yes(flexible):
        final_bin = "Trash"
        final_label = "Likely trash"
        reasons.append("flexible plastic film or wrapper usually does not belong in regular recycling")
        steps = [
            "Place this item in trash unless your site has a special film drop-off.",
            "Do not place flexible wrappers or film in regular recycling.",
        ]
        changed = True

    elif yes(mixed):
        if original_bin == "Recycling":
            final_bin = "Trash"
            final_label = "Likely trash"
            reasons.append("mixed material packaging can contaminate recycling")
            steps = [
                "Place mixed-material packaging in trash unless your site has a special program.",
                "Use recycling only when the item is clearly accepted and clean.",
            ]
            changed = True
        else:
            reasons.append("mixed material packaging needs local confirmation")

    elif yes(has_food) or yes(has_liquid) or yes(greasy) or no(clean_empty):
        if original_bin == "Recycling":
            final_bin = "Trash"
            final_label = "Likely trash"
            reasons.append("item is not clean and empty enough for recycling as-is")
            steps = [
                "Do not recycle the item as-is.",
                "Empty or clean it if your school accepts the material.",
                "If it cannot be cleaned, place it in trash.",
            ]
            changed = True
        else:
            reasons.append("residue, liquid, or grease may affect disposal")

    elif material == "Plastic" or "bottle" in item_name or "container" in item_name:
        if resin_code in ["#1 PET or PETE", "#2 HDPE"]:
            if clean_empty == "Yes":
                final_bin = "Recycling"
                final_label = "Likely recycle"
                reasons.append(f"{resin_code} plastic and item is clean and empty")
                steps = [
                    "Recycle the bottle or container if your school accepts this plastic number.",
                    "Make sure it is empty before placing it in recycling.",
                ]
                changed = final_bin != original_bin
            elif clean_empty == "Unsure":
                final_bin = "Uncertain"
                final_label = "Uncertain, check local rules"
                reasons.append("plastic number may be recyclable, but cleanliness is unknown")
                steps = [
                    "Check whether the item is empty and clean.",
                    "If it has residue, do not recycle it as-is.",
                ]
                changed = True

            if cap_status == "Cap on" and cap_rule == "No, remove caps":
                steps.append("Remove the cap before recycling because the site does not accept caps on.")
                reasons.append("cap rule says caps should be removed")
            elif cap_status == "Cap off" and cap_rule == "Yes, caps accepted on":
                steps.append("The cap can be attached if your school accepts caps on.")
                reasons.append("cap rule allows caps on")

        elif resin_code in ["#3 PVC", "#4 LDPE", "#5 PP", "#6 PS", "#7 Other", "No symbol visible"]:
            final_bin = "Uncertain"
            final_label = "Uncertain, check local rules"
            reasons.append(f"plastic code is {resin_code}, which needs local confirmation")
            steps = [
                "Check whether your school or hauler accepts this plastic number.",
                "If the program does not accept it, place it in trash.",
            ]
            changed = True

    elif material == "Glass":
        if glass_accepted == "Yes" and clean_empty == "Yes":
            final_bin = "Recycling"
            final_label = "Likely recycle"
            reasons.append("glass is accepted and item is clean and empty")
            steps = [
                "Place clean glass in recycling if your site accepts glass.",
                "Do not recycle broken glass unless your site specifically allows it.",
            ]
            changed = final_bin != original_bin
        elif glass_accepted == "No":
            final_bin = "Trash"
            final_label = "Likely trash"
            reasons.append("glass is not accepted in this recycling program")
            steps = [
                "Do not place this glass item in recycling at this site.",
                "Use trash or a separate glass drop-off if available.",
            ]
            changed = True
        else:
            final_bin = "Uncertain"
            final_label = "Uncertain, check local rules"
            reasons.append("glass acceptance is unknown")
            steps = [
                "Check whether glass is accepted at this site.",
                "If not accepted, do not place it in recycling.",
            ]
            changed = True

    if not reasons:
        reasons.append("AI recommendation accepted because follow-up answers did not contradict it")

    return {
        "final_bin": final_bin,
        "final_label": final_label,
        "reasons": reasons,
        "steps": steps,
        "changed": changed,
    }


def infer_audit_judgment(bin_found, refined):
    final_bin = refined["final_bin"]
    reasons = ", ".join(refined["reasons"])

    if bin_found == "Unsure":
        return "Needs review", "the original bin was not recorded"

    if final_bin == "Special drop-off":
        if bin_found == "Special drop-off":
            return "Likely correctly sorted", "special item was placed in special drop-off"
        return "Special handling needed", "item should not go in normal trash, recycling, or compost"

    if final_bin == "Uncertain":
        return "Mismatch or uncertainty", reasons

    if bin_found == final_bin:
        if final_bin == "Recycling" and any(word in reasons.lower() for word in ["residue", "liquid", "grease", "mixed", "wrapper"]):
            return "Possible contamination", reasons
        return "Likely correctly sorted", f"found in the same bin as the refined recommendation: {final_bin}"

    if bin_found == "Trash" and final_bin in ["Recycling", "Compost"]:
        return "Not diverted", f"refined result suggests {final_bin.lower()}, but item was found in trash"

    if bin_found in ["Recycling", "Compost"] and final_bin == "Trash":
        return "Likely contamination", f"refined result suggests trash, but item was found in {bin_found.lower()}"

    return "Mismatch or uncertainty", f"found in {bin_found}, but refined result suggests {final_bin}"


def make_recommendation(df):
    if df.empty:
        return "Save items first to generate an intervention idea."

    problem_df = df[df["audit_judgment"].isin(PROBLEM_JUDGMENTS)]

    if problem_df.empty:
        return "No clear problem pattern yet. Keep logging items or audit a busier bin."

    reasons = []
    for value in problem_df["audit_reason"].dropna():
        reasons.extend([part.strip() for part in str(value).split(",") if part.strip()])

    if not reasons:
        return "Review the saved items and identify the repeated sorting issue."

    top_reason = pd.Series(reasons).value_counts().idxmax()
    lower = top_reason.lower()

    if "wrapper" in lower or "film" in lower or "flexible" in lower:
        return "Create a visual sign showing that chip bags, candy wrappers, and flexible plastic film usually go to trash, not regular recycling."
    if "food" in lower or "residue" in lower:
        return "Add a sign near recycling bins that says: empty food first. Use pictures of cafeteria containers students actually use."
    if "liquid" in lower:
        return "Add a reminder or liquid-emptying station next to the recycling bins."
    if "grease" in lower:
        return "Create a cafeteria sign explaining that greasy cardboard and greasy paper can contaminate recycling."
    if "mixed" in lower:
        return "Show examples of mixed material packaging and explain why these items are difficult to recycle."
    if "special" in lower:
        return "Create a clearly labeled special drop-off box for batteries, electronics, aerosols, and other risky items."
    if "plastic code" in lower or "plastic number" in lower:
        return "Add a sign showing which plastic numbers your school accepts, especially #1 and #2 if those are accepted."

    return f"Focus the next sign or announcement on this repeated issue: {top_reason}."


def load_demo_rows():
    return [
        {
            "date": str(date.today()),
            "location": "Cafeteria",
            "bin_found": "Recycling",
            "item_name": "Chip bag",
            "material_category": "Mixed material",
            "ai_bin": "Trash",
            "final_bin": "Trash",
            "decision_label": "Likely trash",
            "confidence": 0.86,
            "audit_judgment": "Likely contamination",
            "audit_reason": "flexible plastic film or wrapper usually does not belong in regular recycling",
            "education_message": "Wrappers and chip bags usually do not belong in regular recycling.",
        },
        {
            "date": str(date.today()),
            "location": "Cafeteria",
            "bin_found": "Trash",
            "item_name": "Clean aluminum can",
            "material_category": "Metal",
            "ai_bin": "Recycling",
            "final_bin": "Recycling",
            "decision_label": "Likely recycle",
            "confidence": 0.93,
            "audit_judgment": "Not diverted",
            "audit_reason": "refined result suggests recycling, but item was found in trash",
            "education_message": "Clean cans are often recyclable, but local rules should still be checked.",
        },
        {
            "date": str(date.today()),
            "location": "Cafeteria",
            "bin_found": "Recycling",
            "item_name": "Greasy pizza box",
            "material_category": "Cardboard",
            "ai_bin": "Trash",
            "final_bin": "Trash",
            "decision_label": "Likely trash",
            "confidence": 0.78,
            "audit_judgment": "Likely contamination",
            "audit_reason": "item is not clean and empty enough for recycling as-is",
            "education_message": "Grease can make cardboard difficult to recycle.",
        },
    ]


def get_df():
    if not st.session_state.rows:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.rows)


def render_result_card(title, bin_name, label, item_name, plain_text):
    color = BIN_COLORS.get(bin_name, "#2563EB")
    icon = BIN_ICONS.get(bin_name, "❓")

    st.markdown(
        f"""
<div class="result-card" style="background: linear-gradient(135deg, {color} 0%, #12231C 100%);">
    <div class="label">{escape(title)}</div>
    <div class="bin">{escape(icon)} {escape(bin_name)}</div>
    <div class="sub"><strong>{escape(label)}</strong> for <strong>{escape(item_name)}</strong></div>
    <div class="sub">{escape(plain_text)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def bullet_list(title, items, empty_text):
    st.markdown(f"<div class='detail-title'>{escape(title)}</div>", unsafe_allow_html=True)
    if items:
        for item in items:
            st.write(f"• {item}")
    else:
        st.caption(empty_text)


def render_snapshot(df):
    if df.empty:
        st.info("No saved audit items yet. Save one analyzed item and the dashboard will update here.")
        return

    total = len(df)
    problem_count = len(df[df["audit_judgment"].isin(PROBLEM_JUDGMENTS)])
    problem_rate = problem_count / total if total else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Saved items", total)
    m2.metric("Problem items", problem_count)
    m3.metric("Problem rate", f"{problem_rate:.0%}")

    st.dataframe(
        df.tail(5)[["item_name", "bin_found", "final_bin", "audit_judgment"]],
        use_container_width=True,
        hide_index=True,
    )


def build_report(df):
    if df.empty:
        return "No saved data yet."

    total = len(df)
    problem_count = len(df[df["audit_judgment"].isin(PROBLEM_JUDGMENTS)])
    problem_rate = problem_count / total if total else 0
    top_location = df["location"].mode()[0]
    top_material = df["material_category"].mode()[0]
    top_reason = df["audit_reason"].mode()[0]
    recommendation = make_recommendation(df)

    return f"""
This audit recorded {total} waste items, mostly from the {top_location.lower()} area.

The app found a {problem_rate:.0%} sorting problem rate. The most common material category was {top_material.lower()}. The most common sorting issue was: {top_reason}.

Recommended intervention: {recommendation}

The value of this audit is that it does not only count waste. It combines AI image recognition with follow-up answers about residue, recycling symbols, caps, and local program rules. That makes the final result more useful than a simple photo classifier.
""".strip()


st.sidebar.header("Site settings")
st.sidebar.caption("These settings affect the final recommendation.")
compost_available = st.sidebar.selectbox("Compost available?", ["Yes", "No", "Unsure"])
glass_accepted = st.sidebar.selectbox("Glass accepted in recycling?", ["Yes", "No", "Unsure"])
st.sidebar.caption("AI vision is enabled when a server-side API key is configured.")


tabs = st.tabs(["Scanner", "Dashboard", "Audit log", "Report"])


with tabs[0]:
    left, right = st.columns([0.9, 1.1], gap="large")

    with left:
        st.subheader("Scan an item")

        audit_date = st.date_input("Audit date", value=date.today())

        location = st.selectbox(
            "Where are you auditing?",
            ["Cafeteria", "Classroom", "Hallway", "Library", "School event", "Home", "Community space"],
        )

        bin_found = st.selectbox(
            "Where was this item found?",
            ["Recycling", "Trash", "Compost", "Special drop-off", "Unsure"],
        )

        image_source = st.radio("Photo source", ["Upload", "Camera"], horizontal=True)

        image_file = None
        if image_source == "Upload":
            image_file = st.file_uploader("Upload one item photo", type=["jpg", "jpeg", "png", "webp"])
        else:
            image_file = st.camera_input("Take a photo")

        if image_file is not None:
            image_bytes = image_file.getvalue()
            mime_type = getattr(image_file, "type", None) or "image/jpeg"
            signature = f"{len(image_bytes)}-{image_bytes[:16]}"

            if st.session_state.last_image_signature != signature:
                st.session_state.ai_result = None
                st.session_state.last_image_signature = signature

            st.image(image_bytes, caption="Item ready to analyze", use_container_width=True)

            if st.button("Analyze item", type="primary", use_container_width=True):
                api_key = st.secrets.get("VISION_API_KEY") or st.secrets.get("GEMINI_API_KEY")

                if not api_key:
                    st.error("No vision API key found. Add VISION_API_KEY to .streamlit/secrets.toml.")
                else:
                    try:
                        with st.spinner("Identifying object, material, disposal route, and audit risks..."):
                            st.session_state.ai_result = analyze_image_with_vision(
                                image_bytes=image_bytes,
                                mime_type=mime_type,
                                api_key=api_key,
                                compost_available=compost_available,
                                glass_accepted=glass_accepted,
                            )

                    except Exception as error:
                        st.error(f"AI analysis failed: {error}")
        else:
            st.info("Upload or take one clear photo. The sorting card will appear on the right.")

    with right:
        st.subheader("Sorting result")

        result = st.session_state.ai_result

        if result is None:
            st.info("No result yet. After analysis, this area will show the AI recommendation and smart follow-up questions.")
        else:
            render_result_card(
                "AI first pass",
                result["likely_bin"],
                result["decision_label"],
                result["item_name"],
                result["plain_english"],
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("Detected item", result["item_name"])
            m2.metric("Material family", result["material_category"])
            m3.metric("AI confidence", f"{result['confidence']:.0%}")

            st.caption("The first pass is not the final answer. The follow-up answers below can change the disposal recommendation.")

            with st.expander("What AI saw and why", expanded=False):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("<div class='detail-card'>", unsafe_allow_html=True)
                    bullet_list("Visible clues", result["visible_clues"], "No visible clues listed.")
                    st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("<div class='detail-card'>", unsafe_allow_html=True)
                    st.markdown("<div class='detail-title'>Why AI suggested this</div>", unsafe_allow_html=True)
                    st.write(result["why"])
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_b:
                    st.markdown("<div class='detail-card'>", unsafe_allow_html=True)
                    bullet_list("AI contamination risks", result["contamination_risks"], "No obvious risks detected from the image.")
                    st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("<div class='detail-card'>", unsafe_allow_html=True)
                    bullet_list("AI follow-up questions", result["follow_up_questions"], "No follow-up questions suggested.")
                    st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("Smart follow-up checks")
            st.caption("These answers update the final disposal result automatically.")

            q1, q2 = st.columns(2)

            with q1:
                clean_empty = st.selectbox("Is it completely empty and clean?", ["Unsure", "Yes", "No"])
                has_food = st.selectbox("Is there food residue?", ["Unsure", "Yes", "No"])
                has_liquid = st.selectbox("Is there liquid inside or residue from liquid?", ["Unsure", "Yes", "No"])
                greasy = st.selectbox("Is it greasy or oil-stained?", ["Unsure", "Yes", "No"])

            with q2:
                flexible = st.selectbox("Is it flexible plastic film, a wrapper, or a bag?", ["Unsure", "Yes", "No"])
                mixed = st.selectbox("Is it mixed-material packaging?", ["Unsure", "Yes", "No"])
                special = st.selectbox("Is it a battery, electronic, aerosol, sharp, chemical, or medical item?", ["Unsure", "Yes", "No"])

                if should_show_plastic_checks(result):
                    resin_code = st.selectbox(
                        "Visible recycling symbol or plastic number",
                        [
                            "Unsure",
                            "#1 PET or PETE",
                            "#2 HDPE",
                            "#3 PVC",
                            "#4 LDPE",
                            "#5 PP",
                            "#6 PS",
                            "#7 Other",
                            "No symbol visible",
                            "Not applicable",
                        ],
                    )
                    cap_status = st.selectbox("Bottle cap status", ["Unsure", "Cap on", "Cap off", "Not applicable"])
                    cap_rule = st.selectbox("Does your school accept caps on bottles?", ["Unsure", "Yes, caps accepted on", "No, remove caps", "Not applicable"])
                else:
                    resin_code = "Not applicable"
                    cap_status = "Not applicable"
                    cap_rule = "Not applicable"

            answers = {
                "clean_empty": clean_empty,
                "has_food": has_food,
                "has_liquid": has_liquid,
                "greasy": greasy,
                "flexible": flexible,
                "mixed": mixed,
                "special": special,
                "resin_code": resin_code,
                "cap_status": cap_status,
                "cap_rule": cap_rule,
            }

            refined = refine_recommendation(
                result=result,
                answers=answers,
                compost_available=compost_available,
                glass_accepted=glass_accepted,
            )

            render_result_card(
                "Final recommendation after follow-up",
                refined["final_bin"],
                refined["final_label"],
                result["item_name"],
                "This result uses both the image analysis and your follow-up answers.",
            )

            if refined["changed"]:
                st.warning("The follow-up answers changed the original AI recommendation.")
            else:
                st.success("The follow-up answers support the original AI recommendation.")

            st.markdown("<div class='detail-card'>", unsafe_allow_html=True)
            bullet_list("Final disposal steps", refined["steps"], "No disposal steps available.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='detail-card'>", unsafe_allow_html=True)
            bullet_list("Why the final answer changed or stayed the same", refined["reasons"], "No reason available.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.info(result["education_message"])
            st.caption(result["local_rule_note"])

            audit_judgment, audit_reason = infer_audit_judgment(bin_found, refined)

            st.subheader("Audit interpretation")
            st.success(f"{audit_judgment}: {audit_reason}")

            with st.expander("Optional manual correction"):
                edited_item = st.text_input("Item name", value=result["item_name"])
                edited_material = st.selectbox(
                    "Material family",
                    MATERIALS,
                    index=MATERIALS.index(result["material_category"]),
                )
                edited_bin = st.selectbox(
                    "Final disposal bin",
                    BINS,
                    index=BINS.index(refined["final_bin"]),
                )
                edited_decision = st.selectbox(
                    "Decision label",
                    DECISIONS,
                    index=DECISIONS.index(refined["final_label"]),
                )
                notes = st.text_area("Notes", value="")

            if st.button("Save final result to audit dashboard", type="primary", use_container_width=True):
                row = {
                    "date": str(audit_date),
                    "location": location,
                    "bin_found": bin_found,
                    "item_name": edited_item,
                    "material_category": edited_material,
                    "ai_bin": result["likely_bin"],
                    "final_bin": edited_bin,
                    "decision_label": edited_decision,
                    "confidence": result["confidence"],
                    "clean_empty": clean_empty,
                    "food_residue": has_food,
                    "liquid_residue": has_liquid,
                    "greasy": greasy,
                    "flexible_plastic": flexible,
                    "mixed_material": mixed,
                    "special_waste": special,
                    "resin_code": resin_code,
                    "cap_status": cap_status,
                    "cap_rule": cap_rule,
                    "audit_judgment": audit_judgment,
                    "audit_reason": audit_reason,
                    "education_message": result["education_message"],
                    "notes": notes,
                }

                st.session_state.rows.append(row)
                st.success("Saved. The dashboard has been updated.")

    st.divider()
    st.subheader("Live audit snapshot")
    render_snapshot(get_df())


with tabs[1]:
    st.subheader("Audit dashboard")

    df = get_df()

    if df.empty:
        st.info("No saved rows yet. Use the scanner or load demo data.")
        if st.button("Load demo data", type="primary"):
            st.session_state.rows.extend(load_demo_rows())
            st.rerun()
    else:
        total = len(df)
        problem_df = df[df["audit_judgment"].isin(PROBLEM_JUDGMENTS)]
        problem_rate = len(problem_df) / total if total else 0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total items", total)
        m2.metric("Problem items", len(problem_df))
        m3.metric("Problem rate", f"{problem_rate:.0%}")
        m4.metric("Most common site", df["location"].mode()[0])

        chart_left, chart_right = st.columns(2)

        with chart_left:
            material_counts = df["material_category"].value_counts().reset_index()
            material_counts.columns = ["material", "count"]
            fig = px.bar(material_counts, x="material", y="count", title="Material types found")
            fig.update_layout(height=390, margin=dict(l=20, r=20, t=55, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with chart_right:
            bin_counts = df["final_bin"].value_counts().reset_index()
            bin_counts.columns = ["bin", "count"]
            fig = px.pie(
                bin_counts,
                names="bin",
                values="count",
                title="Final disposal recommendations",
                hole=0.45,
            )
            fig.update_layout(height=390, margin=dict(l=20, r=20, t=55, b=20))
            st.plotly_chart(fig, use_container_width=True)

        judgment_counts = df["audit_judgment"].value_counts().reset_index()
        judgment_counts.columns = ["judgment", "count"]
        fig = px.bar(judgment_counts, x="judgment", y="count", title="Sorting outcomes")
        fig.update_layout(height=360, margin=dict(l=20, r=20, t=55, b=20))
        st.plotly_chart(fig, use_container_width=True)

        reason_counts = df["audit_reason"].value_counts().head(10).reset_index()
        reason_counts.columns = ["reason", "count"]
        fig = px.bar(reason_counts, x="count", y="reason", orientation="h", title="Top audit problems")
        fig.update_layout(height=420, margin=dict(l=20, r=20, t=55, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recommended intervention")
        st.success(make_recommendation(df))


with tabs[2]:
    st.subheader("Audit log")

    a, b, c = st.columns([1, 1, 2])

    with a:
        if st.button("Load demo data", use_container_width=True):
            st.session_state.rows.extend(load_demo_rows())
            st.rerun()

    with b:
        if st.button("Clear all data", use_container_width=True):
            st.session_state.rows = []
            st.session_state.ai_result = None
            st.rerun()

    with c:
        st.caption("Download the CSV for your competition write-up or presentation.")

    df = get_df()

    if df.empty:
        st.info("No rows yet.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="binsight_audit_data.csv",
            mime="text/csv",
        )


with tabs[3]:
    st.subheader("Automatic report")

    df = get_df()

    if df.empty:
        st.info("No report yet. Save items first or load demo data.")
    else:
        report = build_report(df)

        st.write(report)

        st.download_button(
            "Download report",
            data=report,
            file_name="binsight_audit_report.txt",
            mime="text/plain",
        )
