"""
Multi-Agent Travel Planner — single-file Streamlit app
Test target: BeeAI Framework 0.1.35, Python 3.11

Run:
    streamlit run main.py

Set GEMINI_API_KEY in a .env file in the same directory.
Never commit your .env file to source control.
"""

import asyncio
import logging
import os
from datetime import date, timedelta

import streamlit as st
from dotenv import load_dotenv

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import (
    ConditionalRequirement,
)
from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools import Tool
from beeai_framework.tools.handoff import HandoffTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool


# ------------------------------ Configuration ------------------------------

load_dotenv()
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

st.set_page_config(
    page_title="Voyage AI — Multi-Agent Travel Planner",
    page_icon="🧭",
    layout="wide",
)


def build_travel_agents():
    """Create the Gemini model and four BeeAI agents."""

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY was not found. Add it to a .env file beside main.py."
        )

    # BeeAI's Gemini adapter reads GEMINI_API_KEY from the environment.
    # If this model/provider ID is not recognized by your installed 0.1.35
    # package, see the setup notes provided with this file.
    model_name = os.getenv("GEMINI_CHAT_MODEL", "gemini:gemini-3.8-flash")
    llm = ChatModel.from_name(
        model_name,
        ChatModelParameters(temperature=0),
    )

    trajectory = lambda: GlobalTrajectoryMiddleware(included=[Tool])

    destination_expert = RequirementAgent(
        llm=llm,
        tools=[WikipediaTool(), ThinkTool()],
        memory=UnconstrainedMemory(),
        instructions=(
            "You are a Destination Research Expert. Research landmarks, historical "
            "sites, activities, transportation, accessibility, safety, and practical "
            "travel guidance. Use Wikipedia when useful, distinguish sourced facts "
            "from general advice, and never invent citations or current advisories."
        ),
        middlewares=[trajectory()],
        requirements=[
            ConditionalRequirement(
                ThinkTool,
                force_at_step=1,
                min_invocations=1,
                max_invocations=3,
                consecutive_allowed=False,
            ),
            ConditionalRequirement(
                WikipediaTool,
                only_after=[ThinkTool],
                min_invocations=1,
                max_invocations=3,
                consecutive_allowed=False,
            ),
        ],
    )

    weather_expert = RequirementAgent(
        llm=llm,
        tools=[OpenMeteoTool(), ThinkTool()],
        memory=UnconstrainedMemory(),
        instructions=(
            "You are a Travel Weather Specialist. Use Open-Meteo for available "
            "forecast data when possible. Clearly state the forecast dates and limits. "
            "For dates beyond the forecast horizon, give seasonal climate expectations "
            "instead of pretending to know the exact forecast. Provide packing advice."
        ),
        middlewares=[trajectory()],
        requirements=[
            ConditionalRequirement(
                ThinkTool,
                force_at_step=1,
                min_invocations=1,
                max_invocations=2,
                consecutive_allowed=False,
            ),
            ConditionalRequirement(
                OpenMeteoTool,
                only_after=[ThinkTool],
                min_invocations=1,
                max_invocations=1,
                consecutive_allowed=False,
            ),
        ],
    )

    culture_expert = RequirementAgent(
        llm=llm,
        tools=[WikipediaTool(), ThinkTool()],
        memory=UnconstrainedMemory(),
        instructions=(
            "You are a Language and Cultural Guidance Expert. Provide useful local "
            "phrases, pronunciation approximations, etiquette, religious sensitivities, "
            "dining customs, and respectful ways to interact with residents. Avoid "
            "stereotypes; explain that customs vary by person and situation."
        ),
        middlewares=[trajectory()],
        requirements=[
            ConditionalRequirement(
                ThinkTool,
                force_at_step=1,
                min_invocations=1,
                max_invocations=2,
                consecutive_allowed=False,
            ),
        ],
    )

    destination_handoff = HandoffTool(
        destination_expert,
        name="DestinationResearch",
        description=(
            "Ask the destination expert about landmarks, historical sites, transport, "
            "activities, safety considerations, and practical destination guidance."
        ),
    )
    weather_handoff = HandoffTool(
        weather_expert,
        name="WeatherPlanning",
        description=(
            "Ask the weather specialist for available forecasts, seasonal climate "
            "expectations, weather risks, and packing recommendations."
        ),
    )
    culture_handoff = HandoffTool(
        culture_expert,
        name="LanguageCulturalGuidance",
        description=(
            "Ask the language and culture expert for local phrases, etiquette, customs, "
            "and respectful communication advice."
        ),
    )

    coordinator = RequirementAgent(
        llm=llm,
        tools=[
            destination_handoff,
            weather_handoff,
            culture_handoff,
            ThinkTool(),
        ],
        memory=UnconstrainedMemory(),
        instructions=(
            "You are the lead Travel Coordinator. Read the user's full travel request, "
            "think about which expertise is needed, and consult all relevant specialist "
            "agents using their handoff tools. Then synthesize their findings into a "
            "cohesive, personalized plan. Include: (1) a practical itinerary, (2) "
            "destination and historical highlights, (3) weather and packing guidance, "
            "(4) language and cultural tips, (5) transport and safety considerations, "
            "and (6) assumptions or information that still needs checking. Do not claim "
            "a booking has been made. Distinguish current forecast data from seasonal "
            "expectations. If exact dates are missing, state that weather guidance is "
            "general. Be clear and actionable."
        ),
        middlewares=[trajectory()],
        requirements=[
            ConditionalRequirement(
                ThinkTool,
                consecutive_allowed=False,
            ),
        ],
    )

    return coordinator


async def create_plan(user_request: str) -> str:
    """Run the coordinator asynchronously and return its answer text."""
    coordinator = build_travel_agents()
    result = await coordinator.run(user_request)
    return result.answer.text


# ---------------------------------- UI --------------------------------------

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.8rem; padding-bottom: 2.5rem;}
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #172b4d, #245b75);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0; color: white;}
    .hero p {margin: .45rem 0 0 0; color: #e2edf7;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🧭 Voyage AI</h1>
      <p>A BeeAI multi-agent travel planner powered by Gemini</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "Tell the coordinator what kind of trip you want. It can delegate research to "
    "destination, weather, and language/culture specialists, then combine their findings."
)

with st.form("travel_request_form"):
    left, right = st.columns(2)

    with left:
        destination = st.text_input(
            "Destination",
            value="Japan",
            placeholder="e.g., Japan",
        )
        cities = st.text_input(
            "Cities or regions",
            value="Tokyo and Osaka",
            placeholder="e.g., Tokyo and Osaka",
        )
        departure_date = st.date_input(
            "Approximate start date",
            value=date.today() + timedelta(days=60),
            min_value=date.today(),
        )
        duration_days = st.slider("Trip duration (days)", 1, 30, 14)
        budget = st.selectbox(
            "Approximate budget per person",
            [
                "Not specified",
                "Budget",
                "Moderate",
                "Comfortable",
                "Luxury",
            ],
            index=2,
        )

    with right:
        travel_style = st.multiselect(
            "Interests",
            [
                "Traditional culture",
                "Historical sites",
                "Food",
                "Nature",
                "Museums and art",
                "Shopping",
                "Nightlife",
                "Family activities",
                "Local experiences",
            ],
            default=["Traditional culture", "Historical sites", "Local experiences"],
        )
        language_level = st.selectbox(
            "Local-language ability",
            [
                "English only",
                "A few local phrases",
                "Conversational",
                "Fluent",
            ],
        )
        travel_party = st.selectbox(
            "Who is travelling?",
            ["Solo", "Couple", "Friends", "Family", "Group"],
        )
        pace = st.select_slider(
            "Preferred pace",
            options=["Relaxed", "Balanced", "Busy"],
            value="Balanced",
        )

    additional_notes = st.text_area(
        "Additional requirements (optional)",
        placeholder=(
            "e.g., accessibility needs, dietary preferences, temples, "
            "avoiding crowds, transport preferences..."
        ),
        height=100,
    )

    consent = st.checkbox(
        "I approve running the AI planning workflow, including specialist-agent handoffs."
    )

    submitted = st.form_submit_button(
        "✨ Generate my travel plan",
        type="primary",
        use_container_width=True,
    )

if submitted:
    if not destination.strip():
        st.error("Please enter a destination.")
    elif not consent:
        st.warning("Please approve the planning workflow before running it.")
    elif not os.getenv("GEMINI_API_KEY"):
        st.error(
            "GEMINI_API_KEY is missing. Create a .env file beside main.py and add "
            "GEMINI_API_KEY=your_key_here, then restart Streamlit."
        )
    else:
        interests_text = ", ".join(travel_style) if travel_style else "General sightseeing"
        user_request = f"""
Plan a trip using the following user-provided details.

Destination: {destination.strip()}
Cities/regions: {cities.strip() or "Please recommend suitable locations"}
Start date: {departure_date.isoformat()}
Duration: {duration_days} days
Budget level: {budget}
Interests: {interests_text}
Local-language ability: {language_level}
Travel party: {travel_party}
Preferred pace: {pace}
Additional requirements: {additional_notes.strip() or "None provided"}

Please tailor the recommendations to these details. Consult the destination,
weather, and language/culture specialists where useful. Be transparent about
weather forecast limitations and anything that requires verification.
"""
        with st.spinner(
            "The coordinator is consulting the specialist agents. This may take a few minutes..."
        ):
            try:
                answer = asyncio.run(create_plan(user_request))
                st.session_state["last_travel_plan"] = answer
                st.session_state["last_travel_request"] = user_request
            except Exception as exc:
                st.error("The travel-planning workflow failed.")
                st.exception(exc)
                st.info(
                    "If the error mentions an unknown model/provider, check "
                    "GEMINI_CHAT_MODEL and the provider support in your installed "
                    "BeeAI version. If it mentions a requirement or tool call, test "
                    "the agents one at a time before running the full workflow."
                )

if st.session_state.get("last_travel_plan"):
    st.divider()
    st.subheader("🗺️ Your personalized travel plan")
    st.caption("Generated by the BeeAI coordinator using your form responses.")
    st.markdown(st.session_state["last_travel_plan"])

    st.download_button(
        "Download plan as Markdown",
        data=st.session_state["last_travel_plan"],
        file_name="travel_plan.md",
        mime="text/markdown",
        use_container_width=True,
    )

with st.sidebar:
    st.header("About this app")
    st.write("**LLM:** Gemini API")
    st.write("**Agent framework:** BeeAI")
    st.write("**Interface:** Streamlit")
    st.write("**Research tools:** Wikipedia and Open-Meteo")
    st.divider()
    st.caption(
        "The app runs on your computer, but Gemini and the research/weather tools "
        "need internet access. Do not share your API key."
    )
