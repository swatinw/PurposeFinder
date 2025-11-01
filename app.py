import streamlit as st
import pandas as pd
import numpy as np
import os
import json

# Optional OpenAI integration
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    use_ai = True
except Exception:
    use_ai = False

# -----------------------------
# Helper functions
# -----------------------------
def calculate_personality_score(responses):
    """Compute Big Five-like mini score."""
    traits = {
        "Openness": np.mean([responses.get("imagination", 3), responses.get("curiosity", 3)]),
        "Conscientiousness": np.mean([responses.get("organized", 3), responses.get("responsibility", 3)]),
        "Extraversion": np.mean([responses.get("outgoing", 3), responses.get("energy", 3)]),
        "Agreeableness": np.mean([responses.get("helpful", 3), responses.get("trust", 3)]),
        "Neuroticism": np.mean([responses.get("stress", 3), responses.get("worry", 3)]),
    }
    return traits

def get_top_domains(traits):
    """Recommend focus domain based on personality"""
    if traits["Extraversion"] > 3.5:
        return ["Leadership", "Social impact"]
    elif traits["Openness"] > 3.5:
        return ["Creativity", "Innovation"]
    elif traits["Conscientiousness"] > 3.5:
        return ["Achievement", "Structure"]
    elif traits["Agreeableness"] > 3.5:
        return ["Helping", "Community"]
    else:
        return ["Balance", "Stability"]

def generate_purpose_statement(traits, values, motivations):
    """Use AI if available, else fallback text"""
    domains = get_top_domains(traits)
    values_text = ", ".join(values) if values else "growth and balance"
    motive_text = ", ".join(motivations) if motivations else "inner satisfaction"

    if use_ai:
        prompt = f"""
        Act as a life coach and purpose mentor.
        Based on these traits: {traits},
        core values: {values_text},
        and motivations: {motive_text},
        write a warm and concise life purpose statement (1–2 sentences)
        followed by a short 7-day micro-goal plan.
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.warning(f"AI generation failed: {e}")
    
    # --- fallback
    purpose = f"Focus on {domains[0].lower()} while honoring your values of {values_text}."
    plan = "Start small this week: journal daily and take one step toward something that feels meaningful."
    return purpose + "\n\n" + plan

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Purpose Finder", page_icon="🌱", layout="centered")

st.title("🌱 Purpose Finder AI")
st.write("Discover your core motivations, values, and life direction using psychology-inspired insights.")

st.divider()

# -----------------------------
# Step 1: Personality Quiz
# -----------------------------
st.header("Step 1: Quick Personality Snapshot")
st.caption("Rate how much you agree with each statement (1 = Strongly Disagree, 5 = Strongly Agree)")

personality_questions = {
    "imagination": "I enjoy exploring new ideas and experiences.",
    "curiosity": "I often wonder about how things work or why people behave as they do.",
    "organized": "I like to keep things in order and plan ahead.",
    "responsibility": "I take my commitments seriously.",
    "outgoing": "I feel energized by being around people.",
    "energy": "I tend to be enthusiastic and talkative.",
    "helpful": "I go out of my way to help others.",
    "trust": "I usually see the best in people.",
    "stress": "I get upset easily under pressure.",
    "worry": "I spend a lot of time worrying about things."
}

responses = {}
cols = st.columns(2)
i = 0
for key, q in personality_questions.items():
    with cols[i % 2]:
        responses[key] = st.slider(q, 1, 5, 3)
    i += 1

# -----------------------------
# Step 2: Values
# -----------------------------
st.header("Step 2: Core Values")
st.caption("Select the values that feel most important to you.")
all_values = ["Growth", "Family", "Independence", "Creativity", "Security", "Helping Others", "Adventure", "Spirituality", "Learning", "Achievement"]
values = st.multiselect("Choose up to 5 values", all_values)

# -----------------------------
# Step 3: Motivation
# -----------------------------
st.header("Step 3: What Drives You?")
st.caption("Select what best describes your main motivations.")
motives = ["Autonomy (freedom, control over life)",
           "Competence (mastery, achievement)",
           "Relatedness (connection, belonging)",
           "Stability (security, comfort)",
           "Contribution (helping, impact)"]
motivations = st.multiselect("Your main motivators", motives)

# -----------------------------
# Step 4: Generate Purpose
# -----------------------------
if st.button("✨ Discover My Purpose"):
    traits = calculate_personality_score(responses)
    purpose_text = generate_purpose_statement(traits, values, motivations)

    st.success("Here’s your personalized purpose insight:")
    st.write(purpose_text)

    st.subheader("🔍 Summary Snapshot")
    df = pd.DataFrame(list(traits.items()), columns=["Trait", "Score"])
    st.dataframe(df, use_container_width=True)

    # Download results
    result_data = {
        "traits": traits,
        "values": values,
        "motivations": motivations,
        "purpose_text": purpose_text
    }

    st.download_button(
        "📥 Download My Purpose Report (JSON)",
        data=json.dumps(result_data, indent=2),
        file_name="purpose_report.json",
        mime="application/json"
    )

st.divider()
st.caption("Built with ❤️ using Streamlit & psychological insights. (c) 2025")

