# PurposeFinder_Streamlit_MVP.py
# Streamlit MVP: "PurposeFinder" - a psychology-grounded app to help users discover life goals
# Features:
# - Short Big Five + Values + Self-Determination assessments
# - Scoring & interpretive summary
# - AI-driven goal statement + 3-step action plan (uses OpenAI API if available)
# - Save results locally (JSON) and allow user to download
# Requirements:
# pip install streamlit openai pandas numpy

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ---------------------------
# Configuration / Constants
# ---------------------------
APP_TITLE = "PurposeFinder — Discover Your Life Goals"
APP_SUBTITLE = "A gentle, psychology-backed journey to find your values, strengths, and an actionable first step."

# Short Big Five items (very reduced set for MVP). Responses 1-5
BIG5_ITEMS = {
    'Openness': [
        "I enjoy trying new things and new experiences.",
        "I prefer variety over routine."],
    'Conscientiousness': [
        "I pay attention to details and like to be organized.",
        "I finish tasks I start."],
    'Extraversion': [
        "I feel energized spending time with other people.",
        "I enjoy being the center of attention sometimes."],
    'Agreeableness': [
        "I usually consider other people's feelings.",
        "I prefer cooperation over competition."],
    'Neuroticism': [
        "I sometimes feel anxious or stressed.",
        "I get upset easily by small things."]
}

# Self-Determination (autonomy, competence, relatedness) -- short prompts
SDT_ITEMS = {
    'Autonomy': [
        "I have the freedom to make choices about how I spend my time."],
    'Competence': [
        "I feel capable of achieving goals I set for myself."],
    'Relatedness': [
        "I feel connected to people who care about me."]
}

# Values checklist (short list)
VALUES = [
    'Creativity', 'Security', 'Helping others', 'Achievement', 'Freedom', 'Family', 'Adventure', 'Learning', 'Wealth', 'Stability'
]

# Simple mapping from dimensions to suggested goal domains (fallback rule-based)
DIMENSION_TO_DOMAINS = {
    'Openness': ['Creative pursuits (writing, design, arts)', 'Research, learning, or travel'],
    'Conscientiousness': ['Project-based careers (engineering, operations, product)', 'Entrepreneurship that requires discipline'],
    'Extraversion': ['Community-oriented roles (sales, teaching, public-facing)', 'Events, hospitality, or advocacy'],
    'Agreeableness': ['Helping professions (counseling, social work, healthcare)', 'Team-based roles and volunteering'],
    'Neuroticism': ['Care-oriented roles with predictable structure', 'Focus on wellbeing, therapy, or coaching']
}

# ---------------------------
# Helper functions
# ---------------------------

def score_likert_responses(responses):
    # responses: dict of {item_text: int}
    # For big five: average per trait
    trait_scores = {}
    for trait, items in BIG5_ITEMS.items():
        vals = [responses.get(q, 3) for q in items]
        trait_scores[trait] = np.mean(vals)
    return trait_scores


def score_sdt(responses):
    sdt_scores = {}
    for trait, items in SDT_ITEMS.items():
        vals = [responses.get(q, 3) for q in items]
        sdt_scores[trait] = np.mean(vals)
    return sdt_scores


def values_summary(chosen_values):
    if not chosen_values:
        return 'No strong values selected.'
    return ', '.join(chosen_values)


def recommend_domains_from_scores(trait_scores):
    # pick top 2 traits by score
    sorted_traits = sorted(trait_scores.items(), key=lambda x: x[1], reverse=True)
    top_traits = [t for t, s in sorted_traits[:2]]
    domains = []
    for t in top_traits:
        domains.extend(DIMENSION_TO_DOMAINS.get(t, []))
    return domains


def build_ai_prompt(profile_summary):
    prompt = (
        "You are a compassionate career & life coach. Given the user's psychological profile, "
        "provide a short, inspiring life-purpose statement (1-2 sentences), followed by 3 practical starter goals (each 1 sentence), "
        "and then suggest a 7-day micro-plan to test one of the starter goals. Be concise and kind.\n\n"
        "User profile:\n" + profile_summary + "\n\nOutput format:\nPurpose:\n- <one-line>\nStarter Goals:\n1. ...\n2. ...\n3. ...\nMicro-plan (7 days):\nDay 1: ...\n...\nDay 7: ...\n"
    )
    return prompt


def call_openai(prompt, model='gpt-4o-mini', temperature=0.7, max_tokens=400):
    # model name can be adjusted by the user; defaults to a safe value
    if not OPENAI_AVAILABLE:
        raise RuntimeError('OpenAI library not installed')
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError('OPENAI_API_KEY not set')
    openai.api_key = api_key
    # Try ChatCompletion / chat API
    try:
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        text = resp['choices'][0]['message']['content'].strip()
        return text
    except Exception as e:
        st.error(f"OpenAI error: {e}")
        return None

# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title=APP_TITLE, layout='centered')
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

with st.expander('About this app (method):'):
    st.write(
        '- Short Big Five + Self-Determination measures to capture personality & intrinsic motivations.\n'
        '- Values checklist to highlight what the user cares about.\n'
        '- AI synthesis (OpenAI) to generate a warm, actionable purpose statement and tiny experiments.\n'
        '- Local save/export for privacy.'
    )

st.header('Step 1 — Tell us a bit about yourself')
name = st.text_input('Your name (optional)')
age = st.number_input('Age (optional)', min_value=13, max_value=120, value=30)

st.header('Step 2 — Quick personality check (choose how much each statement describes you)')
st.write('Use 1 (Strongly disagree) — 5 (Strongly agree)')

big5_responses = {}
for trait, items in BIG5_ITEMS.items():
    st.subheader(trait)
    for q in items:
        big5_responses[q] = st.slider(q, 1, 5, 3, key=q[:40])

st.header('Step 3 — Intrinsic motivations (Self-Determination)')
sdt_responses = {}
for trait, items in SDT_ITEMS.items():
    for q in items:
        sdt_responses[q] = st.slider(q, 1, 5, 3, key=q[:30]+"_sdt")

st.header('Step 4 — Your values')
st.write('Select up to 4 values that matter to you:')
selected_values = st.multiselect('Values', VALUES, default=[])[:4]

st.header('Step 5 — Quick reflection')
free_text = st.text_area('What activities make you lose track of time? (2-3 lines)')

if st.button('Generate my purpose summary'):
    # Score
    trait_scores = score_likert_responses(big5_responses)
    sdt_scores = score_sdt(sdt_responses)
    domains = recommend_domains_from_scores(trait_scores)
    values_text = values_summary(selected_values)

    # Build profile summary
    profile_lines = [f"Name: {name or 'Anonymous'}",
                     f"Age: {age}",
                     "\nBig Five scores (1-5):"]
    for t, s in trait_scores.items():
        profile_lines.append(f"- {t}: {s:.2f}")
    profile_lines.append('\nSelf-determination (1-5):')
    for t, s in sdt_scores.items():
        profile_lines.append(f"- {t}: {s:.2f}")
    profile_lines.append('\nTop suggested domains: ' + (', '.join(domains) if domains else 'General exploration'))
    profile_lines.append('\nValues: ' + values_text)
    profile_lines.append('\nActivities I love: ' + (free_text or 'Not provided'))

    profile_summary = '\n'.join(profile_lines)

    st.subheader('Your profile snapshot')
    st.code(profile_summary)

    # AI generation (if possible)
    ai_text = None
    if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
        with st.spinner('Generating personalized purpose & micro-plan with AI...'):
            prompt = build_ai_prompt(profile_summary)
            try:
                ai_text = call_openai(prompt)
            except Exception as e:
                st.warning('AI generation unavailable — will fall back to rule-based suggestions.')

    if ai_text:
        st.subheader('AI-generated purpose & micro-plan')
        st.markdown(ai_text)
    else:
        st.subheader('Suggested purpose & starter goals (rule-based fallback)')
        st.write('Purpose (example):')
        # create a small purpose sentence from top domains/values
        if domains:
            purpose = f"Focus on {domains[0].lower()} while honoring your values of {values_text}."
        else:
            purpose = f"Explore activities that combine your values ({values_text}) with your strengths."
        st.write('- ' + purpose)
        st.write('\nStarter Goals:')
        for i, dom in enumerate(domains[:3], start=1):
            st.write(f"{i}. Try a small project or class related to {dom} for 4 weeks.")
        if not domains:
            st.write('1. Try a 2-week hobby challenge: 3 sessions each week.')

        st.write('\n7-day micro-plan (example):')
        st.write('Day 1: Research one local class or online tutorial (30 mins).')
        st.write('Day 2: Schedule first 1-hour session this week.')
        st.write('Day 3: Try a 45-minute session; reflect for 10 minutes.')
        st.write('Day 4: Share your progress with a friend or journal about it.')
        st.write('Day 5: Try another short session; note what you enjoyed.')
        st.write('Day 6: Look for small ways to incorporate this into your week.')
        st.write('Day 7: Review and choose a next small commitment (e.g., weekly hour).')

    # Save results button
    results = {
        'timestamp': datetime.utcnow().isoformat(),
        'name': name,
        'age': age,
        'big5': trait_scores,
        'sdt': sdt_scores,
        'values': selected_values,
        'activities': free_text,
        'domains': domains,
        'ai_text': ai_text
    }
    st.markdown('---')
    if st.button('Download my results (JSON)'):
        st.download_button('Click to download', json.dumps(results, indent=2), file_name='purposefinder_results.json')

    st.success('Done — save your results and try small experiments for 1 week!')

# Footer / tips
st.markdown('---')
st.caption('Privacy tip: This MVP stores results locally in your browser session only. For production, connect secure user accounts and encrypted storage.')

st.markdown('### Developer notes')
st.write('- To enable AI generation, set environment variable OPENAI_API_KEY before running and install `openai` package.\n- Adjust prompt template in `build_ai_prompt()` for tone changes.\n- For deployment, consider Streamlit Cloud or a containerized approach.')