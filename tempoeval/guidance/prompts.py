"""Temporal guidance generation prompts."""

# System prompt for all guidance generation
SYSTEM_PROMPT = "You output strict JSON only."

# Query-only guidance generation prompt
QUERY_GUIDANCE_PROMPT = """You are an expert in *temporal* information retrieval. Analyze ONLY the QUERY below and produce retrieval guidance and categories.

Goals:
1) Determine whether the query is temporal and classify its intent.
   - temporal_intent: one of ["when","duration","order","before_after","ongoing_status","period_definition","timeline","none"]
   - query_temporal_signals: phrases in the query indicating time (e.g., "in 1914", "during", "after", "first", "since", "today", "in the 18th century")
   - query_temporal_events: ONLY time-bound events (e.g., "Battle of Hastings", "signing of the Treaty of X", "election of Y"). Exclude generic actions unless anchored in time.
2) Provide a compact, specific plan to retrieve *temporal* evidence.
3) Identify time anchors, expected granularity, and sanity checks.

Allowed temporal reasoning classes (choose one primary, optional secondaries):
- "event_analysis_and_localization"
- "time_period_contextualization"
- "event_verification_and_authenticity"
- "sources_methods_and_documentation"
- "materials_artifacts_and_provenance"
- "trends_changes_and_cross_period"
- "origins_evolution_comparative_analysis"
- "historical_misinterpretation_or_reenactment"
- "causation_analysis"
- "artifact_verification"
- "historical_attribution_and_context"

CRUCIAL RULES:
- Use only the QUERY content (do NOT assume any passage).
- All arrays must be present even if empty. Use "" for missing strings.
- Return ONLY one JSON object with EXACT keys and value types below.

JSON schema to output:
{{
  "is_temporal_query": true,
  "temporal_intent": "when",
  "query_temporal_signals": ["..."],
  "query_temporal_events": ["..."],
  "query_summary": "summary of the query <=50 words",
  "temporal_reasoning_class_primary": "time_period_contextualization",
  "temporal_reasoning_class_secondary": ["materials_artifacts_and_provenance"],
  "retrieval_reasoning": "explanation of how to retrieve temporal evidence",
  "retrieval_plan": [
    {{"step": 1, "action": ".."}},
    {{"step": 2, "action": ".."}}
  ],
  "key_time_anchors": ["..."],
  "expected_granularity": "date",
  "quality_checks": ["cross-check dates from multiple sources", "prefer primary/authoritative sources"]
}}

QUERY:
{query}
"""

# Passage annotation prompt
PASSAGE_ANNOTATION_PROMPT = """You are an expert annotator for *temporal* information retrieval.

Given a QUERY and a PASSAGE, extract TEMPORAL info from the PASSAGE only:
- passage_temporal_signals: time cues (e.g., "in 1914", "during the 18th century", "after the treaty")
- passage_temporal_events: ONLY time-bound events (battle/treaty/reign/election/founding). Exclude non-temporal events.
- time_mentions: explicit or relative expressions (years, dates, centuries, eras, "after X", "during Y")
- time_scope_guess:
  - start_iso: ISO-like if visible (YYYY or YYYY-MM or YYYY-MM-DD), else ""
  - end_iso: same format; "" if none
  - granularity: one of ["date","month","year","decade","century","multicentury","unknown"]
- tense_guess: one of ["past","present","future","mixed","unknown"]
- confidence: 0.0–1.0

CRUCIAL RULES:
- Do NOT output any query-level fields here (no is_temporal_query, temporal_intent, etc.).
- Return empty lists (not null) when nothing is found.
- Return ONLY one JSON object with EXACT keys and value types below.

JSON schema to output:
{{
  "passage_temporal_signals": ["..."],
  "passage_temporal_events": ["..."],
  "time_mentions": ["..."],
  "time_scope_guess": {{"start_iso": "", "end_iso": "", "granularity": "unknown"}},
  "tense_guess": "past",
  "confidence": 0.0
}}

QUERY:
{query}

PASSAGE:
{passage}
"""
