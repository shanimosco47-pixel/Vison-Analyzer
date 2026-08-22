"""LLM-assisted efflux timing: a bounded feasibility spike.

Status: **not wired into the main detector registry or the web layer.**
This package is intentionally standalone.  It exists to answer one question
- can a general-purpose multimodal model, called through a production API
(not an interactive chat session), read Zahn-cup efflux timing off real
footage accurately and honestly enough to be worth productionising - before
any of it touches the user-facing pipeline.

See ``diagnostics/llm_spike/DESIGN.md`` for the design, the evidence gates
this spike must clear, and what is/isn't implemented yet.
"""
