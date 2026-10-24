"""WebShop skill routing for the debug loop.

Domain skill comes only from the original task description: word-boundary
match, top-1 category. general_skills is always kept. Each step then keeps
only the general rules for its page type. Step hits are not unioned into
extra domain skills.
"""

from __future__ import annotations

import re


# Phrases the stock keyword list misses. All map onto an existing category.
EXTRA_TASK_KEYWORDS = {
    "home_decor": ["desk mat", "standing mat", "kitchen mat", "floor mat", "desk", "table"],
}

PAGE_ROUTES = {
    "search": {
        "name": "query formulation",
        "rules": (1, 2, 14),
    },
    "results": {
        "name": "scan/click candidate",
        "rules": (3, 9, 12),
    },
    "product": {
        "name": "variant/price/attribute verification",
        "rules": (4, 5, 6, 7, 8, 13),
    },
}


def _keyword_pattern(keyword: str) -> re.Pattern:
    """Match the keyword as its own word, plus a trailing plural s.

    A left boundary stops ``heel`` from matching inside ``wheels``. The
    optional ``s`` still matches ``shoe`` / ``shoes``.
    """
    body = re.escape(keyword.lower().strip())
    return re.compile(rf"(?<![a-z0-9]){body}s?(?![a-z0-9])")


def keyword_hits(text: str, keywords: list[str]) -> list[str]:
    lowered = text.lower()
    hits = []
    for keyword in keywords:
        if keyword and _keyword_pattern(keyword).search(lowered):
            hits.append(keyword)
    return hits


def merged_keywords(task_keywords: dict) -> dict[str, list[str]]:
    merged = {}
    for task_type, keywords in task_keywords.items():
        extra = EXTRA_TASK_KEYWORDS.get(task_type, [])
        seen = []
        for keyword in list(extra) + list(keywords or []):
            if keyword and keyword not in seen:
                seen.append(keyword)
        merged[task_type] = seen
    return merged


def match_domain(task_text: str, task_keywords: dict) -> dict:
    """Pick at most one domain skill from the task description."""
    best_name = None
    best_hits: list[str] = []
    best_key = None
    for task_type, keywords in merged_keywords(task_keywords).items():
        hits = keyword_hits(task_text, keywords)
        if not hits:
            continue
        earliest = min(task_text.lower().find(hit.lower()) for hit in hits)
        longest = max(len(hit) for hit in hits)
        rank = (len(hits), longest, -earliest)
        if best_key is None or rank > best_key:
            best_name = task_type
            best_hits = hits
            best_key = rank
    return {"domain_skill": best_name, "domain_keywords": best_hits}


def classify_page(observation: str) -> str:
    text = (observation or "").lower()
    if "total results" in text:
        return "results"
    if "buy now" in text or "description" in text or "features" in text:
        return "product"
    return "search"


def slice_numbered_rules(text: str, rule_ids: tuple[int, ...] | list[int]) -> str:
    matches = list(re.finditer(r"(?m)^(\d+)\.\s", text))
    wanted = set(rule_ids)
    chunks = []
    for index, match in enumerate(matches):
        rule_id = int(match.group(1))
        if rule_id not in wanted:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        chunks.append(text[match.start() : end].strip())
    return "\n\n".join(chunks)


def resolve_step(provider, *, task_text: str, observation: str, skill_all: bool = False) -> dict:
    """Build the skill text for one step. Does not look at other steps."""
    if skill_all:
        resolved = provider.resolve(prompt_text=task_text or observation)
        return {
            "domain_skill": None,
            "domain_keywords": [],
            "page_type": classify_page(observation),
            "page_route": None,
            "page_rules": [],
            "selected_skills": list(resolved["selected_skills"]),
            "matched_tasks": list(resolved["matched_tasks"]),
            "skill_text": resolved["skill_text"],
            "keyword_skills": list(resolved["selected_skills"]),
        }

    domain = match_domain(task_text, provider.task_keywords)
    page_type = classify_page(observation)
    route = PAGE_ROUTES[page_type]
    general_full = provider.skill_contents.get("general_skills", "")
    general_slice = slice_numbered_rules(general_full, route["rules"])
    general_text = (
        "### GENERAL SKILLS ###\n"
        f"Page route: {page_type} -> {route['name']}\n\n"
        f"{general_slice}"
    ).strip()

    selected = ["general_skills"]
    parts = [general_text]
    matched = []
    domain_name = domain["domain_skill"]
    if domain_name and domain_name in provider.skill_contents:
        selected.append(domain_name)
        matched.append(domain_name)
        parts.append(provider.skill_contents[domain_name])

    return {
        "domain_skill": domain_name,
        "domain_keywords": list(domain["domain_keywords"]),
        "page_type": page_type,
        "page_route": route["name"],
        "page_rules": list(route["rules"]),
        "selected_skills": selected,
        "matched_tasks": matched,
        "general_text": general_text,
        "skill_text": "\n\n".join(parts),
        "keyword_skills": list(selected),
    }
