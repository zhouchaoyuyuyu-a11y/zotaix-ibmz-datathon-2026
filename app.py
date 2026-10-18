"""ZOTAIX Intent-to-Brief: small, fully reproducible datathon prototype.

An explainable TF-IDF retrieval baseline + bounded engineering parameters.
Runs using Python's standard library on Linux, including IBM LinuxONE (s390x).
The model is trained on synthetic examples; never pass actual customer records.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any

EVENT_CODE = "SSF2026"

# Public, synthetic examples created for this prototype; no ZOTAIX production data.
TRAINING = [
    ("minimal", "clean modern minimal design for a business thank you gift"),
    ("minimal", "simple elegant refined bottle for a colleague with subtle colors"),
    ("minimal", "极简 商务 礼物 瓶子 清爽 简洁"),
    ("minimal", "understated corporate appreciation gift modern ivory and silver"),
    ("romantic", "romantic anniversary gift floral soft pink and warm memories"),
    ("romantic", "sweet birthday surprise for my partner with delicate flowers"),
    ("romantic", "浪漫 纪念日 礼物 温柔 粉色 花香"),
    ("romantic", "a joyful wedding keepsake with warm expressive colors"),
    ("heritage", "heritage handcrafted porcelain cultural gift with traditional patterns"),
    ("heritage", "local craftsmanship and cultural storytelling on a gift vessel"),
    ("heritage", "非遗 文化 陶瓷 手作 地方 纪念 礼物"),
    ("heritage", "traditional craftsmanship archive piece with warm earth tones"),
    ("bold", "bold experimental modern futuristic sculptural design gift"),
    ("bold", "dramatic black metallic futuristic 3D vessel for a launch"),
    ("bold", "先锋 未来 科技 个性 金属 雕塑 风格"),
    ("bold", "high contrast energetic product with vivid accents"),
]

COLORWAYS = {
    "minimal": {"main": "#F1EFEA", "accent": "#9DABB4", "finish": "satin"},
    "romantic": {"main": "#F1CDD3", "accent": "#B7878E", "finish": "translucent"},
    "heritage": {"main": "#D9C6AD", "accent": "#745744", "finish": "ceramic-inspired"},
    "bold": {"main": "#1D2332", "accent": "#D6A765", "finish": "metallic"},
}

OCCASION_KEYWORDS = {
    "anniversary": ["anniversary", "纪念日"],
    "birthday": ["birthday", "生日"],
    "wedding": ["wedding", "婚礼", "结婚"],
    "appreciation": ["thank you", "thanks", "appreciation", "感谢", "答谢"],
    "business": ["business", "corporate", "colleague", "商务", "企业"],
    "launch": ["launch", "发布", "新品"],
}

STOP_WORDS = {"a", "the", "for", "to", "my", "and", "with", "of", "a", "an", "in", "on", "is", "i"}


def tokens(s: str) -> list[str]:
    s = s.lower()
    words = [x for x in re.findall(r"[a-z]{2,}|\d+", s) if x not in STOP_WORDS]
    chinese = "".join(re.findall(r"[\u4e00-\u9fff]+", s))
    # Character bigrams allow short Chinese user expressions without an external tokenizer.
    bigrams = [chinese[i : i + 2] for i in range(len(chinese) - 1)]
    return words + bigrams


def private_text_redacted(s: str) -> str:
    s = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", s)
    s = re.sub(r"(?<!\d)(?:\+?\d[\d\s-]{8,}\d)(?!\d)", "[PHONE]", s)
    return s


@dataclass
class TfidfStyleModel:
    documents: list[dict[str, float]]
    idf: dict[str, float]

    @classmethod
    def train(cls, examples: list[tuple[str, str]]) -> "TfidfStyleModel":
        docs = [Counter(tokens(s)) for _, s in examples]
        df = Counter(t for d in docs for t in d)
        N = len(docs)
        idf = {t: 1 + math.log((N + 1) / (df[t] + 1)) for t in df}
        weighted = [{t: (1 + math.log(c)) * idf[t] for t, c in d.items()} for d in docs]
        return cls(weighted, idf)

    def infer(self, description: str) -> tuple[str, float, list[str]]:
        term_counts = Counter(tokens(description))
        q = {t: (1 + math.log(c)) * self.idf.get(t, 0) for t, c in term_counts.items()}
        qnorm = math.sqrt(sum(v * v for v in q.values())) or 1
        ranking = []
        for i, d in enumerate(self.documents):
            dnorm = math.sqrt(sum(v * v for v in d.values())) or 1
            sim = sum(v * d.get(t, 0) for t, v in q.items()) / (qnorm * dnorm)
            ranking.append((sim, i))
        best, index = max(ranking)
        style = TRAINING[index][0] if best > 0 else "minimal"
        overlap = sorted((set(q) & set(self.documents[index])), key=lambda t: -(q.get(t, 0)), reverse=False)
        return style, round(best, 3), overlap[:5]


MODEL = TfidfStyleModel.train(TRAINING)


def build_brief(user_intent: str) -> dict[str, Any]:
    if len(user_intent.strip()) < 5:
        raise ValueError("Please provide at least five characters describing the request.")
    if len(user_intent) > 3000:
        raise ValueError("Please shorten the request to 3,000 characters.")
    sanitized = private_text_redacted(user_intent.strip())
    low = sanitized.lower()
    style, confidence, evidence = MODEL.infer(sanitized)
    occasion = "unspecified"
    for label, variants in OCCASION_KEYWORDS.items():
        if any(v in low for v in variants):
            occasion = label
            break
    budget_match = re.search(r"(?:\$|usd\s*)(\d{2,6})\b", low)
    budget = int(budget_match.group(1)) if budget_match else None
    mentions_vessel = any(t in low for t in ["bottle", "vessel", "瓶", "器物", "容器"])
    slender = any(t in low for t in ["slender", "slim", "tall", "修长", "细长"])
    design_patch = {
        "shape": "vessel" if mentions_vessel else "packaged gift object",
        "height_ratio": 1.18 if slender else 1.0,
        "waist_indent": 0.12 if slender else 0.06,
        "edge_softness": 0.75 if style in {"romantic", "minimal"} else 0.35,
        "material_finish": COLORWAYS[style]["finish"],
        "main_color": COLORWAYS[style]["main"],
        "accent_color": COLORWAYS[style]["accent"],
    }
    assert 0.6 <= design_patch["height_ratio"] <= 1.6
    assert 0 <= design_patch["waist_indent"] <= 0.3
    assert 0 <= design_patch["edge_softness"] <= 1
    return {
        "event_code": EVENT_CODE,
        "prototype": "ZOTAIX Intent-to-Brief",
        "privacy": {"record_storage": "none", "contact_data": "redacted", "data_origin": "synthetic training examples"},
        "request_sha256_prefix": hashlib.sha256(sanitized.encode()).hexdigest()[:12],
        "request_preview_redacted": sanitized[:160],
        "inferred_occasion": occasion,
        "inferred_style": {"name": style, "similarity": confidence, "matched_tokens": evidence},
        "editable_design_patch": design_patch,
        "estimated_budget_usd": budget,
        "human_review_required": [
            "confirm recipients, style and intent",
            "verify materials, intellectual property, safety and regulatory fit",
            "obtain supplier quotation, MOQ, lead time and production approval",
        ],
        "next_step": "Human reviews and adjusts the structured brief before any manufacturing decision.",
        "runtime_target": "Python standard library; deployable on LinuxONE (s390x), not yet validated there",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="ZOTAIX datathon prototype: intent to editable brief")
    p.add_argument("--intent", help="free-text synthetic request")
    p.add_argument("--file", help="read a UTF-8 request from a local text file")
    args = p.parse_args()
    if bool(args.intent) == bool(args.file):
        p.error("Please specify exactly one of --intent or --file")
    if args.file:
        with open(args.file, encoding="utf-8") as handle:
            intent = handle.read()
    else:
        intent = args.intent
    print(json.dumps(build_brief(intent), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
