# ZOTAIX Intent-to-Brief | IBM Z Datathon 2026

**Event code:** `SSF2026`  
**Challenge:** Wildcard: Beyond the Obvious  
**Status:** Independent, minimal prototype created during the 2026-10-17 datathon period. **IBM LinuxONE execution has not been verified**; integration is proposed, not claimed complete.

## Problem
A meaningful gift starts with an ambiguous intention, but producing a reviewable object requires explicit requirements. Hand-offs among creative tools and suppliers can fragment user intent, design choices, and practical constraints.

## Demonstration
This standalone, Python 3 standard-library implementation transforms a synthetic free-text gift request into a structured, editable product brief. It uses a small explainable TF-IDF retrieval model trained on **synthetic** descriptions. A constrained parameter patch keeps basic shape values inside documented ranges; it redacts contact details and flags human review for materials, compliance, pricing and manufacturing.

```sh
python3 app.py --file examples/request.txt
python3 -m unittest discover -s tests -v
```

Sample result fields: `inferred_style`, `inferred_occasion`, `editable_design_patch`, `human_review_required`, `runtime_target`.

## Architecture

`Synthetic request → PII redaction → TF-IDF style retrieval → Occasion/budget parsing → Bounded editable design patch → Human review`

This prototype **does not** send input to external APIs, read production customer data, or collect account credentials.

## Why IBM Z / LinuxONE?

The code relies exclusively on Python's standard library and is designed to run unchanged on the LinuxONE Community Cloud's Linux/s390x virtual server. Proposed next step: create a community-cloud instance using event code `SSF2026`, install Python 3, and run the exact CLI and tests. Mainframe/LinuxONE capabilities of interest include isolated compute, secure data handling and robust application operations. **No LinuxONE benchmark, deployment, or protected-enclave claim is made in this repository.**

## Relation to ZOTAIX

The original ZOTAIX platform (https://www.zotaix.cn) is a separate, pre-existing AI co-creation product. This entry is a new, independently written educational proof of concept **inspired by its problem domain**. No code, private assets, customer records, or production endpoints from the commercial repository are included. This is not the production ZOTAIX codebase, and the demo does not imply production capability.

## Limits and next steps

- TF-IDF trained on 16 simple synthetic prompts is an explainable ML baseline, not a validated production model.
- Occasion/budget extraction is rule-based, and Unicode tokenization is basic.
- It is not a manufacturing order or safety certification system.
- No live LinuxONE instance was provisioned or tested as part of these files.
- Future integration: human-approved JSON schema, persistent versioned project object, and tests on LinuxONE.

License: MIT. No user data or business secrets are included.
