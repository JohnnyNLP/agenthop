"""Two-dimensional question mode schema: relationship x cognitive skill.

Crosses citation relationship (METHOD, RESULT, MOTIVE, GROUND) with
cognitive skill (retrieve, reason, contrast, assess) to produce diverse,
controlled question types instead of the LLM defaulting to easy metric lookups.

Usage:
    from question_modes import assign_modes, assign_dfs_modes, get_mode_instructions
"""

import random
from collections import Counter

from config import SKILL_DISTRIBUTION


# ── Section filtering by relationship ───────────────────────────────────────

RELATIONSHIP_SECTION_KEYWORDS = {
    "MOTIVE":  ["introduction", "motivation", "challenge", "limitation", "problem"],
    "GROUND":  ["related work", "background", "preliminary", "prior work", "overview", "survey"],
    "METHOD":  ["method", "approach", "architecture", "framework", "design", "implementation", "model"],
    "RESULT":  ["result", "experiment", "evaluation", "ablation", "analysis", "performance", "comparison", "benchmark"],
}


def filter_sections_by_relationship(
    sections: list[dict],
    relationship: str,
    min_chars: int = 500,
) -> list[dict]:
    """Filter paper sections to those relevant to the relationship type.

    Uses lowercase substring matching against section headers.
    Falls back to unfiltered sections if filtered content is too short.
    """
    keywords = RELATIONSHIP_SECTION_KEYWORDS.get(relationship)
    if not keywords or not sections:
        return sections

    filtered = [
        sec for sec in sections
        if any(kw in sec.get("header", "").lower() for kw in keywords)
    ]

    total_chars = sum(len(sec.get("text", "")) for sec in filtered)
    if total_chars < min_chars:
        return sections

    return filtered


# ── Schema definitions ──────────────────────────────────────────────────────

RELATIONSHIPS = {
    "METHOD": "The seed paper uses, adopts, or extends a technique from the cited work",
    "RESULT": "The seed paper compares against or cites a finding/benchmark from the cited work",
    "MOTIVE": "The seed paper cites a limitation, problem, or evidence that motivates its approach",
    "GROUND": "The seed paper cites the work as foundational context or theoretical basis",
}

SKILLS = {
    "retrieve":   "Find a specific stated fact (number, configuration, name, date)",
    "reason":     "Explain why — rationale, motivation, causal mechanism behind a choice or finding",
    "contrast":   "Identify differences, tradeoffs, or limitations compared to alternatives",
    "assess":     "Evaluate significance, validity, scope, or conditions of a claim",
    "synthesize": "Combine findings from multiple experiments, tables, or sections to draw a conclusion not stated in any single passage",
    "justify":    "Trace the evidential chain — identify what specific evidence supports or undermines a claim",
}

# context_type → relationship mapping
CONTEXT_TYPE_TO_RELATIONSHIP = {
    "evolution":             "METHOD",
    "provenance":            "METHOD",
    "comparison":            "RESULT",
    "data_reuse":            "RESULT",
    "limitation_resolution": "MOTIVE",
    "evidence_assessment":   "MOTIVE",
    "background_cite":       "GROUND",
    "unclassified":          "GROUND",
}


# ── Skill-specific instruction templates ────────────────────────────────────

_SKILL_INSTRUCTIONS = {
    "retrieve": (
        "Ask for a specific fact stated in the terminal paper — a number, "
        "configuration, name, or date — that CORRESPONDS to a specific claim "
        "or comparison made in the seed paper. The seed's claim should determine "
        "WHICH fact in the terminal is relevant. Without the seed context, a "
        "reader would not know which of the terminal paper's many facts to "
        "look for. The answer should be 1-2 sentences reporting this fact with "
        "enough context to be meaningful."
    ),
    "reason": (
        "Ask WHY the terminal paper made a specific choice or WHY a specific "
        "result occurred, where the RELEVANCE of that choice/result is "
        "established by something the seed paper says. The seed should frame "
        "which aspect of the terminal's reasoning matters. The answer should "
        "be 1-2 sentences explaining the rationale or causal mechanism — cite "
        "a specific constraint, observation, or prior result the authors used "
        "to justify their decision, not just a restatement of the goal. For "
        "this skill, the answer MAY connect observations from the paper into "
        "a causal explanation rather than quoting a single verbatim fact."
    ),
    "contrast": (
        "Ask how the terminal paper's approach/finding DIFFERS from an "
        "alternative discussed in the paper, where the seed paper establishes "
        "WHY this comparison matters or WHICH aspect to compare. The answer "
        "should be 1-2 sentences describing the specific difference or tradeoff."
    ),
    "assess": (
        "Ask about the SCOPE, VALIDITY, or CONDITIONS of a claim in the "
        "terminal paper, where the seed paper identifies WHICH claim to "
        "assess or provides a characterization to verify. The answer should "
        "be 1-2 sentences describing what the paper says about when/where/how "
        "well their claim holds."
    ),
    "synthesize": (
        "Ask a question whose answer REQUIRES combining findings from at least "
        "two different experiments, tables, or sections of the terminal paper, "
        "and where the seed paper's context determines what to synthesize. "
        "The answer should integrate these into a conclusion that no single "
        "paragraph or table states directly. Do NOT ask for a fact from one "
        "location — the question must force cross-referencing within the paper."
    ),
    "justify": (
        "Ask what specific evidence the terminal paper provides to support "
        "or undermine a claim that the SEED paper makes about the terminal's "
        "work. The answer should identify concrete experimental results, "
        "ablations, or analyses the authors present as evidence — not just "
        "restate the claim. A good justify question targets a claim where the "
        "evidence is spread across multiple results or where the support is "
        "partial/conditional."
    ),
}

_SKILL_INSTRUCTIONS_DFS = {
    "retrieve": (
        "Ask for specific facts stated across the target papers — numbers, "
        "configurations, names, or dates — that CORRESPOND to a specific "
        "claim or comparison the seed paper makes about them. The seed's "
        "framing should determine WHICH facts in each target are relevant. "
        "The answer should be 1-2 sentences combining these facts with enough "
        "context to be meaningful."
    ),
    "reason": (
        "Ask WHY the target papers made specific choices or WHY specific "
        "results occurred, where the seed paper establishes WHICH choices "
        "or results are relevant to compare. The answer should be 1-2 "
        "sentences combining the rationale or causal mechanisms — cite "
        "specific constraints, observations, or prior results each paper "
        "used to justify their decisions, not just restatements of goals. "
        "For this skill, the answer MAY connect observations from the papers "
        "into a causal explanation rather than quoting verbatim facts."
    ),
    "contrast": (
        "Ask how the target papers' approaches/findings DIFFER from each "
        "other, where the seed paper frames the DIMENSION of comparison "
        "(e.g., efficiency, accuracy, design philosophy). The answer should "
        "be 1-2 sentences describing the specific differences or tradeoffs."
    ),
    "assess": (
        "Ask about the SCOPE, VALIDITY, or CONDITIONS of claims across the "
        "target papers, where the seed paper identifies WHICH claims to "
        "assess or provides a characterization to verify. The answer should "
        "be 1-2 sentences describing what the papers say about when/where/how "
        "well their claims hold."
    ),
    "synthesize": (
        "Ask a question whose answer REQUIRES combining findings from "
        "different experiments, tables, or sections ACROSS the target papers, "
        "and where the seed paper's characterization of these works determines "
        "what to synthesize. The answer should integrate these into a "
        "comparative conclusion that neither paper states alone. Do NOT ask "
        "for isolated facts — the question must force cross-referencing "
        "across papers."
    ),
    "justify": (
        "Ask what collective evidence the target papers provide to support "
        "or challenge a specific claim made by the seed paper. The answer "
        "should identify concrete results from each target paper and assess "
        "whether they converge or diverge on the seed's claim."
    ),
}

# Relationship framing hints
_RELATIONSHIP_HINTS = {
    "METHOD": "Focus on a technique, model, or framework the seed paper adopts from the cited work.",
    "RESULT": "Focus on a benchmark result, experimental finding, or empirical comparison.",
    "MOTIVE": "Focus on a limitation, problem, or piece of evidence that motivated the seed paper's approach.",
    "GROUND": "Focus on a foundational assumption, theoretical basis, or established principle.",
}

_RELATIONSHIP_HINTS_DFS = {
    "METHOD": "Focus on techniques, models, or frameworks the seed paper adopts from the cited works.",
    "RESULT": "Focus on benchmark results, experimental findings, or empirical comparisons.",
    "MOTIVE": "Focus on limitations, problems, or evidence that motivated the seed paper's approach.",
    "GROUND": "Focus on foundational assumptions, theoretical bases, or established principles.",
}

# Skill-specific examples for BFS
_SKILL_EXAMPLES_BFS = {
    "retrieve": {
        "METHOD": (
            'Example: "TimeMixer claims to reduce forecasting error by 12% over a prior '
            'decomposition framework by replacing its fixed-level scheme. How many '
            'decomposition levels did that framework use in the configuration that '
            'TimeMixer\'s comparison targets?"'
        ),
        "RESULT": (
            'Example: "Mini-Splatting reports a 1.2 dB PSNR improvement over a point-based '
            'neural radiance field baseline on the synthetic NeRF dataset. What PSNR did '
            'that baseline report on synthetic NeRF in its own evaluation?"'
        ),
        "MOTIVE": (
            'Example: "The authors motivate their approach by claiming existing methods '
            'degrade catastrophically beyond 4K tokens, citing a specific study. What '
            'accuracy drop did that study measure at the 4K→8K transition that the '
            'authors reference?"'
        ),
        "GROUND": (
            'Example: "The paper assumes a convergence rate of O(1/sqrt(T)) based on '
            'a foundational optimization result. Under what specific convexity conditions '
            'did that foundational work prove this bound holds?"'
        ),
    },
    "reason": {
        "METHOD": (
            'Example: "TimeMixer builds on a frequency-domain decomposition method cited in '
            'its related work. Why did that method choose to decompose time series in the '
            'frequency domain rather than the time domain?"'
        ),
        "RESULT": (
            'Example: "The paper compares against a baseline that shows surprisingly strong '
            'performance on low-resource tasks. Why did the authors of that baseline attribute '
            'its effectiveness in the low-resource setting?"'
        ),
        "MOTIVE": (
            'Example: "The paper cites a study showing that standard attention fails on '
            'long documents as motivation. Why did that study conclude that attention '
            'mechanisms degrade with document length?"'
        ),
        "GROUND": (
            'Example: "The paper\'s theoretical framework rests on a prior proof about sample '
            'complexity. Why did that prior work argue that their bound was tight?"'
        ),
    },
    "contrast": {
        "METHOD": (
            'Example: "TimeMixer adapts a multiscale mixing strategy from a prior architecture. '
            'How does that prior architecture\'s mixing mechanism differ from standard '
            'self-attention in its treatment of temporal dependencies?"'
        ),
        "RESULT": (
            'Example: "The paper benchmarks against a model that reports strong results on '
            'long-horizon forecasting. How did that model\'s performance differ between '
            'short-horizon and long-horizon settings, and what tradeoff did the authors identify?"'
        ),
        "MOTIVE": (
            'Example: "The paper cites a limitation of sparse attention methods as motivation. '
            'What specific tradeoff did the cited work identify between computational '
            'efficiency and attention coverage?"'
        ),
        "GROUND": (
            'Example: "The paper grounds its approach in a classical approximation theorem. '
            'How did that theorem\'s assumptions differ from those of the competing '
            'universal approximation result the paper also discusses?"'
        ),
    },
    "assess": {
        "METHOD": (
            'Example: "TimeMixer extends a decomposition technique from a prior method. '
            'Under what conditions did the original authors report their decomposition '
            'approach was most effective, and when did it break down?"'
        ),
        "RESULT": (
            'Example: "The paper cites a baseline\'s results on the ETT benchmark. '
            'What limitations did the baseline\'s authors acknowledge about the '
            'generalizability of their reported results?"'
        ),
        "MOTIVE": (
            'Example: "The paper is motivated by a cited study on attention degradation. '
            'What scope conditions did that study place on their findings — for which '
            'model sizes and sequence lengths did the degradation hold?"'
        ),
        "GROUND": (
            'Example: "The theoretical foundation relies on a convergence guarantee from '
            'a prior work. What assumptions must hold for that guarantee to apply, and '
            'which did the authors flag as potentially unrealistic?"'
        ),
    },
    "synthesize": {
        "METHOD": (
            'Example: "The paper adapts a framework that reports separate evaluations '
            'on synthetic and real-world datasets using different backbone architectures. '
            'Combining these results, how does the choice of backbone interact with '
            'the dataset type — does the framework\'s advantage hold uniformly or '
            'does it depend on the backbone-dataset pairing?"'
        ),
        "RESULT": (
            'Example: "The cited baseline reports both per-category detection AP and '
            'overall inference speed across model variants. Combining the accuracy-speed '
            'tradeoff across variants, at what point does scaling the model yield '
            'diminishing returns in detection quality relative to latency cost?"'
        ),
        "MOTIVE": (
            'Example: "The motivating study presents both a quantitative analysis of '
            'failure rates and a qualitative error taxonomy. Combining these, which '
            'error categories account for the majority of the quantitative failures, '
            'and does the distribution shift across dataset domains?"'
        ),
        "GROUND": (
            'Example: "The foundational work proves bounds under two different settings: '
            'convex and non-convex objectives. Combining these results with the '
            'empirical convergence curves reported in the same paper, do the '
            'theoretical gaps between settings match the observed practical gaps?"'
        ),
    },
    "justify": {
        "METHOD": (
            'Example: "The paper adopts a cited method\'s claim that their architecture '
            'is more parameter-efficient than alternatives. What specific ablation '
            'results and comparisons does the cited work present to support this '
            'efficiency claim, and are there settings where the evidence is weaker?"'
        ),
        "RESULT": (
            'Example: "The cited baseline claims state-of-the-art on three benchmarks. '
            'For each benchmark, what is the margin over the previous best, and do the '
            'ablation studies confirm that the proposed component (rather than '
            'hyperparameter tuning or data) drives the improvement?"'
        ),
        "MOTIVE": (
            'Example: "The paper cites a study claiming that existing methods fail on '
            'long-tail distributions. What experimental evidence does that study present '
            'to support this claim — how many long-tail categories were tested, and '
            'how consistent was the degradation across them?"'
        ),
        "GROUND": (
            'Example: "The paper relies on a foundational claim that pre-training on '
            'diverse data improves downstream transfer. What ablations or controlled '
            'experiments does the cited work provide to isolate the effect of data '
            'diversity from data scale?"'
        ),
    },
}

# Skill-specific examples for DFS
_SKILL_EXAMPLES_DFS = {
    "retrieve": {
        "METHOD": (
            'Example: "DAB-DETR claims its dynamic anchor boxes unify two earlier query '
            'formulations — one using learned spatial priors and another using conditional '
            'cross-attention keys. How many learned queries does each approach use in the '
            'configuration that DAB-DETR\'s comparison table evaluates?"'
        ),
        "RESULT": (
            'Example: "The paper reports a 2.1 AP gap between two baselines on COCO '
            'val2017, attributing it to differences in query initialization. What AP '
            'scores did each baseline report in their own papers for the ResNet-50 '
            'backbone the seed uses for comparison?"'
        ),
        "MOTIVE": (
            'Example: "The paper cites two studies whose identified failure modes together '
            'motivate the proposed hybrid approach. What specific failure rates did each '
            'study report for the failure mode the seed references?"'
        ),
        "GROUND": (
            'Example: "The framework claims to satisfy bounds from two foundational '
            'results simultaneously. What convergence bounds did each establish under '
            'the assumptions the seed claims to meet?"'
        ),
    },
    "reason": {
        "METHOD": (
            'Example: "DAB-DETR discusses two approaches to object queries. '
            'Why did each approach choose its particular query representation strategy?"'
        ),
        "RESULT": (
            'Example: "The paper cites two models with contrasting performance profiles. '
            'Why did each attribute their respective strengths on different data regimes?"'
        ),
        "MOTIVE": (
            'Example: "Two cited studies identify different bottlenecks that motivate '
            'the current work. Why did each conclude their identified bottleneck was '
            'the primary limitation?"'
        ),
        "GROUND": (
            'Example: "The paper rests on two complementary theoretical results. '
            'Why did each prior work argue their framework was necessary?"'
        ),
    },
    "contrast": {
        "METHOD": (
            'Example: "DAB-DETR compares two earlier detection approaches: one that '
            'eliminates hand-designed anchors and another that introduces conditional '
            'queries. How do these two approaches differ in their treatment of object '
            'queries during cross-attention?"'
        ),
        "RESULT": (
            'Example: "The paper cites two models benchmarked on the same task. '
            'How do their results differ, and what tradeoff does each make between '
            'accuracy and efficiency?"'
        ),
        "MOTIVE": (
            'Example: "Two cited studies identify different failure modes. How do '
            'the failure conditions they identify differ, and what does each suggest '
            'as the root cause?"'
        ),
        "GROUND": (
            'Example: "The paper builds on two competing theoretical frameworks. '
            'How do their assumptions differ, and what are the practical implications '
            'of choosing one over the other?"'
        ),
    },
    "assess": {
        "METHOD": (
            'Example: "DAB-DETR discusses two query-based detection approaches. Under what '
            'conditions does each approach work best, and where does each break down?"'
        ),
        "RESULT": (
            'Example: "The paper cites two baselines with strong results. What scope '
            'limitations did each acknowledge about the generalizability of their findings?"'
        ),
        "MOTIVE": (
            'Example: "Two studies motivate the current work by identifying limitations. '
            'What conditions did each study place on their findings, and when might '
            'the identified limitations not apply?"'
        ),
        "GROUND": (
            'Example: "Two foundational results underpin the approach. What assumptions '
            'must hold for each, and which did their respective authors flag as '
            'potentially unrealistic?"'
        ),
    },
    "synthesize": {
        "METHOD": (
            'Example: "The seed paper draws on two cited methods that each report '
            'efficiency metrics under different hardware settings. Combining their '
            'reported throughput and memory usage, which method offers a better '
            'efficiency-accuracy tradeoff for resource-constrained deployment?"'
        ),
        "RESULT": (
            'Example: "Two baselines report results on overlapping benchmarks but '
            'different metrics (one reports AP, the other recall at fixed precision). '
            'Combining their results on the shared benchmark, what can be inferred '
            'about how precision-recall tradeoffs differ between the two approaches?"'
        ),
        "MOTIVE": (
            'Example: "Two cited studies each quantify a different failure mode of '
            'existing methods. Combining their failure analyses, what fraction of '
            'errors in current approaches can be attributed to architectural '
            'limitations versus data limitations?"'
        ),
        "GROUND": (
            'Example: "Two foundational works provide complementary theoretical '
            'guarantees under different assumptions. Combining their results, '
            'what is the effective bound when both sets of assumptions hold?"'
        ),
    },
    "justify": {
        "METHOD": (
            'Example: "The seed paper claims the two cited methods fail on long sequences. '
            'What evidence does each cited paper provide about its long-sequence behavior, '
            'and do their own results support or contradict the seed\'s characterization?"'
        ),
        "RESULT": (
            'Example: "The seed paper claims both cited baselines are outperformed on '
            'a specific benchmark. What do the baselines\' own ablation studies reveal '
            'about their performance ceiling — is the gap due to fundamental design '
            'limitations or suboptimal hyperparameters?"'
        ),
        "MOTIVE": (
            'Example: "The seed paper is motivated by limitations identified in two '
            'prior works. What strength of evidence does each study provide for its '
            'claimed limitation — large-scale experiments, theoretical arguments, or '
            'anecdotal examples?"'
        ),
        "GROUND": (
            'Example: "Two cited works provide the theoretical basis for the seed\'s '
            'approach. What empirical validation does each offer for their theoretical '
            'claims, and are there gaps between their theory and experiments?"'
        ),
    },
}


# ── Core functions ──────────────────────────────────────────────────────────

def infer_relationship(context_types: list[str]) -> str:
    """Infer citation relationship from chain metadata context_types.

    Maps each context type to a relationship code and returns the
    most common one (majority vote). Defaults to GROUND if empty.
    """
    if not context_types:
        return "GROUND"

    mapped = [
        CONTEXT_TYPE_TO_RELATIONSHIP.get(ct, "GROUND")
        for ct in context_types
    ]
    counts = Counter(mapped)
    return counts.most_common(1)[0][0]


def assign_skill(rng: random.Random, distribution: dict | None = None) -> str:
    """Weighted random skill assignment using the given RNG."""
    dist = distribution or SKILL_DISTRIBUTION
    skills = list(dist.keys())
    weights = [dist[s] for s in skills]
    return rng.choices(skills, weights=weights, k=1)[0]


def assign_modes(
    chains: list[dict],
    seed: int = 42,
    distribution: dict | None = None,
) -> list[tuple[str, str]]:
    """Assign (relationship, skill) pairs for a list of BFS chains.

    Relationship is inferred from each chain's context_types metadata.
    Skill is assigned by weighted random distribution.
    """
    rng = random.Random(seed)
    modes = []
    for chain in chains:
        context_types = chain.get("metadata", {}).get("context_types", [])
        rel = infer_relationship(context_types)
        skill = assign_skill(rng, distribution)
        modes.append((rel, skill))
    return modes


def assign_dfs_modes(
    groups: list[dict],
    seed: int = 42,
    distribution: dict | None = None,
) -> list[tuple[str, str]]:
    """Assign (relationship, skill) pairs for DFS sibling groups.

    Relationship is inferred from the dominant context type across
    all chains in the group.
    """
    rng = random.Random(seed)
    modes = []
    for group in groups:
        # Collect context types across all chains in the group
        all_ctypes = []
        for chain in group["chains"]:
            ctypes = chain.get("metadata", {}).get("context_types", [])
            all_ctypes.extend(ctypes)
        rel = infer_relationship(all_ctypes)
        skill = assign_skill(rng, distribution)
        modes.append((rel, skill))
    return modes


def get_mode_instructions(
    relationship: str,
    skill: str,
    question_type: str,
) -> str:
    """Build mode-specific instruction block for the user prompt.

    Args:
        relationship: One of METHOD, RESULT, MOTIVE, GROUND
        skill: One of retrieve, reason, contrast, assess, synthesize, justify
        question_type: "bfs" or "dfs"

    Returns a text block to embed in the === INSTRUCTIONS === section.
    """
    is_dfs = question_type == "dfs"

    skill_instr = (_SKILL_INSTRUCTIONS_DFS if is_dfs else _SKILL_INSTRUCTIONS)[skill]
    rel_hint = (_RELATIONSHIP_HINTS_DFS if is_dfs else _RELATIONSHIP_HINTS)[relationship]
    examples = (_SKILL_EXAMPLES_DFS if is_dfs else _SKILL_EXAMPLES_BFS)[skill][relationship]

    rel_desc = RELATIONSHIPS[relationship]
    skill_desc = SKILLS[skill]

    lines = [
        f"=== QUESTION MODE: {relationship} × {skill} ===",
        f"Relationship: {relationship} — {rel_desc}",
        f"Skill: {skill} — {skill_desc}",
        "",
        f"Relationship guidance: {rel_hint}",
        "",
        f"Skill constraint: {skill_instr}",
        "",
        examples,
    ]

    return "\n".join(lines)
