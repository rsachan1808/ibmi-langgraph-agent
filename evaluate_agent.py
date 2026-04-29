import json
import time
from anthropic import Anthropic
from test_cases import TEST_CASES

# Import your working agent
from langgraph_agent import ask_agent

# Load API keys from config module
from config import load_keys
ANTHROPIC_KEY, VOYAGE_KEY = load_keys()

eval_client = Anthropic(api_key=ANTHROPIC_KEY)

# ── Method 1: Keyword check ───────────────────────────────
def evaluate_keywords(answer: str, test_case: dict) -> dict:
    answer_lower = answer.lower()

    must_contain     = test_case.get("must_contain", [])
    must_not_contain = test_case.get("must_not_contain", [])

    present = [
        kw for kw in must_contain
        if kw.lower() in answer_lower
    ]
    absent = [
        kw for kw in must_contain
        if kw.lower() not in answer_lower
    ]
    hallucinated = [
        kw for kw in must_not_contain
        if kw.lower() in answer_lower
    ]

    score = len(present) / len(must_contain) if must_contain else 1.0

    return {
        "score":        round(score, 2),
        "present":      present,
        "missing":      absent,
        "hallucinated": hallucinated,
        "passed":       score == 1.0 and len(hallucinated) == 0
    }

# ── Method 2: LLM as judge ────────────────────────────────
def evaluate_llm_judge(
    question: str,
    answer: str,
    test_case: dict
) -> dict:

    ideal = test_case.get("ideal_answer", "")

    prompt = f"""You are evaluating an AI agent's answer to a technical question.

Question: {question}

Agent answer: {answer}

Expected answer should cover: {ideal}

Evaluate on two criteria:
1. Correctness — is the information accurate?
2. Completeness — does it cover the key points?

Respond in JSON format only, no markdown fences, no other text:
{{
    "correctness_score": <0.0 to 1.0>,
    "completeness_score": <0.0 to 1.0>,
    "overall_score": <0.0 to 1.0>,
    "passed": <true or false>,
    "reasoning": "<one sentence>",
    "issues": "<specific problem, or empty string>"
}}"""

    response = eval_client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Remove first line (```json or ```) and last line (```)
        raw = "\n".join(lines[1:-1]).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "overall_score": 0,
            "passed":        False,
            "reasoning":     "Could not parse evaluation response",
            "raw":           raw
        }

# ── Run full evaluation ───────────────────────────────────
def run_evaluation():
    print("\n" + "="*60)
    print("AGENT EVALUATION REPORT")
    print("="*60)

    results      = []
    total_passed = 0

    for i, test_case in enumerate(TEST_CASES):
        question = test_case["question"]
        print(f"\nTest {i+1}: {question}")
        print("-" * 40)

        # Get agent answer
        t      = time.time()
        answer = ask_agent(question)
        elapsed = time.time() - t

        print(f"Answer: {answer[:150]}...")
        print(f"Time:   {elapsed:.2f}s")

        # Run all three evaluations
        keyword_result    = evaluate_keywords(answer, test_case)
        llm_result        = evaluate_llm_judge(question, answer, test_case)

        # Combine into overall result
        overall_passed = (
            keyword_result["passed"] and
            llm_result.get("passed", False)
        )

        if overall_passed:
            total_passed += 1
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"

        print(f"\nResults: {status}")
        print(f"  Keyword check:      {keyword_result['score']:.0%} "
              f"| Missing: {keyword_result['missing']} "
              f"| Hallucinated: {keyword_result['hallucinated']}")
        print(f"  LLM judge score:    {llm_result.get('overall_score', 'N/A'):.0%}")
        print(f"  LLM reasoning:      {llm_result.get('reasoning', 'N/A')}")

        if llm_result.get("issues"):
            print(f"  Issues:             {llm_result['issues']}")

        results.append({
            "question":   question,
            "answer":     answer,
            "time":       elapsed,
            "keyword":    keyword_result,
            "llm_judge":  llm_result,
            "passed":     overall_passed
        })

    # Summary
    print("\n" + "="*60)
    print(f"SUMMARY: {total_passed}/{len(TEST_CASES)} tests passed")
    print(f"Pass rate: {total_passed/len(TEST_CASES):.0%}")

    failed = [r for r in results if not r["passed"]]
    if failed:
        print(f"\nFailed tests:")
        for r in failed:
            print(f"  - {r['question']}")

    print("="*60)

    # Save results to file
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nFull results saved to evaluation_results.json")

    return results


if __name__ == "__main__":
    run_evaluation()

