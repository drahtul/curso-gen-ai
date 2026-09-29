import json
from app import answer, TOOL_NAMES
from _eval_utils import print_header, print_table, asserts, normalize_numbers

with open("golden_set.json", encoding="utf-8") as f:
    GOLDEN_SET = json.load(f)

MAX_REASONABLE_STEPS = 4


# Las reglas de respuesta evalúan el resultado visible: palabras, restricciones
# de longitud y rechazo. Son deterministas, baratas y reproducibles, aunque no
# garantizan que el texto sea útil o factual.

def rule_required_keywords(case, result):
    """normalize_numbers() means a '1830' assertion still matches '1,830.90'.
    Without it the rule fails on a formatting choice, not on a wrong answer."""
    low = normalize_numbers(result["text"].lower())
    missing = [k for k in case["must_include"] if normalize_numbers(k.lower()) not in low]
    return not missing, f"missing {missing}" if missing else "ok"


def rule_forbidden_keywords(case, result):
    """asserts() ignores a forbidden string that the answer is denying, so
    "the discount is 15%, not 60%" is a pass and not a failure."""
    found = [k for k in case["must_not_include"] if asserts(result["text"], k)]
    return not found, f"asserted {found}" if found else "ok"


def rule_max_words(case, result):
    words = len(result["text"].split())
    return words <= case["max_words"], f"{words} words"


def rule_refusal(case, result):
    """Out-of-scope questions must produce exactly NO_DATA, nothing else."""
    if not case["expect_no_data"]:
        return True, "n/a"
    return result["text"].strip().upper().startswith("NO_DATA"), "did not refuse"


# Las reglas de traza evalúan el proceso: herramientas usadas y cantidad de
# pasos. Separar resultado y proceso permite detectar respuestas correctas que
# llegaron mediante un camino costoso o sin grounding.

def rule_expected_tools(case, result):
    """Every tool the case needs must actually appear in the trace."""
    expected = set(case["expected_tools"])
    used = set(result["tools_used"])
    if not expected:
        return True, "n/a"
    missing = expected - used
    return not missing, f"never called {sorted(missing)}" if missing else "ok"


def rule_grounded_in_tools(case, result):
    """No tool call at all means the answer came from the model's memory."""
    if case["expected_tools"]:
        return bool(result["tools_used"]), "answered without calling anything"
    return True, "n/a"


def rule_no_runaway(case, result):
    """A correct answer that took eight turns is still a cost incident."""
    return result["steps"] <= MAX_REASONABLE_STEPS, f"{result['steps']} steps"


ANSWER_RULES = [
    ("keywords", rule_required_keywords),
    ("forbidden", rule_forbidden_keywords),
    ("length", rule_max_words),
    ("refusal", rule_refusal),
]

TRACE_RULES = [
    ("tools", rule_expected_tools),
    ("grounded", rule_grounded_in_tools),
    ("steps", rule_no_runaway),
]

RULES = ANSWER_RULES + TRACE_RULES


def run_suite(variant: str):
    # El golden set combina casos normales, bordes, fuera de alcance y
    # adversariales; debe versionarse junto con el comportamiento esperado.
    print(f"\nRunning {len(GOLDEN_SET)} cases against variant {variant}\n")
    rows = []
    failures = []
    passed_cases = 0
    no_tool_calls = 0
    total_steps = 0

    for case in GOLDEN_SET:
        result = answer(case["question"], variant=variant)
        total_steps += result["steps"]
        no_tool_calls += not result["tools_used"]

        # must_include valida señales observables, no necesariamente una
        # respuesta completa; las búsquedas textuales pueden producir falsos
        # positivos o negativos.
        marks = []
        case_ok = True
        for name, rule in RULES:
            ok, detail = rule(case, result)
            marks.append("PASS" if ok else "FAIL")
            if not ok:
                case_ok = False
                failures.append((case["id"], name, detail, result))
        passed_cases += case_ok
        rows.append([case["id"], case["category"], *marks, "PASS" if case_ok else "FAIL"])

    print_table(["case", "category", *[n for n, _ in RULES], "result"], rows)
    print(f"\nCases passed: {passed_cases}/{len(GOLDEN_SET)}")
    print(f"Answers with no tool call at all: {no_tool_calls}/{len(GOLDEN_SET)}")
    print(f"Average LLM turns per question: {total_steps / len(GOLDEN_SET):.1f}")

    if failures:
        print("\n--- failures in detail ---")
        for case_id, rule_name, detail, result in failures:
            print(f"\n{case_id} / {rule_name}: {detail}")
            print(f"  tools:  {result['tools_used'] or 'none'}")
            print(f"  output: {result['text'][:150]}")
    return passed_cases


def show_tool_usage(variant: str):
    """How often each tool gets picked. A tool nobody calls is dead weight."""
    # La frecuencia ayuda a detectar herramientas innecesarias o casos que no
    # están cubiertos por el golden set, pero no demuestra que fueron elegidas
    # correctamente.
    print(f"\n--- tool usage, variant {variant} ---\n")
    counts = {name: 0 for name in TOOL_NAMES}
    for case in GOLDEN_SET:
        result = answer(case["question"], variant=variant)
        for name in result["tools_used"]:
            counts[name] = counts.get(name, 0) + 1
    print_table(["tool", "calls over the suite"], [[k, v] for k, v in counts.items()])


if __name__ == "__main__":
    print_header("Demo 3 - Rule-based evals on a golden set")
    print("A golden set of 15 cases: happy path, edge cases, out of scope, adversarial.")
    print("15 well-chosen cases beat 500 random ones.")
    print(f"\nRules on the answer: {[n for n, _ in ANSWER_RULES]}")
    print(f"Rules on the trace:  {[n for n, _ in TRACE_RULES]}")

    grounded = run_suite("B")
    loose = run_suite("A")

    print("\n" + "=" * 80)
    print(f"variant B (grounded): {grounded}/{len(GOLDEN_SET)} cases")
    print(f"variant A (loose):    {loose}/{len(GOLDEN_SET)} cases")
    print("=" * 80)

    show_tool_usage("B")

    print("\nWhat rules cannot tell you: whether the answer is actually helpful,")
    print("and whether the tool it picked was the *right* one for the question.")
