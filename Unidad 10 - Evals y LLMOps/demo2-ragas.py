import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)

from _eval_utils import print_header, print_table
from app import EMBEDDING_MODEL, answer

load_dotenv()

with open("golden_set.json", encoding="utf-8") as f:
# Solo se evalúan casos con herramientas porque las metricas de contexto
# necesitan evidencia recuperada para poder comparar la respuesta.
    GOLDEN_SET = [c for c in json.load(f) if c["expected_tools"]]

GRADER_MODEL = "gpt-4.1-nano"
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

RETRY_ATTEMPTS = 6
RETRY_BASE_DELAY_S = 3.0


def trace_to_contexts(trace: list) -> list:
    # Ragas recibe textos como contexto. Esta conversion compacta cada llamada
    # y su resultado, aunque no equivale necesariamente a documentos RAG puros.
    return [
        f"{step['tool']}({', '.join(f'{k}={v!r}' for k, v in step['args'].items())}) -> {step['result']}"
        for step in trace
    ]


async def with_retries(coro_fn, *args, **kwargs):
    """Retry on a 429 with exponential backoff; anything else raises immediately."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as exc:
            is_rate_limit = "rate_limit" in str(exc).lower() or "429" in str(exc)
            if not is_rate_limit or attempt == RETRY_ATTEMPTS - 1:
                raise
            await asyncio.sleep(RETRY_BASE_DELAY_S * (2 ** attempt))


async def score_case(metrics: dict, question: str, response_text: str, reference: str, contexts: list) -> dict:
    """Every metric for one case, fired concurrently instead of one round-trip each."""
    # Faithfulness mide apoyo en el contexto, no verdad factual global.
    # Relevancy compara respuesta y pregunta; precision/recall evalúan la
    # utilidad del contexto respecto de la referencia. Todas usan un LLM juez
    # y por eso pueden variar y tienen coste adicional.
    faithfulness, relevancy, precision, recall = await asyncio.gather(
        with_retries(
            metrics["faithfulness"].ascore,
            user_input=question, response=response_text, retrieved_contexts=contexts,
        ),
        with_retries(
            metrics["answer_relevancy"].ascore,
            user_input=question, response=response_text,
        ),
        with_retries(
            metrics["context_precision"].ascore,
            user_input=question, reference=reference, retrieved_contexts=contexts,
        ),
        with_retries(
            metrics["context_recall"].ascore,
            user_input=question, retrieved_contexts=contexts, reference=reference,
        ),
    )
    return {
        "faithfulness": faithfulness.value,
        "answer_relevancy": relevancy.value,
        "context_precision": precision.value,
        "context_recall": recall.value,
    }


async def score_all(metrics: dict, scored_inputs: list) -> list:
    return [await score_case(metrics, **inputs) for inputs in scored_inputs]


def run_ragas(variant: str, metrics: dict) -> tuple:
    # La variante A puede responder desde la memoria del modelo; B exige
    # grounding mediante herramientas. La comparación evalua el sistema, no
    # solo la redaccion final.
    print(f"\nRunning {len(GOLDEN_SET)} cases against variant {variant}...")

    runs = []
    for case in GOLDEN_SET:
        result = answer(case["question"], variant=variant)
        contexts = trace_to_contexts(result["trace"]) or ["NO_TOOL_CALLED"]
        runs.append({
            "case": case,
            "result": result,
            "contexts": contexts,
        })

    scores = asyncio.run(score_all(metrics, [
        {
            "question": r["case"]["question"],
            "response_text": r["result"]["text"],
            "reference": r["case"]["reference"],
            "contexts": r["contexts"],
        }
        for r in runs
    ]))
    for run, score in zip(runs, scores):
        run["scores"] = score

    rows = [
        [r["case"]["id"], r["case"]["category"], *[f"{r['scores'][m]:.2f}" for m in METRIC_NAMES]]
        for r in runs
    ]
    print_table(["case", "category", *METRIC_NAMES], rows)

    averages = {}
    for m in METRIC_NAMES:
        # Se excluyen nan porque no hay evidencia evaluable; el promedio no
        # debe ocultar los peores casos, que se muestran mas abajo.
        values = [r["scores"][m] for r in runs if r["scores"][m] == r["scores"][m]]
        skipped = len(runs) - len(values)
        averages[m] = sum(values) / len(values) if values else float("nan")
        if skipped:
            print(f"  ({m}: {skipped} case(s) returned nan - nothing checkable in the answer - excluded from the average)")

    return averages, runs


def show_worst(runs: list, metric: str, n: int = 2):
    """The lowest-scoring cases on one metric, to read instead of trusting the average."""
    worst = sorted(runs, key=lambda r: r["scores"][metric])[:n]
    print(f"\n--- lowest {metric} ---")
    for r in worst:
        print(f"  [{r['scores'][metric]:.2f}] {r['case']['question']}")
        print(f"    answer:  {r['result']['text'][:150]}")
        print(f"    context: {r['contexts'][0][:150]}")


if __name__ == "__main__":
    print_header("Demo 2 - RAG metrics computed by Ragas")
    print(f"Grader model: {GRADER_MODEL} | metrics: {METRIC_NAMES}")
    print(f"Cases with a tool in the loop: {len(GOLDEN_SET)}/16 "
          "(3 out-of-scope, no-tool cases are skipped - no context to score)")

    llm = llm_factory(GRADER_MODEL, client=AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60))
    embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL)

    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecisionWithReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }

    averages_a, runs_a = run_ragas("A", metrics)
    averages_b, runs_b = run_ragas("B", metrics)

    print("\n" + "=" * 80)
    print(f"{'metric':<40}{'A (loose)':>15}{'B (grounded)':>15}")
    print("-" * 80)
    for m in METRIC_NAMES:
        print(f"{m:<40}{averages_a[m]:>15.3f}{averages_b[m]:>15.3f}")
    print("=" * 80)

    show_worst(runs_a, "faithfulness")

    print("\nWhat this run does not tell you: whether a low context_precision/recall on")
    print("check_stock or store_policy cases is a real retrieval problem or just this demo's")
    print("choice to treat a lookup's whole output as a single 'context'. For the one real")
    print("retriever (search_catalog), those two numbers mean what Ragas' docs say they mean.")
