import re
from dotenv import load_dotenv

load_dotenv()

PRICING = {
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o": (2.50, 10.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost of a single request. Unknown models cost 0 so demos never crash."""
    price_in, price_out = PRICING.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def print_table(headers: list, rows: list) -> None:
    """Minimal aligned table so results are readable in a terminal."""
    cells = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in cells:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for row in cells:
        print(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


NEGATIONS_BEFORE = ("not ", "no ", "never", "n't", "rather than", "instead of", "isn't")

NEGATIONS_AFTER = (
    "isn't", "is not", "was not", "wasn't", "does not apply", "doesn't apply",
    "not applicable", "not correct", "is incorrect", "not available", "rather than",
)


def normalize_numbers(text: str) -> str:
    """Drop thousands separators so '1,830.90' matches a '1830' assertion.

    Another detector bug found by running the suite: the agent was right and the
    rule was comparing against a formatting choice.
    """
    return re.sub(r"(?<=\d),(?=\d)", "", text)


def asserts(text: str, needle: str) -> bool:
    """True when the text states the needle rather than denying it.

    Used by every check for forbidden content, so that an answer which corrects
    a false premise is not counted as repeating it.
    """
    low = normalize_numbers(text.lower())
    needle = normalize_numbers(needle.lower())
    start = 0
    while True:
        pos = low.find(needle, start)
        if pos == -1:
            return False

        before = low[max(0, pos - 30):pos]
        after = low[pos + len(needle):pos + len(needle) + 35]
        denied = (any(m in before for m in NEGATIONS_BEFORE)
                  or any(m in after for m in NEGATIONS_AFTER))
        if not denied:
            return True
        start = pos + 1


def print_header(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)
