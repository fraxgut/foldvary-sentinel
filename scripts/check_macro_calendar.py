import ast
from datetime import datetime
from pathlib import Path
import sys


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def main():
    sentinel_path = Path("sentinel.py")
    if not sentinel_path.exists():
        fail("sentinel.py not found.")

    tree = ast.parse(sentinel_path.read_text())
    events_dict = None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_macro_event":
            for inner in node.body:
                if isinstance(inner, ast.Assign):
                    for target in inner.targets:
                        if isinstance(target, ast.Name) and target.id == "events":
                            try:
                                events_dict = ast.literal_eval(inner.value)
                            except Exception as exc:
                                fail(f"Unable to parse events dict: {exc}")
            break

    if not events_dict:
        fail("Macro calendar events dict not found.")

    years = []
    for key in events_dict.keys():
        try:
            years.append(int(key.split("-")[0]))
        except Exception:
            continue

    if not years:
        fail("No date keys found in macro calendar.")

    max_year = max(years)
    current_year = datetime.now().year
    if current_year > max_year:
        fail(f"Macro calendar out of date (last year {max_year}, current {current_year}).")

    print(f"PASS: macro calendar covers through {max_year}.")


if __name__ == "__main__":
    main()
