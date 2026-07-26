"""Checks for the resource index and retrieval, runnable without Streamlit:

    python test_resources.py
"""

import sys

from resources import load_resources, retrieval_answer, search, tokenise

FAILURES = []


def check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} {detail}")
        FAILURES.append(label)


def main():
    df = load_resources()

    print("\nIndex")
    check("loads at least 50 resources", len(df) >= 50, f"(got {len(df)})")
    check("has the expected columns",
          {"category", "resource_name", "description", "keywords"} <= set(df.columns))
    check("no blank resource names", (df["resource_name"].str.strip() != "").all())
    check("no blank descriptions", (df["description"].str.strip() != "").all())
    check("no duplicate resource names", not df["resource_name"].duplicated().any())
    check("covers 10+ categories", df["category"].nunique() >= 10,
          f"(got {df['category'].nunique()})")

    print("\nTokenising")
    check("drops stopwords and short words", tokenise("Where can I get help") == [])
    check("keeps meaningful terms", tokenise("free tutoring for calculus")
          == ["free", "tutoring", "calculus"])

    print("\nRetrieval")
    cases = [
        ("Where can I get free tutoring?", "Drop-In Tutoring"),
        ("I'm struggling with my mental health", "Counseling and Psychiatric Services"),
        ("How do I find an internship?", "UGA Career Center"),
        ("I can't afford groceries this month", "UGA Food Pantry"),
        ("I need help with my resume", "UGA Career Center"),
        ("how do I register for classes", "Athena"),
        ("how do I request a transcript", "Office of the Registrar"),
        ("my wifi isn't working", "EITS Help Desk"),
        ("I need a parking permit", "Parking Services"),
        ("where do I get accommodations for my adhd", "Disability Resource Center"),
        ("help paying tuition", "Office of Student Financial Aid"),
        ("I want to study abroad", "Office of Global Engagement"),
        ("somewhere to study with a group", "Miller Learning Center"),
    ]
    for question, expected in cases:
        names = list(search(question, df)["resource_name"])
        check(f"{question!r} -> {expected}", expected in names, f"(got {names[:3]})")

    print("\nEdge cases")
    empty = search("???", df)
    check("nonsense query returns nothing", empty.empty)
    check("nonsense query answers honestly",
          "couldn't find anything" in retrieval_answer(empty))
    check("stopword-only query returns nothing", search("what is the", df).empty)
    check("result count is capped", len(search("student help campus resource center", df)) <= 6)

    answer = retrieval_answer(search("free tutoring", df))
    check("answer names a resource", "Tutoring" in answer)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed.")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
