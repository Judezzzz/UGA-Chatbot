"""Behaviour checks for the index, retrieval, and safety triage.

Runnable without Streamlit:

    python test_resources.py

Retrieval *quality* is measured separately in evaluate.py; this file asserts
the things that must not silently break.
"""

import sys

from resources import (
    build_index,
    is_crisis,
    needs_context,
    official_link,
    resolve_query,
    retrieval_answer,
    stem,
    tokenise,
)

FAILURES = []


def check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} {detail}")
        FAILURES.append(label)


def test_index(index):
    print("\nIndex")
    frame = index.frame
    check("holds at least 50 resources", len(index) >= 50, f"(got {len(index)})")
    check("has the expected columns",
          {"category", "resource_name", "description", "keywords"} <= set(frame.columns))
    check("no blank names", (frame["resource_name"].str.strip() != "").all())
    check("no blank descriptions", (frame["description"].str.strip() != "").all())
    check("no blank keywords", (frame["keywords"].str.strip() != "").all())
    check("no duplicate names", not frame["resource_name"].duplicated().any())
    check("covers 10+ categories", len(index.categories) >= 10)
    check("every doc produced tokens", all(len(d) > 0 for d in index.docs))
    check("idf is populated", len(index.idf) > 200, f"(got {len(index.idf)})")


def test_text(index):
    print("\nText handling")
    check("stems plurals", stem("loans") == "loan")
    check("stems gerunds", stem("tutoring") == "tutor")
    check("stems -ies to -y", stem("libraries") == "library")
    check("leaves short words alone", stem("aid") == "aid")
    check("drops stopwords", tokenise("what are the") == [])
    check("keeps content words", "calculu" in " ".join(tokenise("calculus help")))


def test_retrieval(index):
    print("\nRetrieval")
    check("finds tutoring", "Drop-In Tutoring"
          in list(index.search("free tutoring")["resource_name"]))
    check("synonym reaches counseling",
          "Counseling and Psychiatric Services"
          in list(index.search("i need a therapist")["resource_name"]))
    check("typo still resolves",
          "Counseling and Psychiatric Services"
          in list(index.search("councelling")["resource_name"]))
    check("prefix morphology resolves",
          "Counseling and Psychiatric Services"
          in list(index.search("i am depressed")["resource_name"]))
    scores = list(index.search("food")["_score"])
    check("results are ranked descending", scores == sorted(scores, reverse=True))
    check("respects the limit", len(index.search("student help center", limit=3)) <= 3)

    empty = index.search("zzzzqqqq")
    check("nonsense returns nothing", empty.empty)
    check("nonsense answers honestly", "couldn't find anything" in retrieval_answer(empty))
    check("stopword-only returns nothing", index.search("what is the").empty)
    check("empty string returns nothing", index.search("").empty)


def test_crisis(index):
    print("\nCrisis triage")
    for phrase in [
        "i want to kill myself",
        "i've been thinking about suicide",
        "i don't want to live anymore",
        "i was sexually assaulted",
        "someone is stalking me and I'm in danger",
        "i've been hurting myself",
    ]:
        check(f"flags {phrase!r}", is_crisis(phrase))

    for phrase in [
        "where can I get free tutoring",
        "how do I pay my tuition bill",
        "my family had an emergency and I'm missing class",
        "i need help with my resume",
    ]:
        check(f"does not flag {phrase!r}", not is_crisis(phrase))

    crisis_rows = index.crisis_resources()
    check("crisis resources exist", len(crisis_rows) >= 3, f"(got {len(crisis_rows)})")
    check("988 is in the index", "988" in " ".join(crisis_rows["resource_name"]))


def test_followups():
    print("\nFollow-up handling")
    history = [
        {"role": "user", "content": "where can I get free tutoring"},
        {"role": "assistant", "content": "Try Drop-In Tutoring."},
    ]
    check("thin question needs context", needs_context("what about hours?"))
    check("bare question needs context", needs_context("cost?"))
    check("full question does not", not needs_context("how do I apply for financial aid"))
    check("resolves against last question",
          "tutoring" in resolve_query("what about hours?", history))
    check("leaves standalone questions alone",
          resolve_query("how do I find an internship", history)
          == "how do I find an internship")
    check("handles empty history", resolve_query("hours?", []) == "hours?")


def test_links():
    print("\nLinks")
    link = official_link("UGA Career Center")
    check("builds a search link", link.startswith("https://"))
    check("link has no spaces", " " not in link)


def main():
    index = build_index()
    test_index(index)
    test_text(index)
    test_retrieval(index)
    test_crisis(index)
    test_followups()
    test_links()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed.")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
