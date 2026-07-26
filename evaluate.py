"""Retrieval evaluation.

Measures the index against a labelled question set instead of eyeballing a
few queries:

    python evaluate.py

Reports Recall@1, Recall@3, Recall@5 and MRR, then lists every miss so the
data or synonyms can be fixed. Questions are written the way a student would
actually type them, including vague ones and typos.
"""

import sys

from resources import build_index

# (question, resource that should surface)
CASES = [
    # Academic
    ("where can I get free tutoring", "Drop-In Tutoring"),
    ("i'm failing calculus", "Math Study Hall"),
    ("someone to look over my essay", "Writing Center"),
    ("I keep procrastinating and can't manage my time", "Academic Coaching"),
    ("group study sessions for a hard class", "Supplemental Instruction"),
    ("where do I take a placement exam", "Testing Services"),
    # Advising and records
    ("I need to talk to my advisor", "Academic Advising"),
    ("how do I register for classes", "Athena"),
    ("I need an official transcript", "Office of the Registrar"),
    ("thinking about switching majors", "Major Change and Exploration"),
    ("advising for business students", "Terry College Undergraduate Advising"),
    # Career
    ("help with my resume", "UGA Career Center"),
    ("how do I find an internship", "UGA Career Center"),
    ("where are internships posted", "Handshake"),
    ("looking for an on campus job", "Student Employment"),
    # Health and wellness
    ("i think i'm depressed", "Counseling and Psychiatric Services"),
    ("i need a therapist", "Counseling and Psychiatric Services"),
    ("i'm sick and need to see a doctor", "University Health Center"),
    ("where do I pick up a prescription", "University Health Center"),
    ("i've been drinking too much", "Fontaine Center"),
    ("cheap therapy options", "Aspire Clinic"),
    ("i want to talk to a dietitian", "Nutrition Services"),
    # Basic needs
    ("i can't afford groceries", "UGA Food Pantry"),
    ("i ran out of money for food", "UGA Food Pantry"),
    ("free shampoo and toothpaste", "Bulldog Basics"),
    ("my family had an emergency and I'm missing class", "Student Care and Outreach"),
    ("i have adhd and need accommodations", "Disability Resource Center"),
    # Money
    ("help paying tuition", "Office of Student Financial Aid"),
    ("where do I pay my bill", "Bursar's Office"),
    ("how do I apply for fafsa", "Office of Student Financial Aid"),
    ("i need help budgeting", "Student Money Management"),
    # Technology
    ("my wifi isn't working", "EITS Help Desk"),
    ("i forgot my password", "EITS Help Desk"),
    ("where do I see my assignments online", "eLearning Commons"),
    ("can I get microsoft office for free", "Software Downloads"),
    # Libraries and study
    ("somewhere to study with a group", "Miller Learning Center"),
    ("i need help finding sources for a paper", "Ask a Librarian"),
    ("where are the archives", "Special Collections Libraries"),
    # Campus life
    ("how do I join a club", "Center for Student Activities and Involvement"),
    ("i want to play intramural soccer", "Intramural Sports"),
    ("where is the gym", "Recreational Sports"),
    ("i want to volunteer", "Center for Leadership and Service"),
    ("how do I do undergraduate research", "CURO"),
    # Community
    ("i want to study abroad", "Office of Global Engagement"),
    ("support for international students", "International Student Life"),
    ("resources for lgbtq students", "LGBT Resource Center"),
    ("i'm a veteran using my gi bill", "Student Veterans Resource Center"),
    # Living
    ("problems with my roommate", "University Housing"),
    ("how do meal plans work", "Dining Services"),
    ("my landlord won't return my deposit", "Student Legal Services"),
    # Getting around
    ("what bus goes to campus", "Campus Transit"),
    ("i need a parking permit", "Parking Services"),
    ("getting home late at night safely", "Safe Ride"),
    # Conduct and equity
    ("i was accused of cheating", "Office of Student Conduct"),
    ("someone is harassing me", "Equal Opportunity Office"),
    # Typos — should still resolve
    ("tutorin for calculas", "Math Study Hall"),
    ("finacial aid", "Office of Student Financial Aid"),
    ("councelling services", "Counseling and Psychiatric Services"),
    ("libary study rooms", "Miller Learning Center"),
]


def main():
    index = build_index()

    ranks = []
    misses = []

    for question, expected in CASES:
        names = list(index.search(question, limit=5)["resource_name"])
        rank = names.index(expected) + 1 if expected in names else 0
        ranks.append(rank)
        if rank == 0:
            misses.append((question, expected, names[:3]))
        elif rank > 3:
            misses.append((question, f"{expected} (rank {rank})", names[:3]))

    total = len(ranks)
    recall_at = lambda k: sum(1 for r in ranks if 0 < r <= k) / total
    mrr = sum(1 / r for r in ranks if r) / total

    print(f"\nRetrieval evaluation — {total} labelled questions, {len(index)} resources\n")
    print(f"  Recall@1   {recall_at(1):.1%}")
    print(f"  Recall@3   {recall_at(3):.1%}")
    print(f"  Recall@5   {recall_at(5):.1%}")
    print(f"  MRR        {mrr:.3f}")

    if misses:
        print(f"\n  {len(misses)} question(s) worth a look:")
        for question, expected, got in misses:
            print(f"    {question!r}")
            print(f"      want: {expected}")
            print(f"      got:  {got}")
    else:
        print("\n  Every question returned its expected resource in the top 3.")

    print()
    # Recall@5 is the number that matters: the model sees the top 6, so a
    # resource landing anywhere in that window can still inform the answer.
    return 0 if recall_at(5) >= 0.90 else 1


if __name__ == "__main__":
    sys.exit(main())
