def check_spam(text: str) -> tuple[str, int]:
    text = text.lower().strip()
    if text == "":
        return "ham", 0

    spam_keywords = [
        "free",
        "win",
        "winner",
        "prize",
        "click",
        "buy now",
        "urgent",
        "cash",
        "money",
        "offer",
        "deal",
        "bonus",
        "limited",
        "gurantee"
    ]

    hit = 0
    for kw in spam_keywords:
        if kw in text:
            hit += 1

    return "spam" if hit >= 2 else "ham", hit
