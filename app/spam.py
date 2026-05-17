try:
    from app.model_loader import load_model
except ModuleNotFoundError:
    from model_loader import load_model


def check_spam_rules(text: str) -> tuple[str, float]:
    text = text.lower().strip()
    if text == "":
        return "ham", 0.0

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
        "guarantee",
    ]

    hit = 0
    for kw in spam_keywords:
        if kw in text:
            hit += 1

    return "spam" if hit >= 2 else "ham", hit


def check_spam_ml(text: str) -> tuple[str, float]:
    text = text.strip()
    if text == "":
        return "ham", 0.0

    model = load_model()
    pred = model.predict([text])[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        classes = list(model.classes_)
        pred_index = classes.index(pred)
        score = float(proba[pred_index])
    else:
        score = 1.0

    return str(pred), score


def check_spam(text: str) -> tuple[str, float]:
    return check_spam_rules(text)
