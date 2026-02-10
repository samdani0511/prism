def compute_metrics(prompt_objs, image_objs):
    prompt_set = set(prompt_objs)
    image_set = set(image_objs)

    matched = prompt_set & image_set
    missing = prompt_set - image_set
    extra = image_set - prompt_set

    precision = len(matched) / max(len(image_set), 1)
    recall = len(matched) / max(len(prompt_set), 1)

    hallucination_rate = len(extra) / max(len(image_set), 1)
    missing_rate = len(missing) / max(len(prompt_set), 1)

    return {
        "matched": list(matched),
        "missing": list(missing),
        "extra": list(extra),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "hallucination_rate": round(hallucination_rate, 3),
        "missing_rate": round(missing_rate, 3)
    }
