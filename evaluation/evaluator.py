from nlp.prompt_parser import parse_prompt
from vision.captioning import generate_caption
from vision.clip_utils import clip_similarity
from evaluation.metrics import compute_metrics
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_objects_from_caption(caption):
    doc = nlp(caption)
    return list(set([t.lemma_ for t in doc if t.pos_ == "NOUN"]))

def evaluate_sample(prompt, image_path):
    parsed_prompt = parse_prompt(prompt)
    caption = generate_caption(image_path)
    image_objects = extract_objects_from_caption(caption)

    metrics = compute_metrics(parsed_prompt["objects"], image_objects)
    clip_score = clip_similarity(prompt, image_path)

    adherence_score = (
        0.4 * metrics["recall"] +
        0.3 * metrics["precision"] +
        0.3 * max(clip_score, 0)
    ) * 100

    return {
        "prompt": prompt,
        "caption": caption,
        "clip_score": round(clip_score, 3),
        "adherence_score": round(adherence_score, 2),
        **metrics
    }
