import spacy

nlp = spacy.load("en_core_web_sm")

def parse_prompt(prompt: str):
    doc = nlp(prompt)

    objects = []
    attributes = []

    for token in doc:
        if token.pos_ == "NOUN":
            objects.append(token.lemma_)
        if token.pos_ == "ADJ":
            attributes.append(token.lemma_)

    return {
        "objects": list(set(objects)),
        "attributes": list(set(attributes))
    }
