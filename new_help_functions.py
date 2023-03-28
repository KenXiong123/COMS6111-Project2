import spacy
from collections import defaultdict
from spanbert import special_tokens
import openai
import ast

spacy2bert = {
    "ORG": "ORGANIZATION",
    "PERSON": "PERSON",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "DATE": "DATE"
}

bert2spacy = {
    "ORGANIZATION": "ORG",
    "PERSON": "PERSON",
    "LOCATION": "LOC",
    "CITY": "GPE",
    "COUNTRY": "GPE",
    "STATE_OR_PROVINCE": "GPE",
    "DATE": "DATE"
}


def get_entities(sentence):
    """
    Extracts named entities from a given sentence using spaCy's named entity recognizer and maps them to corresponding entity types from BERT's ontology
    :param sentence: The sentence to extract entities from
    :return: A list of tuples containing the named entity and its corresponding BERT entity type
    """
    return [(e.text, spacy2bert[e.label_]) for e in sentence.ents if e.label_ in spacy2bert]


def extract_relations_spbt(doc, spanbert, entities_of_interest=None, conf=0.7, target_relation='no_relation'):
    """
    Extracts relations between named entities in a given document using a pre-trained SpanBERT model
    :param doc: The document to extract relations from
    :param spanbert: The pre-trained SpanBERT model to use for relation extraction
    :param entities_of_interest: A list of entity types to extract relations between. If not specified, all entity types will be used
    :param conf: The confidence threshold to use for relation extraction
    :param target_relation: The type of relation to extract. If not specified, all relations will be extracted
    :return: A dictionary containing the extracted relations, with each relation represented as a tuple containing the subject, relation, and object
    """
    num_sentences = len([s for s in doc.sents])
    print("\tExtracted {} sentences. Processing each sentence one by one to check for presence of right pair of named "
          "entity types; if so, will run the second pipeline ...".format(num_sentences))
    res = defaultdict(int)
    count = 1
    annot_count = 0
    for sentence in doc.sents:
        if count % 5 == 0:
            print(f"\tProcessed {count} / {num_sentences} sentences")
        count += 1
        entity_pairs = create_entity_pairs(sentence, entities_of_interest)
        examples = []
        if len(entity_pairs) == 0:
            continue
        for ep in entity_pairs:
            if "SUBJ=%s" % ep[1][1] in special_tokens and "OBJ=%s" % ep[2][1] in special_tokens:
                examples.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
            if "SUBJ=%s" % ep[2][1] in special_tokens and "OBJ=%s" % ep[1][1] in special_tokens:
                examples.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})
        if len(examples) == 0:
            continue
        preds = spanbert.predict(examples)

        for ex, pred in list(zip(examples, preds)):
            relation = pred[0]
            if relation != target_relation:
                continue
            else:
                annot_count += 1
                break
        for ex, pred in list(zip(examples, preds)):
            relation = pred[0]
            if relation != target_relation:
                continue
            print("\n\t\t=== Extracted Relation ===")
            print("\t\tInput tokens: {}".format(ex['tokens']))
            subj = ex["subj"][0]
            obj = ex["obj"][0]
            confidence = pred[1]
            print("\t\tOutput Confidence: {:.7f} ; Subject: {} ; Object: {}".format(confidence, subj, obj))

            if confidence > conf:
                if res[(subj, relation, obj)] < confidence:
                    res[(subj, relation, obj)] = confidence
                    print("\t\tAdding to set of extracted relations")
                else:
                    print("\t\tDuplicate with lower confidence than existing record. Ignoring this.")
            else:
                print("\t\tConfidence is lower than threshold confidence. Ignoring this.")
            print("\t\t==========")
    print(f"\tExtracted annotations for  {annot_count}  out of total  {num_sentences}  sentences")

    return res


def extract_relations_gpt(doc, openai, entities_of_interest=None, target_relation='no_relation'):
    """
    Extracts relations between named entities in a given document using OpenAI's GPT-3 language model
    :param doc: The document to extract relations from
    :param openai: The OpenAI API object to use for relation extraction
    :param entities_of_interest: A list of entity types to extract relations between. If not specified, all entity types will be used
    :param target_relation: The type of relation to extract. If not specified, all relations will be extracted
    :return: A dictionary containing the extracted relations, with each relation represented as a tuple containing the subject, relation, and object.
    """
    num_sentences = len([s for s in doc.sents])
    print("\tExtracted {} sentences. Processing each sentence one by one to check for presence of right pair of named "
          "entity types; if so, will run the second pipeline ...".format(num_sentences))
    res = defaultdict(int)
    count = 1
    annot_count = 0
    for sentence in doc.sents:
        if count % 5 == 0:
            print(f"\tProcessed {count} / {num_sentences} sentences")
        count += 1
        etypes = [s.ent_type_ for s in sentence]
        for ent in entities_of_interest[1:]:
            if bert2spacy[entities_of_interest[0]] in etypes and bert2spacy[ent] in etypes:
                prompt_text = """ Given a sentence, extract all the instances of the following relationship as possible:
                    relationship type: {}
                    output [Subject: {}, relationship type, Object: {}]
                    example output for all possible relationships:["Jeff Bezos", "Schools_Attended", "Princeton University"], 
                    ["Alec Radford", "Work_For", "OpenAI"], ["Mariah Carey", "Live_In", "New York City"], 
                    ["Jensen Huang", "Top_Member_Employees", "Nvidia"] 
                    sentence: {}  your answer must be a list and all elements types in the list must be string!"""\
                    .format(target_relation, entities_of_interest[0], entities_of_interest[1], sentence)

                model = 'text-davinci-003'
                max_tokens = 100
                temperature = 0.1
                top_p = 1
                frequency_penalty = 0
                presence_penalty = 0

                response_text = get_openai_completion(prompt_text, model, max_tokens, temperature, top_p,
                                                      frequency_penalty, presence_penalty)

                if len(response_text) != 0:
                    response_text = response_text[response_text.find("["):]
                    try:
                        response_text = ast.literal_eval(response_text)
                    except:
                        pass
                    for result in response_text:
                        if len(result) == 3:
                            if result[1] == target_relation:
                                annot_count += 1
                                break
                    for result in response_text:
                        if len(result) == 3:
                            print("\n\t\t=== Extracted Relation ===")
                            print("\t\tSentence: {}".format(sentence))
                            print("\t\tSubject: {} ; Object: {} ;".format(result[0], result[2]))
                            subj = result[0]
                            obj = result[2]
                            if res[(subj, result[1], obj)] == 0:
                                res[(subj, result[1], obj)] = 1
                                print("\t\tAdding to set of extracted relations.")
                            else:
                                print("\t\tDuplicate. Ignoring this.")
                            print("\t\t==========")
    print(f"\tExtracted annotations for  {annot_count}  out of total  {num_sentences}  sentences")

    return res


def get_openai_completion(prompt, model, max_tokens, temperature = 0.2, top_p = 1, frequency_penalty = 0, presence_penalty =0):
    """
    Uses the OpenAI API to generate text completion based on the given prompt and parameters
    :param prompt: The text prompt to be used for the text generation
    :param model: The name of the OpenAI model to be used for the text generation
    :param max_tokens: The maximum number of tokens to be generated by the API
    :param temperature: A value controlling the randomness of the generated text
    :param top_p: A value controlling the diversity of the generated text
    :param frequency_penalty: A value controlling the repetition of the generated text
    :param presence_penalty: A value controlling the presence of certain words in the generated text
    :return: The generated text completion
    """
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    response_text = response['choices'][0]['text']
    return response_text

def create_entity_pairs(sents_doc, entities_of_interest, window_size=40):
    """
    Extracts entity pairs from a spaCy Sentence object based on the given entities of interest and window size
    :param sents_doc: A spaCy Sentence object to be analyzed
    :param entities_of_interest: A list of entities of interest to be extracted from the Sentence object
    :param window_size: The maximum window size to be used for entity pair extraction
    :return: A list of extracted entity pairs in the format (text, entity1, entity2)
    """
    if entities_of_interest is not None:
        entities_of_interest = {bert2spacy[b] for b in entities_of_interest}
    ents = sents_doc.ents  # get entities for given sentence
    length_doc = len(sents_doc)
    entity_pairs = []
    for i in range(len(ents)):
        e1 = ents[i]
        if entities_of_interest is not None and e1.label_ not in entities_of_interest:
            continue

        for j in range(1, len(ents) - i):
            e2 = ents[i + j]
            if entities_of_interest is not None and e2.label_ not in entities_of_interest:
                continue
            if e1.text.lower() == e2.text.lower():  # make sure e1 != e2
                continue

            if (1 <= (e2.start - e1.end) <= window_size):

                punc_token = False
                start = e1.start - 1 - sents_doc.start
                if start > 0:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start -= 1
                        if start < 0:
                            break
                    left_r = start + 2 if start > 0 else 0
                else:
                    left_r = 0

                # Find end of sentence
                punc_token = False
                start = e2.end - sents_doc.start
                if start < length_doc:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start += 1
                        if start == length_doc:
                            break
                    right_r = start if start < length_doc else length_doc
                else:
                    right_r = length_doc

                if (right_r - left_r) > window_size:  # sentence should not be longer than window_size
                    continue

                x = [token.text for token in sents_doc[left_r:right_r]]
                gap = sents_doc.start + left_r
                e1_info = (e1.text, spacy2bert[e1.label_], (e1.start - gap, e1.end - gap - 1))
                e2_info = (e2.text, spacy2bert[e2.label_], (e2.start - gap, e2.end - gap - 1))
                if e1.start == e1.end:
                    assert x[e1.start - gap] == e1.text, "{}, {}".format(e1_info, x)
                if e2.start == e2.end:
                    assert x[e2.start - gap] == e2.text, "{}, {}".format(e2_info, x)
                entity_pairs.append((x, e1_info, e2_info))
    return entity_pairs

