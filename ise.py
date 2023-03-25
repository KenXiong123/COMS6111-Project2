import json
import requests
import sys
from bs4 import BeautifulSoup
import os
import re
import spacy

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

def get_entities(sentence, entities_of_interest):
    return [(e.text, spacy2bert[e.label_]) for e in sentence.ents if e.label_ in spacy2bert]


def create_entity_pairs(sents_doc, entities_of_interest, window_size=40):
    '''
    Input: a spaCy Sentence object and a list of entities of interest
    Output: list of extracted entity pairs: (text, entity1, entity2)
    '''
    entities_of_interest = {bert2spacy[b] for b in entities_of_interest}
    ents = sents_doc.ents # get entities for given sentence

    length_doc = len(sents_doc)
    entity_pairs = []
    for i in range(len(ents)):
        e1 = ents[i]
        if e1.label_ not in entities_of_interest:
            continue

        for j in range(1, len(ents) - i):
            e2 = ents[i + j]
            if e2.label_ not in entities_of_interest:
                continue
            if e1.text.lower() == e2.text.lower(): # make sure e1 != e2
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

                if (right_r - left_r) > window_size: # sentence should not be longer than window_size
                    continue

                x = [token.text for token in sents_doc[left_r:right_r]]
                gap = sents_doc.start + left_r
                e1_info = (e1.text, spacy2bert[e1.label_], (e1.start - gap, e1.end - gap - 1))
                e2_info = (e2.text, spacy2bert[e2.label_], (e2.start - gap, e2.end - gap - 1))
                if e1.start == e1.end:
                    assert x[e1.start-gap] == e1.text, "{}, {}".format(e1_info, x)
                if e2.start == e2.end:
                    assert x[e2.start-gap] == e2.text, "{}, {}".format(e2_info, x)
                entity_pairs.append((x, e1_info, e2_info))
    return entity_pairs

class ISE:
    def __init__(self):
        self.relation_map = {1: "Schools_Attended", 2: "Work_For", 3: "Live_In", 4: "Top_Member_Employees"}

        self.entity_map = {1: ['PERSON', 'ORGANIZATION'], 2: ['PERSON', 'ORGANIZATION'], 3: ['PERSON', 'LOCATION'],
                           4: ['ORGANIZATION', 'PERSON']}

        self.queries = set()

        # (word1, word2) -> (entity1, entity2, relation, prob)
        self.X = {}

        # Parameters from input
        self.GOOGLE_JSON_API_KEY = ""
        self.GOOGLE_ENGINE_ID = ""
        self.OPENAI_KEY = ""
        self.METHOD = ""
        self.RELATION = 0
        self.THRESHOLD = 0
        self.QUERY = ""
        self.k = 0

        # Retrieved set
        self.retrieved_url = set()

        # Client
        # self.client = NLPCoreClient(os.path.abspath("stanford-corenlp-full-2017-06-09"))

    def google_search(self):
        """
        Return the Top-10 results of Google search using QUERY
        :return: list
        """
        results = []

        # Google search
        url = "https://www.googleapis.com/customsearch/v1?key=" + self.GOOGLE_JSON_API_KEY + "&cx=" + self.GOOGLE_ENGINE_ID + "&q=" + self.QUERY
        response = requests.get(url)
        search_results = json.loads(response.text)['items']

        # Retrieve each url and extract plain text
        for item in search_results:
            item_url = item['link']
            if url not in self.retrieved_url:
                try:
                    text = self.extract_plain_text(url)
                    results.append(text)
                except:
                    pass
            self.retrieved_url.add(url)

    def extract_plain_text(self):
        """
          Extract plain text from a web page pointed by url
          :return: list of sentences(str)
        """
        res = requests.get(url)
        info = BeautifulSoup(res.text, "html.parser")
        for script in info(["script", "style"]):
            script.extract()

        txt = info.get_text()

        # the resulting plain text is longer than 10,000 characters, truncate the text to its first 10,000 characters
        if len(txt) > 10000:
            txt = txt[:10000]
        return txt

    def read(self):
        """
        Read parameters from command line and assign to global vars
        :return: void
        """
        inputs = sys.argv

        if len(inputs) < 8:
            print("Please enter valid usage: python3 ise.py [-spanbert|-gpt3] "
                  "<google api key> <google engine id> <openai secret key> <r> <t> <q> <k>")
            sys.exit(1)

        self.METHOD = inputs[1]
        self.GOOGLE_JSON_API_KEY = inputs[2]
        self.GOOGLE_ENGINE_ID = inputs[3]
        self.OPENAI_KEY = inputs[4]
        self.RELATION = int(inputs[5])
        self.THRESHOLD = float(inputs[6])
        self.QUERY = inputs[7]
        self.k = int(inputs[8])

        # Print to console
        print("Parameters:")
        print("Client key \t= " + self.GOOGLE_JSON_API_KEY)
        print("Engine key \t= " + self.GOOGLE_ENGINE_ID)
        print("OpenAI key \t= " + self.OPENAI_KEY)
        print("Method     \t= " + self.METHOD[1:])
        print("RELATION \t= " + self.relation_map[self.RELATION])
        print("THRESHOLD \t= " + str(self.THRESHOLD))
        print("QUERY     \t= " + self.QUERY)
        print("# of Tuples \t= " + str(self.k))
        print("Loading necessary libraries; This should take a minute or so...")
    def spbt(self):

    def make(self):
        self.read()
        if self.METHOD == "-spanbert":
            self.spbt()
        if self.METHOD == '-gpt3':
            self.gpt()

if __name__ == "__main__":
    instance = ISE()
    instance.make()