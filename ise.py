import json
import requests
import sys
from bs4 import BeautifulSoup
import os
import re
import spacy
from spanbert import SpanBERT
from spacy_help_functions import extract_relations


class ISE:
    def __init__(self):
        self.relation_map = {1: ("Schools_Attended", ["PERSON"], ["ORGANIZATION"]),
                             2: ("Work_For", ["PERSON"], ["ORGANIZATION"]),
                             3: ("Live_In", ["PERSON"], ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]),
                             4: ("Top_Member_Employees", ["ORGANIZATION"], ["PERSON"])}

        self.queries = set()

        # (subj, relation, obj) -> confidence score
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

        # Retrieve top 10 results
        for i in range(0, 10):
            result = {
                'URL': search_results[i]['link'],
                'Title': search_results[i]['title'],
                'Summary': search_results[i]['snippet']
            }
            results.append(result)
        return results

    def read_params(self):
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

    def extract_plain_text(self, url):
        """
          Extract plain text from a web page pointed by url
          :return: list of sentences(str)
        """
        res = requests.get(url)
        info = BeautifulSoup(res.content, "html.parser")
        txt = info.get_text(separator='', strip=True)

        # the resulting plain text is longer than 10,000 characters, truncate the text to its first 10,000 characters
        if len(txt) > 10000:
            print(f'\tTrimming webpage content from {len(txt)} to 10000 characters')
            txt = txt[:10000]
        return txt

    def extract_relations_spbt(self, text, entities_of_interest):
        nlp = spacy.load("en_core_web_lg")
        doc = nlp(text)
        spanbert = SpanBERT("./pretrained_spanbert")
        relations = extract_relations(doc, spanbert, entities_of_interest)
        return relations

    def add_tups_to_set(self, relations):
        for relation in relations:
            if relation not in self.X:
                self.X[relation] = relations[relation]
            else:
                if self.X[relation] < relations[relation]:
                    self.X[relation] = relations[relation]
                else:
                    print('\t\tDuplicate tuple is ignored because of lower confidence')
                    return 0

    def iterative_set_expansion(self):
        iteration_count = 0
        seen_URLs = set()
        while len(self.X) < self.k:
            print(f'=========== Iteration: {iteration_count} - Query: {self.QUERY} ===========')
            results = self.google_search()
            result_count = 1
            for result in results:
                url = result['URL']
                print(f'\n\nURL ({result_count} / 10): {url}')
                if url in seen_URLs:
                    print('This URL is processed. Move onto next one.')
                    continue
                print("\tFetching text from url ...")
                seen_URLs.add(url)
                text = self.extract_plain_text(url)
                print(f'\tWebpage length (num characters): {len(text)}')
                print("\tAnnotating the webpage using spacy...")
                target_relation = self.relation_map[self.RELATION]
                entities_of_interest = target_relation[1] + target_relation[2]
                relations = self.extract_relations_spbt(text, entities_of_interest)
                self.add_tups_to_set(relations)

    def make(self):
        self.read_params()
        if self.METHOD == "-spanbert":
            self.spbt()
        if self.METHOD == '-gpt3':
            self.gpt()


if __name__ == "__main__":
    instance = ISE()
    instance.make()
