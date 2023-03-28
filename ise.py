import json
import requests
import sys
from bs4 import BeautifulSoup
from bs4.element import Comment
import os
import re
import spacy
from spanbert import SpanBERT
from new_help_functions import extract_relations_spbt, extract_relations_gpt
import openai


class ISE:
    def __init__(self):
        self.relation_map = {1: ("Schools_Attended", "per:schools_attended", ["PERSON"], ["ORGANIZATION"]),
                             2: ("Work_For", "per:employee_of", ["PERSON"], ["ORGANIZATION"]),
                             3: ("Live_In", "per:cities_of_residence", ["PERSON"],
                                 ["LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]),
                             4: ("Top_Member_Employees", "org:top_members/employees", ["ORGANIZATION"], ["PERSON"])}

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
        :return: A list of dicts; each dict represents one search result containing the URL, Title, and Summary
        """
        results = []

        # Google search
        url = "https://www.googleapis.com/customsearch/v1?key=" + self.GOOGLE_JSON_API_KEY + "&cx=" + self.GOOGLE_ENGINE_ID + "&q=" + self.QUERY
        response = requests.get(url)
        search_results = json.loads(response.text)['items']

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
        Read parameters from the command line and assign them to global vars
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
        print("RELATION \t= " + self.relation_map[self.RELATION][0])
        print("THRESHOLD \t= " + str(self.THRESHOLD))
        print("QUERY     \t= " + self.QUERY)
        print("# of Tuples \t= " + str(self.k))
        print("Loading necessary libraries; This should take a minute or so...")

    def extract_plain_text(self, url):
        """
        Extract plain text from a web page pointed by url
        :param url: A string containing the URL of the web page
        :return: A list of strings, each string represents a sentence from the plain text of the web page
        """
        res = requests.get(url)
        soup = BeautifulSoup(res.content, 'html.parser')
        [s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title'])]
        txt = soup.getText().strip()
        # Removing redundant newlines and some whitespace characters
        txt = re.sub(u'\xa0', ' ', txt)

        txt = re.sub('\t+', ' ', txt)

        txt = re.sub('\n+', ' ', txt)

        txt = re.sub(' +', ' ', txt)

        txt = txt.replace('\u200b', '')
        # the resulting plain text is longer than 10,000 characters, truncate the text to its first 10,000 characters
        if len(txt) > 10000:
            print(f'\tTrimming webpage content from {len(txt)} to 10000 characters')
            txt = txt[:10000]
        return txt

    def extract_relations(self, text, entities_of_interest, target_relation):
        """
        Extract relations between entities of interest in the given text using either SpanBERT or GPT-3 method depending
        on the flag. The resulting relations are filtered based on the provided target relation.
        :param text: the text to analyze
        :param entities_of_interest: a list of entities of interest
        :param target_relation: the target relation to filter the extracted relations
        :return: a list of relations filtered by the target relation
        """
        relations = "no_relation"
        nlp = spacy.load("en_core_web_lg")
        doc = nlp(text)

        if self.METHOD == "-spanbert":
            spanbert = SpanBERT("./pretrained_spanbert")
            relations = extract_relations_spbt(doc, spanbert, entities_of_interest, self.THRESHOLD, target_relation)
        if self.METHOD == "-gpt3":
            openai.api_key = self.OPENAI_KEY
            relations = extract_relations_gpt(doc, openai, entities_of_interest, self.relation_map[self.RELATION][0])
        return relations

    def add_tups_to_set(self, relations):
        """
        Add relations to the set X if they are not already in it or have a higher confidence score. Returns the count
        of newly added relations.
        :param relations: a dictionary of relations to add
        :return: the count of newly added relations
        """
        count = 0
        for relation in relations:
            if relation not in self.X:
                self.X[relation] = relations[relation]
                count += 1
            else:
                if self.X[relation] < relations[relation]:
                    self.X[relation] = relations[relation]

        return count

    def iterative_set_expansion(self):
        """
        Perform iterative set expansion to expand the set of extracted relations until the target number of relations is reached.
        """
        iteration_count = 1
        seen_URLs = set()
        while len(self.X) < self.k:
            if iteration_count != 1:
                selected_tup = None
                max_confidence = -float('inf')
                for tup, confidence in self.X.items():
                    if (tup[0] + " " + tup[2]) not in self.QUERY:
                        if "-spanbert" == self.METHOD:
                            if confidence > max_confidence:
                                max_confidence = confidence
                                selected_tup = tup
                        else:
                            selected_tup = tup
                            break
                if selected_tup is not None:
                    self.QUERY = self.QUERY + ' ' + selected_tup[0] + ' ' + selected_tup[2]
                    print("query", self.QUERY)
                else:
                    break
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
                entities_of_interest = target_relation[2] + target_relation[3]
                relations = self.extract_relations(text, entities_of_interest, self.relation_map[self.RELATION][1])
                c = self.add_tups_to_set(relations)
                print(f"\tRelations extracted from this website: {c} (Overall: {len(relations)})")
                result_count += 1
            iteration_count += 1
        if self.METHOD == "-spanbert":
            self.summary_spbt(iteration_count - 1)
        else:
            self.summary_gpt(iteration_count - 1)

    def summary_spbt(self, iteration_count):
        """
        Print a summary of all the extracted relations when using the SpanBERT method.
        :param iteration_count: the number of iterations performed during set expansion
        """
        print(
            f'================== ALL RELATIONS for {self.relation_map[self.RELATION][1]} ( {len(self.X)} ) =================')
        sorted_X = dict(sorted(self.X.items(), key=lambda x: x[1], reverse=True))
        for tup in sorted_X:
            print(f'Confidence: {self.X[tup]}   | Subject: {tup[0]}   | Object: {tup[2]}')
        print(f'Total # of iterations = {iteration_count}')

    def summary_gpt(self, iteration_count):
        """
        Print a summary of all the extracted relations when using the GPT-3 method.
        :param iteration_count: the number of iterations performed during set expansion
        """
        print(
            f'================== ALL RELATIONS for {self.relation_map[self.RELATION][0]} ( {len(self.X)} ) =================')
        sorted_X = dict(sorted(self.X.items(), key=lambda x: x[1], reverse=True))
        for tup in sorted_X:
            print(f'Subject: {tup[0]}   | Object: {tup[2]}')
        print(f'Total # of iterations = {iteration_count}')

    def make(self):
        """
        Read the parameters, perform iterative set expansion, and print a summary of the extracted relations.
        """
        self.read_params()
        self.iterative_set_expansion()


if __name__ == "__main__":
    instance = ISE()
    instance.make()
