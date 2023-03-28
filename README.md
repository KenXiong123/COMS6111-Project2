# CS6111-Project-2

## Team Members

Jingyue Qin(**jq2343**)

Ken Xiong(**kx2175**)

## Files list

1. README.pdf

2. proj2.tar.gz

3. transcript_spanbert.txt

4. transcript_gpt3.txt

## Run the program

1. VM

   ```
   $ sudo apt-get install tar xz-utils
   $ tar xzvf proj2.tar.gz
   $ cd proj2
   $ sudo apt-get install python3-pip
   $ 
   $ pip install -r requirements.txt
   $ python3 project2.py [-spanbert|-gpt3] <google api key> <google engine id> <openai secret key> <r> <t> <q> <k>
   ```

   where:

   - **[-spanbert|-gpt3]** is either **-spanbert** or **-gpt3**, to indicate which relation extraction method we are requesting>
   - **<google api key>** is your Google Custom Search Engine JSON API Key (see above)
   - **<google engine id>** is your Google Custom Search Engine ID (see above)
   - **<openai secret key>** is your OpenAI API secret key (see above)
   - **<r>** is an integer between 1 and 4, indicating the relation to extract: 1 is for **Schools_Attended**, 2 is for **Work_For**, 3 is for **Live_In**, and 4 is for **Top_Member_Employees**
   - **<t>** is a real number between 0 and 1, indicating the "extraction confidence threshold," which is the minimum extraction confidence that we request for the tuples in the output; **t** is ignored if we are specifying **-gpt3**
   - **<q>** is a "seed query," which is a list of words in double quotes corresponding to a plausible tuple for the relation to extract (e.g., "bill gates microsoft" for relation Work_For)
   - **<k>** is an integer greater than 0, indicating the number of tuples that we request in the output

## Internal design of the project

Here is the description of our internal design:

For ise.py:

The main class in the program is `ISE`, which stands for "Information Search Engine". The `ISE` class has the following attributes:

- `relation_map`: a dictionary that maps relation IDs to tuples containing the relation name, the relation label, and the entity types of the subject and object.
- `queries`: a set of URLs that have been searched.
- `X`: a dictionary that maps relation tuples to confidence scores.
- `GOOGLE_JSON_API_KEY`: a string containing the Google API key.
- `GOOGLE_ENGINE_ID`: a string containing the Google engine ID.
- `OPENAI_KEY`: a string containing the OpenAI API key.
- `METHOD`: a string indicating the relation extraction method to be used (`-spanbert` or `-gpt3`).
- `RELATION`: an integer indicating the relation ID.
- `THRESHOLD`: a float indicating the confidence threshold for relation extraction.
- `QUERY`: a string containing the search query.
- `k`: an integer indicating the maximum number of relation tuples to be extracted.
- `retrieved_url`: a set of URLs that have been retrieved.

The `ISE` class has the following methods:

- `google_search()`: a method that performs a Google search using the `QUERY` attribute and returns a list of the top 10 search results, each represented as a dictionary containing the URL, title, and summary of the result.
- `read_params()`: a method that reads command-line arguments and sets the relevant attributes.
- `extract_plain_text(url)`: a method that extracts plain text from a webpage pointed by a given URL and returns a list of sentences.
- `extract_relations(text, entities_of_interest, target_relation)`: a method that extracts relation tuples from a given text using the specified relation extraction method (`-spanbert` or `-gpt3`), the list of entities of interest, the confidence threshold, and the target relation.

The `extract_relations()` method calls either `extract_relations()` or `extract_relations_gpt()` depending on the `METHOD` attribute. Both of these methods take a spacy `doc` object, a list of entities of interest, a confidence threshold, and a target relation, and return a list of relation tuples.

The program reads the command-line arguments using the `read_params()` method, performs a Google search using the `google_search()` method, extracts plain text from the retrieved web pages using the `extract_plain_text()` method, and extracts relation tuples using the `extract_relations()` method. The extracted relation tuples are stored in the `X` attribute and printed to the console. The program stops after extracting `k` relation tuples.

External Libraries used are:

- json: to encode and decode JSON data.
- sys: to handle command-line arguments.
- requests: to make HTTP requests to Google Custom Search API.
- TfidfVectorizer: from scikit-learn, to compute tf-idf weights of the tokens in the collection of documents.
- numpy: to perform mathematical operations on arrays.



For new_help_functions.py:

This program consists of two main functions: "extract_relations" and "extract_relations_gpt". It uses the following external libraries:

- `spacy` for natural language processing
- `collections.defaultdict` for dictionary with default values
- `spanbert` for named entity recognition and relation extraction
- `openai` for generating text completions

The `spacy2bert` and `bert2spacy` dictionaries map entity labels between Spacy and SpanBERT formats.

The `get_entities` function takes a Spacy sentence as input and returns a list of tuples, each containing a named entity's text and its SpanBERT label.

The `extract_relations` function takes a Spacy document and an optional list of named entity types of interest. It processes each sentence in the document, creating pairs of named entities that match the entity types of interest. For each entity pair, it uses SpanBERT to predict the relationship between the entities. If the predicted relationship matches the target relation, it saves the relation and its confidence score to a dictionary.

The `extract_relations_gpt` function is similar to `extract_relations`, but it uses OpenAI's GPT-3 to generate the relation extraction prompt and obtain the relation extraction output. It also converts SpanBERT's entity labels to Spacy's entity labels before processing the document.

## Description of how to carried out Step 3



## Google Custom Search Engine JSON API Key and Engine ID

**API key**: AIzaSyAp1Mlu_GczzUdNm34c8dLG0MC5v09gWXE

**Engine ID**: 02e16cfd8d0bbe48a


