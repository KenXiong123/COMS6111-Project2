import requests
from bs4 import BeautifulSoup
import re
import urllib
# Make a GET request to the webpage
url = "https://en.wikipedia.org/wiki/Mark_Zuckerberg"
page = urllib.request.urlopen(url, timeout=10).read().decode('utf-8')
soup = BeautifulSoup(page, 'html.parser')

# [s.extract() for s in soup(['style', 'script', '[document]', 'head', 'title'])]
txt = soup.getText()
# # Removing redundant newlines and some whitespace characters
txt = re.sub(u'\xa0', ' ', txt)

txt = re.sub('\t+', ' ', txt)

txt = re.sub('\n+', ' ', txt)

txt = re.sub(' +', ' ', txt)

txt = txt.replace('\u200b', '')


# the resulting plain text is longer than 10,000 characters, truncate the text to its first 10,000 characters
if len(txt) > 10000:
    print(f'\tTrimming webpage content from {len(txt)} to 10000 characters')
    txt = txt[:10000]
# Print the plain text
print(txt)
