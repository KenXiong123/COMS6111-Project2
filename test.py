import requests
from bs4 import BeautifulSoup

# Make a GET request to the webpage
url = "https://en.wikipedia.org/wiki/Beautiful_Soup_(HTML_parser)"
response = requests.get(url)

# Use Beautiful Soup to parse the HTML content of the webpage
soup = BeautifulSoup(response.content, "html.parser")

# Extract the plain text from the webpage, excluding whitespace
text = soup.get_text(separator='', strip=True)

# Print the plain text
print(text)
