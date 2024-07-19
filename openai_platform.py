import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = os.environ.get("OPENAI_MODEL")

NUM_RECOMMENDATIONS = 16

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def list_to_string(list_of_strings):
    # Convert each string in the list to be enclosed in quotes
    quoted_list = [f"'{item}'" for item in list_of_strings]
    # Join the quoted strings with commas and enclose the whole thing in brackets
    return f"[{', '.join(quoted_list)}]"

def identify_book_and_get_recommendations(ocrLines):    
    prompt = "You're a librarian. What book do you think this is given these scanned cover keywords (note that not all words may have been scanned) " + \
        list_to_string(ocrLines) + "?\n\n" + \
        "Answer strictly following the format: {Title} %--% {Author}. Ensure that you include the full title even if some title words were missing. " + \
        "Then, following the exact same format, include " + str(NUM_RECOMMENDATIONS) + " different books that you think are most similar to the book you identified (but not the original book). Don't include anything else like numbers or punctuation. Only newlines. " + \
        "Ensure you return the exact amount of books requested and ensure that there are NO duplicates in the recommendations.\n\n" + \
        "Here's an example:\n" + \
        "The Catcher in the Rye %--% J.D. Salinger\n" + \
        "1984 %--% George Orwell\n" + \
        "To Kill a Mockingbird %--% Harper Lee\n" + \
        "Brave New World %--% Aldous Huxley\n" + \
        "The Great Gatsby %--% F. Scott Fitzgerald"
    
    print('--Prompt--')
    print(prompt)
    print('--')

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=OPENAI_MODEL,
    )
    
    responseContent = chat_completion.choices[0].message.content

    print('--Response--')
    print(responseContent)
    print('--')

    responseLines = responseContent.split("\n")
    books = []

    for i in range(len(responseLines)):
        line = responseLines[i].strip()

        if line == "" or " %--% " not in line:
            continue
            
        title = line.split(" %--% ")[0]
        author = line.split(" %--% ")[1]

        # if title is longer than 20 characters, truncate it with an ellipsis
        if len(title) > 20:
            title = title[:20] + "..."

        books.append({
            'title' : title,
            'author' : author,
            'thumbnailURL' : None,
            'thumbnailImage' : None
        })
    
    return {
        'title' : f"\"{books[0]['title']}\" by {books[0]['author']}",
        'recommendations' : books[1:]
    }