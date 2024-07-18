import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_MODEL = os.environ.get("OPENAI_MODEL")

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def list_to_string(list_of_strings):
    # Convert each string in the list to be enclosed in quotes
    quoted_list = [f"'{item}'" for item in list_of_strings]
    # Join the quoted strings with commas and enclose the whole thing in brackets
    return f"[{', '.join(quoted_list)}]"

def identify_book_and_get_recommendations(ocrLines):
    prompt = "What book do you think this is given these scanned cover keywords (note that not all words may have been scanned) " + \
        list_to_string(ocrLines) + "?\n\n" + \
        "Answer strictly following the format: {Title} -- {Author}. Ensure that you include the full title even if some title words were missing."
    
    print(prompt)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        model=OPENAI_MODEL,
    )

    print(chat_completion.choices[0].message.content)
    
    return {
        'title' : chat_completion.choices[0].message.content,
        'recommendations' : [
            'The Catcher in the Rye',
            '1984'
        ]
    }