import string
import requests

from PIL import Image
import numpy as np
import io

GOOGLE_BOOKS_API_URL = "https://www.googleapis.com/books/v1/volumes?q="

def get_thumbnail_image(book):
    thumbnailUrl = book["thumbnailURL"]

    print ("\nDownloading thumbnail for: ", thumbnailUrl)

    # query the URL
    response = requests.get(thumbnailUrl)

    # get the image data
    image_data = response.content

    image = Image.open(io.BytesIO(image_data))

    image_np = np.array(image)

    print(image_np.shape)

    # set the image data in the book object
    book["thumbnailImage"] = image_np

def get_thumbnail_url(book):
    title = book['title']
    author = book['author']

    # remove punctuation from title
    title = title.translate(str.maketrans('', '', string.punctuation))
    # replace title spaces with plus signs
    title = title.replace(" ", "+")
    # replace author spaces with plus signs
    author = author.replace(" ", "+")
    # combine title and author in a query (use intitle and inauthor to ensure both are in the title and author fields)
    query = f"intitle:{title}+inauthor:{author}"

    url = GOOGLE_BOOKS_API_URL + query

    print ("\nSearching for thumbnail for: ", url)

    # query the URL
    response = requests.get(url)

    # get the JSON response
    data = response.json()

    book["thumbnailURL"] = "https://placehold.co/128x200.jpg"

    # if there are items in the response
    if 'items' in data:
        # get the first item
        item = data['items'][0]

        # get the thumbnail if it exists
        if 'imageLinks' in item['volumeInfo']:
            thumbnail = item['volumeInfo']['imageLinks']['thumbnail']

            # set the thumbnail in the book object
            book["thumbnailURL"] = thumbnail

            print("Thumbnail found: ", thumbnail)
        else:
            print("Thumbnail not found")    
    else:
        print("Thumbnail not found")