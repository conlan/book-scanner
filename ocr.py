import os

from google.cloud import vision
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'vision-api-key.json'

visionClient = vision.ImageAnnotatorClient()

def detect_text_from_book(image_path):
    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = visionClient.document_text_detection(image=image)

    full_text_annotation = response.full_text_annotation

     # Extract all lines of text
    lines = []
    for page in full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    line = ''.join([symbol.text for symbol in word.symbols])
                    lines.append(line)
    
    return lines