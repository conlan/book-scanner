import cv2
import argparse
import ocr
import openai_platform
import thumbnail_annotator
import google_books

from ultralytics import YOLO
import supervision as sv
import numpy as np

# tasteDive api key = 1031592-BookScan-0E9A97C5

CLASS_ID_BOOK = 73
CONFIDENCE_THRESHOLD = 0.5

KEY_ESCAPE = 27
KEY_SPACE = 32

BOOK_IMAGE_CAPTURE_PATH = "book.jpg"

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='YOLOV8 Live')

    parser.add_argument(
        '--webcam-resolution',
        default=[1280, 720],
        nargs=2,
        type=int
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO('yolov8l.pt')

    box_annotator = sv.BoxAnnotator(
        thickness=1,
        text_thickness=1,
        text_scale=1
    )

    cooldown = 0
    # bookCooldown = 0

    bookDetection = None
    bookOCR = None
    identifiedBookData = None

    while (True):
        _, frame = cap.read()

        if (cooldown == 0):
            cooldown = 10

            if (bookDetection is None):                
                result = model(frame, agnostic_nms=True)[0]

                detections = sv.Detections.from_yolov8(result)

                detections = detections[detections.class_id == CLASS_ID_BOOK] # filter only book detections

                if len(detections) > 0:
                    for detection in detections:
                        xyxy, confidence, _, _ = detection
                        
                        if confidence > CONFIDENCE_THRESHOLD:
                            x1, y1, x2, y2 = map(int, xyxy)

                            # Crop the image using the bounding box coordinates
                            croppedImage = frame[y1:y2, x1:x2]

                            print("Book detected!")
                            
                            cv2.imwrite(BOOK_IMAGE_CAPTURE_PATH, croppedImage)                    

                            bookDetection = [detection]
                            # bookCooldown = 100
                            break
                        else:
                            print(f"Book detected but confidence is too low: {confidence:.2f}")
            elif (bookDetection is not None) and (bookOCR is None):
                bookOCR = ocr.detect_text_from_book(BOOK_IMAGE_CAPTURE_PATH)
            elif (bookOCR is not None) and (identifiedBookData is None):
                identifiedBookData = openai_platform.identify_book_and_get_recommendations(bookOCR)
                print(identifiedBookData)
            elif (identifiedBookData is not None):
                for recommendation in identifiedBookData["recommendations"]:
                    if (recommendation["thumbnailURL"] is None):
                        hasFetchedAllThumbnailURLs = False
                        google_books.get_thumbnail_url(recommendation)
                        break
                    elif recommendation["thumbnailImage"] is None:
                        google_books.get_thumbnail_image(recommendation)
                        break

                # if hasFetchedAllThumbnailURLs:
                #     for recommendation in identifiedBookData["recommendations"]:
                #         if (recommendation["thumbnailImage"] is None):
                #             google_books.get_thumbnail_image(recommendation)
                #             break
        

        if bookDetection is not None:
            _, confidence, class_id, _ = bookDetection[0]

            labels = []

            if (identifiedBookData is not None):
                labels.append(identifiedBookData["title"])

                # add any thumbnails to the frame that we've fetched
                thumbnail_annotator.annotate_thumbnails(identifiedBookData["recommendations"], frame)
            else:
                labels.append("Book detected...")

            frame = box_annotator.annotate(
                scene=frame,
                detections=bookDetection,
                labels=labels
            )

        cv2.imshow('frame', frame)

        cooldown -= 1

        # if bookCooldown > 0:
        #     bookCooldown -= 1

        #     if bookCooldown == 0:
        #         bookDetection = None
        #         bookOCR = None
        #         identifiedBookData = None

        waitKey = cv2.waitKey(30)

        if waitKey == KEY_ESCAPE:
            break
        elif waitKey == KEY_SPACE:
            bookDetection = None
            bookOCR = None
            identifiedBookData = None

if __name__ == '__main__':
    main()