import cv2
import argparse
import ocr

from ultralytics import YOLO
import supervision as sv
import numpy as np

# tasteDive api key = 1031592-BookScan-0E9A97C5

CLASS_ID_BOOK = 73
CONFIDENCE_THRESHOLD = 0.6
KEY_ESCAPE = 27

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
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    counter = 0
    bookCountdown = 0

    BOOK_IMAGE_CAPTURE_PATH = "book.jpg"

    bookDetection = None
    bookOCR = None

    while (True):
        ret, frame = cap.read()

        if (bookDetection is None) and (counter % 30) == 0:
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

                        # cv2.imwrite("book.jpg", cropped_image)

                        print("Book detected!")
                        
                        # print(pytesseract.image_to_string(croppedImage, config='--psm 11'))
                        
                        cv2.imwrite(BOOK_IMAGE_CAPTURE_PATH, croppedImage)                    

                        bookDetection = [detection]
                        bookCountdown = 60
                        break
                    else:
                        print(f"Book detected but confidence is too low: {confidence:.2f}")
        elif (bookDetection is not None) and (bookOCR is None):
            bookOCR = ocr.detect_text_from_book(BOOK_IMAGE_CAPTURE_PATH)
            print(bookOCR)
        

        if bookDetection is not None:
            labels = [
                f"{model.model.names[class_id]}, {class_id} {confidence:.2f}"
                for _, confidence, class_id, _
                in detections
            ]
            frame = box_annotator.annotate(
                scene=frame,
                detections=bookDetection,
                labels=labels
            )

        cv2.imshow('frame', frame)

        counter += 1

        if bookCountdown > 0:
            bookCountdown -= 1

            if bookCountdown == 0:
                bookDetection = None
                bookOCR = None
                counter = 0

        if cv2.waitKey(30) == KEY_ESCAPE:
            break

if __name__ == '__main__':
    main()