def annotate_thumbnails(books, frame):
    # frame is a numpy array RGB image
    # books is a list of book objects

    baseX = 25
    baseY = 25
    x = baseX
    y = baseY
    xSpacing = 25
    ySpacing = 40

    for book in books:
        if book["thumbnailImage"] is not None:
            thumbnailImage = book["thumbnailImage"]
            
            # get the dimensions of the thumbnail
            thumbnailHeight, thumbnailWidth, _ = thumbnailImage.shape

            # add the thumbnail to the frame
            frame[y:y+thumbnailHeight, x:x+thumbnailWidth] = thumbnailImage

            x += thumbnailWidth + xSpacing

            # if we've reached the end of the frame, move to the next row
            if x + thumbnailWidth > frame.shape[1]:
                x = baseX
                y += thumbnailHeight + ySpacing