#!/usr/bin/env python3

'''
Experiments with water meter image search by circle hough transform

'''

import argparse
import pathlib
import json
import cv2
import numpy as np
import base64

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process image with labelme annotiation. '
                                                 'Bounding box with label "meter" should present in annotation.')
    parser.add_argument('filename', metavar='<filename>', type=str, nargs=1,
                        help='Path to image or json annotation')

    args = parser.parse_args()
    print("Processing file: ", args.filename[0])

    filepath = pathlib.Path(args.filename[0])
    json_filepath = filepath
    if filepath.suffix != '.json':
        json_filepath = filepath.with_suffix(".json")
    print("JSON filepath: ", json_filepath)
    with open(json_filepath, "r") as f:
        annotation = json.load(f)
    for k in annotation:
        print (k)

    image = cv2.imdecode(np.frombuffer(base64.b64decode(annotation['imageData'])), 0)

    image = cv2.medianBlur(image, 5)

    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT_ALT, 1, 20,
                              param1=20, param2=0.9, minRadius=300, maxRadius=2000)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow("Image", image)
    #while not cv2.waitKey(10) in [32, 27, 13]:
    #    pass

    import matplotlib.pyplot as plt

    from skimage import data, color, img_as_ubyte
    from skimage.feature import canny
    from skimage.transform import hough_ellipse
    from skimage.draw import ellipse_perimeter

    # Load picture, convert to grayscale and detect edges
    image_rgb = cv2.imdecode(np.frombuffer(base64.b64decode(annotation['imageData'])), cv2.IMREAD_UNCHANGED)
    image_gray = color.rgb2gray(image_rgb)

    #image_rgb = data.coffee()[0:220, 160:420]
    #image_gray = color.rgb2gray(image_rgb)

    edges = canny(image_gray)#, sigma=12.0,
                  #low_threshold=0.1, high_threshold=0.2)


    plt.imshow(edges, cmap='gray')
    plt.show()
    while not cv2.waitKey(10) in [32, 27, 13]:
        pass
    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators
    result = hough_ellipse(edges, accuracy=50, threshold=250,
                           min_size=900, max_size=1200)
    result.sort(order='accumulator')

    # Estimated parameters for the ellipse
    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]

    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    image_rgb[cy, cx] = (0, 0, 255)
    # Draw the edge (white) and the resulting ellipse (red)
    edges = color.gray2rgb(img_as_ubyte(edges))
    edges[cy, cx] = (250, 0, 0)

    fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
                                    sharex=True, sharey=True)

    ax1.set_title('Original picture')
    ax1.imshow(image_rgb)

    ax2.set_title('Edge (white) and result (red)')
    ax2.imshow(edges)

    plt.show()
    cv2.waitKey(0)
