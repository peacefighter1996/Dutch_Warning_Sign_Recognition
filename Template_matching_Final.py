"""
Created on Tue Dec 17 2018

@author: Ian, Bram
@version: 0.1

Template based road sign recognision.
Uses the sign database to generate templates to match on streetview images.
"""

import copy
import glob

import imutils
import numpy as np
from cairosvg import svg2png

import cv2
from pyimagesearch.shapedetector import ShapeDetector


def get_mask(image, maskrange):
    colorLow = np.array([maskrange[0], maskrange[1], maskrange[2]])
    colorHigh = np.array([maskrange[3], maskrange[4], maskrange[5]])
    mask = cv2.inRange(image, colorLow, colorHigh)
    return mask


# colour set to filter red on low side
icol = (0, 90, 95, 5, 255, 255)
# colour set to filter red on high side
icol2 = (160, 90, 95, 255, 255, 255)
Images = []

for file in glob.glob('test/*.png'):
    print(file)
    Images.append(file)

# Generate template images from scg data and save Template
template_list = []
for file in glob.glob('Signs/*.svg'):
    svgfile = open(file)
    svg = svgfile.read()
    svg2png(svg, write_to=file[:-3]+'png')
for file in glob.glob('Signs/*.png'):
    template_list.append([cv2.imread(file, 0), file])

match_method = 'cv2.TM_SQDIFF_NORMED'

for Image in Images:
    print(Image+"\n")
    Base_Image = cv2.imread(Image)
    if (type(Base_Image) != None):
        Change_image = copy.copy(Base_Image)
        Result_image = copy.copy(Base_Image)
        hsv = cv2.cvtColor(Change_image, cv2.COLOR_BGR2HSV)

        # HSV values to define a colour range.
        mask = get_mask(hsv, icol)
        mask2 = get_mask(hsv, icol2)
        mask = mask+mask2

        kernal = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
        Change_image = cv2.bitwise_and(Change_image, Change_image, mask=mask)

        cv2.imshow('img4', Change_image)

        # find contours in the thresholded image and initialize the
        # shape detector
        cnts = cv2.findContours(mask.copy(),
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        sd = ShapeDetector()

        # loop over the contours
        for c in cnts:

            # compute the center of the contour, then detect the name of the
            # shape using only the contour
            M = cv2.moments(c)
            try:
                cX = int((M["m10"] / M["m00"]))
                cY = int((M["m01"] / M["m00"]))
                (shape, approx) = sd.detect(c)

                if len(approx) == 3 or len(approx) == 4:
                    (x, y, w, h) = cv2.boundingRect(approx)
                    cv2.rectangle(Result_image, (x, y), (x+w, y+h), 255, 2)

                    ar = float(h)/float(w)
                    c = c.astype("float")
                    c = c.astype("int")
                    loc = [x, y]
                    cv2.drawContours(Result_image, [c], -1, (0, 255, 0), 1)
                    cv2.imshow("all", Result_image)
                    if ar < 2 and ar > 0.5 and h >= 25 and w >= 25:
                        # multiply the contour (x, y)-coordinates by the resize ratio,
                        # then draw the contours and the name of the shape on the image
                        crop_img = Base_Image[y:y+h,
                                              x:x+w]
                        # resize to match STD size of SVG sign (the whole sign not the croped out middle)
                        crop_img = cv2.resize(crop_img, (350, 300))
                        crop_found = copy.copy(crop_img)
                        # sharpen image
                        kernel = np.array([[-1, -1, -1, -1, -1],
                                           [-1, -1, -1, -1, -1],
                                           [-1, -1, 25, -1, -1],
                                           [-1, -1, -1, -1, -1],
                                           [-1, -1, -1, -1, -1]])
                        cv2.imshow("compare:before1", crop_img)
                        crop_img = cv2.filter2D(crop_img, -1, kernel)

                        cv2.imshow("compare:before2", crop_img)
                        crop_img = cv2.bilateralFilter(crop_img, 9, 75, 75)
                        cv2.imshow("compare", crop_img)
                        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

                        compair = []
                        for x in range(0, len(template_list)):
                            compair.append([template_list[x][1]])

                        evaluation = []
                        w, h = template_list[0][0].shape[::]

                        for x in range(0, len(template_list)):
                            template = template_list[x][0]

                            evaluation_image = copy.copy(crop_img)
                            method = eval(match_method)

                            # Apply template Matching
                            res = cv2.matchTemplate(
                                evaluation_image, template, method)
                            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(
                                res)

                            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                            top_left = min_loc

                            bottom_right = (top_left[0] + h,
                                            top_left[1] + w)

                            cv2.rectangle(crop_found, top_left,
                                          bottom_right, 255, 2)

                            evaluation.append([min_val, max_val, x])
                            compair[x].append([min_val, max_val])

                        evaluation.sort()
                        lowsave = evaluation[0][0]
                        xsave = 0
                        # Print 5 closest results
                        print("Index : Sign : Result")
                        print("-"*6 + "+"+"-"*6 + "+" + "-"*8)
                        for x in range(0, 5):
                            print('    {} :  {} : {:.7f}'.format(x,
                                                                 compair[evaluation[x]
                                                                         [2]][0][-7:-4],
                                                                 evaluation[x][0]))

                        # Print Result and print on image if combination result is 47.5% sure
                        if lowsave < 0.525:
                            print("\nfound sign: {} with {:2.1f}%\n".format(compair[evaluation[0][2]]
                                                                            [0][-7:-4], 100.0-lowsave*100))
                            cv2.putText(Result_image,
                                        compair[evaluation[0][2]][0][-7:-4],
                                        (cX, cY),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,
                                        (0, 255, 255),
                                        2)
                        # show where matches are found and the mathing tamplate
                        cv2.imshow("CE", crop_found)
                        cv2.imshow('Matched template',
                                   template_list[(evaluation[0][2])][0])

            except ZeroDivisionError:
                _ = 0
        cv2.imshow("all", Result_image)
        # wait till space or esc is pressed
        while(1):
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break
            if k == 32:
                print("next image -----------------------------\n")
                break
        if k == 27:
            break
    else:
        print("could not find image")

cv2.destroyAllWindows()
