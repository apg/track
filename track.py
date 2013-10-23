#!/usr/bin/env python

import cv
import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE


class Points:

    def __init__(self):
        self.pts = []

    def add(self, pt):
        self.pts = self.pts[1:10]
        self.pts.append(pt)

    def get(self, fn):
        if len(self.pts):
            return fn(self.pts)
        else:
            return (0, 0)


class Target:

    def __init__(self):
        self.capture = cv.CaptureFromCAM(0)

    def run(self):
        point_buffer = Points()
        pygame.init()
        fpsClock = pygame.time.Clock()        
        output_size = (1276, 701)
        window = pygame.display.set_mode(output_size)
        black = pygame.Color(0, 0, 0)
        background = pygame.image.load("eye-background.png")
        hand = pygame.image.load("eye-hand.png")
        hand_size = hand.get_size()

        # Capture first frame to get size
        frame = cv.QueryFrame(self.capture)
        frame_size = cv.GetSize(frame)
        color_image = cv.CreateImage(cv.GetSize(frame), 8, 3)
        grey_image = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_8U, 1)
        moving_average = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_32F, 3)

        center_point = ((1276/2), (701/2))
        point_buffer.add(center_point)
        
        first = True

        while True:
            closest_to_left = cv.GetSize(frame)[0]
            closest_to_right = cv.GetSize(frame)[1]

            color_image = cv.QueryFrame(self.capture)
            output_image = cv.CreateImage(cv.GetSize(frame), cv.IPL_DEPTH_8U, 3)

            # Smooth to get rid of false positives
            cv.Smooth(color_image, color_image, cv.CV_GAUSSIAN, 3, 0)

            if first:
                difference = cv.CloneImage(color_image)
                temp = cv.CloneImage(color_image)
                cv.ConvertScale(color_image, moving_average, 1.0, 0.0)
                first = False
            else:
                cv.RunningAvg(color_image, moving_average, 0.020, None)

            # Convert the scale of the moving average.
            cv.ConvertScale(moving_average, temp, 1.0, 0.0)

            # Minus the current frame from the moving average.
            cv.AbsDiff(color_image, temp, difference)

            # Convert the image to grayscale.
            cv.CvtColor(difference, grey_image, cv.CV_RGB2GRAY)

            # Convert the image to black and white.
            cv.Threshold(grey_image, grey_image, 70, 255, cv.CV_THRESH_BINARY)

            # Dilate and erode to get people blobs
            cv.Dilate(grey_image, grey_image, None, 18)
            cv.Erode(grey_image, grey_image, None, 10)

            storage = cv.CreateMemStorage(0)
            contour = cv.FindContours(grey_image, storage, cv.CV_RETR_CCOMP, cv.CV_CHAIN_APPROX_SIMPLE)
            points = []

            while contour:
                bound_rect = cv.BoundingRect(list(contour))
                contour = contour.h_next()

                pt1 = (bound_rect[0], bound_rect[1])
                pt2 = (bound_rect[0] + bound_rect[2], bound_rect[1] + bound_rect[3])
                points.append(pt1)
                points.append(pt2)

            if len(points):
                center_point = reduce(lambda a, b: ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), points)
                # flip it horizontally
                center_point = (frame_size[0] - center_point[0], center_point[1])
                point_buffer.add(center_point)

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    raise SystemExit()
                elif event.type == KEYDOWN and event.key == K_ESCAPE:
                    pygame.event.post(pygame.event.Event(QUIT))


            # draw it!
            window.fill(black)
            window.blit(background, (0, 0))

            FACTORX = 2
            FACTORY = 3

            fit_size = (frame_size[0] / FACTORX, frame_size[1] / FACTORY)

            xx = (output_size[0] - fit_size[0]) / 2
            yy = (output_size[1] - fit_size[1]) / 3.5

            cp = point_buffer.get(lambda xs: (sum(x[0] for x in xs) / len(xs), sum(x[1] for x in xs) / len(xs)))

            # center the center point in the middle of the background.
            window.blit(hand, ((cp[0]/FACTORX) + xx - (hand_size[0] / 2), (cp[1]/FACTORY) + yy - (hand_size[1] / 2)))

            pygame.display.update()
            fpsClock.tick(30)

if __name__=="__main__":
    t = Target()
    t.run()
