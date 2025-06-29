import mediapipe as mp
import time
import cv2
import os
import argparse
import re
from random import randrange
import torchvision.transforms as T
from PIL import Image
import numpy as np

#TODO Add a way to delete the previous batch of pics

#List of transforms to be randomly applied to images
transform = T.Compose([
    T.RandomApply([
        T.ColorJitter(brightness=0.5)
    ], p=0.3),
    T.RandomApply([
        T.RandomRotation(20)
    ], p=0.3),
    T.RandomApply([
        T.RandomPerspective()
    ], p=0.2),
    T.RandomApply([
        T.Lambda(lambda img: T.GaussianBlur(kernel_size=(randrange(1,15,2), randrange(1,15,2)))(img))
    ], p=0.2),
    T.RandomVerticalFlip(p=0.3),
    T.RandomHorizontalFlip(p=0.3),
    T.RandomGrayscale(p=0.3),   
])


def draw_landmarks(image, detected_image):
    if detected_image.multi_hand_landmarks:
         for landmark in detected_image.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, landmark, mp_hands.HAND_CONNECTIONS)


def recv_frame(stream):
    ret, frame = stream.read()
    image = cv2.flip(frame, 1)
    return (image, ret, )

def get_frame_num(save_dir):
    #Updates iter_count to the current amount of photos in the current directory
    files = os.listdir(save_dir)
    if len(files):
        pattern =  re.compile(r'(\d+)')
        files = sorted(files, key = lambda file: int(pattern.search(file).group(1)))
        return int(pattern.search(files[-1]).group(1))+1
    return 0

#Converts ndarry to pil img
def cv2_to_pil(cv_img):
    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv_img_rgb)

#Applies the transformationt list to a PIL img
def applyTransforms(PIL_img, transforms):
    return transforms(PIL_img)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--folder", type=str, required = True,
        help="Directory name to save images. Will create a directory if not currently existing. Will be inside train_images")# EX: python gesture_labeler.py -d five -> train_images/five
    ap.add_argument("-s", "--save_count", default=10, type=int,
        help="Changes how many frames will be saved into a sequence")
    ap.add_argument("-t", "--transform", default=True, type=bool,
        help="Boolean to determine whether or not to apply transformations to images before saving them" )
    ap.add_argument("-d", "--delay", default = 0, type=int,
        help="Adds a time delay to before taking capturing the image in seconds")

    iter_count = 0 # Keeps track of how many images have alr been saved in this directory
    capture_requested = False
    capture_time = -1
    args = vars(ap.parse_args())
    save_frame_count = args["save_count"]
    save_dir = os.path.join("train_images", args["folder"])
    vs = cv2.VideoCapture(0)

    #Opening Webcam and letting it warm up
    if not vs.isOpened():
        raise IOError("Cannot Open Webcam")
    time.sleep(2.0)
    cv2.namedWindow("Hand Recognition")

    #Creates the labeled directory if not already made
    if  not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    iter_count = get_frame_num(save_dir)

    with mp_hands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.5, max_num_hands=1) as hands:
        while True:
            image, ret = recv_frame(vs)
            #If frame was not returned
            if not ret:
                break
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            key = cv2.waitKey(1)
            detected_image = hands.process(rgb_frame)
            draw_landmarks(image, detected_image)
    
            #Saves frames when "C" is pressed
            if key == ord("c"):
                capture_requested = True
                #Accounting for delay 
                if args["delay"]:
                    capture_time = time.time() + args["delay"] 
                else:  
                    capture_time = 0

            if capture_requested and (capture_time - time.time()) <= 0:
                capture_requested = False
                pil_img = cv2_to_pil(image)
                for i in range(save_frame_count):
                    #Saves the first image without any transformations and if transform is false, they are also not applied
                    if i == 0 or not args["transform"]:
                        cv2.imwrite(f"{save_dir}\\{args['folder']}-frame{iter_count}.jpg", image)
                        print(f"Saved {args['folder']}-frame{iter_count}.jpg")
                        iter_count += 1
                        continue

                    #Applies transformations and saves this iamge
                    pil_img = applyTransforms(pil_img, transform)
                    cv2.imwrite(f"{save_dir}\\{args['folder']}-frame{iter_count}.jpg", np.array(pil_img))
                    print(f"Saved {args['folder']}-frame{iter_count}.jpg")
                    iter_count += 1

            if key == ord("d"):
                for i in range(save_frame_count):
                    iter_count -= 1
                    os.remove(f"{save_dir}\\{args['folder']}-frame{iter_count}.jpg")
                    print(f"Deleted {args['folder']}-frame{iter_count}.jpg")
                
            elif key == ord("q"):
                #Removing the directory if no images were saved
                if not os.listdir(save_dir):
                    os.rmdir(save_dir)
                break
            cv2.imshow("Hand Recognition",image)
    vs.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
	main()
	exit(0)
