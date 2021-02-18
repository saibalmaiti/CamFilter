import cv2
import dlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_nose_to_chin_distance(chin_point, nose_point, chin_bottom_point):
    return np.linalg.norm(np.cross(np.array(chin_bottom_point)-np.array(nose_point), np.array(nose_point)-np.array(chin_point)))/np.linalg.norm(np.array(chin_bottom_point)-np.array(nose_point))

def masked_image(frame,mask_img,nose_v,chin_bottom_v,chin_left_point,chin_right_point):
    #process mask image
    height = mask_img.height
    width = mask_img.width
    width_ratio = 1.2
    new_height = int(np.linalg.norm(nose_v - chin_bottom_v))
    mask_left_img = mask_img.crop((0, 0, width // 2, height))
    mask_left_width = int(get_nose_to_chin_distance(chin_left_point, nose_point, chin_bottom_point) * width_ratio)
    mask_left_img = mask_left_img.resize((mask_left_width, new_height))
    mask_right_img = mask_img.crop((width//2, 0, width, height))
    mask_right_width = get_nose_to_chin_distance(chin_right_point,nose_point, chin_bottom_point)
    mask_right_width = int(mask_right_width * width_ratio)
    mask_right_img = mask_right_img.resize((mask_right_width,new_height))
    mask_size = (mask_left_width + mask_right_width, new_height)
    mask = Image.new('RGBA', mask_size)
    mask.paste(mask_left_img, (0, 0))
    mask.paste(mask_right_img,(mask_left_width,0))
    angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
    rotated_mask_img = mask.rotate(angle, expand=True)
    # calculate mask location
    center_x = (nose_point[0] + chin_bottom_point[0]) // 2
    center_y = (nose_point[1] + chin_bottom_point[1]) // 2

    offset = mask.width // 2 - mask_left_img.width
    radian = angle * np.pi / 180
    box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
    box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(frame)
    im_pil.paste(mask, (box_x, box_y), mask)
    frame = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)
    return frame


detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
mask_img = Image.open("default-mask.png") 
cap = cv2.VideoCapture(0)
_, frame = cap.read()
rows, cols, _vid= frame.shape
dog_mask = np.zeros((rows, cols))
while True:
    _, frame = cap.read()
    dog_mask.fill(0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        nose_bridge = []
        chin_point = []
        for n in range(0, 68): 
            x = landmarks.part(n).x 
            y = landmarks.part(n).y
            if(n in range(27,31)):
                nose_bridge.append((x,y))
            elif(n in range(0,17)):
                chin_point.append((x,y))
        nose_point = nose_bridge[len(nose_bridge) * 1// 4]
        nose_v = np.array(nose_point)
        chin_len = len(chin_point)
        chin_bottom_point = chin_point[chin_len // 2]
        chin_bottom_v = np.array(chin_bottom_point)
        chin_left_point = chin_point[chin_len // 8]
        chin_right_point = chin_point[chin_len * 7 // 8]
        #process mask image
        frame = masked_image(frame,mask_img,nose_v,chin_bottom_v,chin_left_point,chin_right_point)
        cv2.circle(frame, (chin_bottom_v[0],chin_bottom_v[1]), 2, (255, 255, 0), -1)
        cv2.circle(frame, (chin_left_point[0],chin_left_point[1]), 2, (255, 255, 0), -1)
        cv2.circle(frame, (chin_right_point[0],chin_right_point[1]), 2, (255, 255, 0), -1)
        cv2.circle(frame, (nose_v[0],nose_v[1]), 2, (255, 255, 0), -1)
        
        
        
    cv2.imshow("Frame", frame) 
    key = cv2.waitKey(1) 
    if key == 27: 
        break
cv2.destroyAllWindows() 

