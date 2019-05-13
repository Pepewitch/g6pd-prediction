
import cv2
from crop import getCropRect, findCircle
import numpy as np
from config import segment_directory, image_directory
from os import listdir, mkdir
from os.path import join, isdir

def drawCircles(img):
    circles = findCircle(img)
    for x ,y ,r in circles:
        cv2.circle(img , (int(x),int(y)) , int(r) , (0,0,255) , 2)

def processFrame(bgrFrame):
    rgb = cv2.cvtColor(bgrFrame, cv2.COLOR_BGR2RGB)
    cropRect, coordinate = getCropRect(rgb)
    cv2.polylines(bgrFrame, np.array([coordinate], dtype=np.int32), isClosed=True, color=(0,255,0), thickness=3)
    drawCircles(cropRect)
    bgr = cv2.cvtColor(cropRect, cv2.COLOR_RGB2BGR)
    return bgr

def saveImage(before, after):
    if not isdir(image_directory):
        mkdir(image_directory)
    if not isdir(segment_directory):
        mkdir(segment_directory)
    image_list = list(filter(lambda name: '.ppm' in name, listdir(image_directory)))
    segment_list = list(filter(lambda name: '.ppm' in name, listdir(segment_directory)))
    next_image = 0 if len(image_list) == 0 else int(max(image_list, key=lambda name: int(name.split('.')[0])).split('.')[0]) + 1
    next_segment = 0 if len(segment_list) == 0 else int(max(segment_list, key=lambda name: int(name.split('.')[0])).split('.')[0]) + 1
    cv2.imwrite(join(image_directory, '{}.ppm'.format(next_image)), before)
    cv2.imwrite(join(segment_directory, '{}.ppm'.format(next_segment)), after)

def show(before, after):
    cv2.imshow('after', after)
    cv2.imshow('before', before)

def useCvCamera(process):
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        after = process(frame)
        # Display the resulting frame
        show(frame, after)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord("c"):
            saveImage(frame,after)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def usePiCamera(process):
    # import the necessary packages
    from picamera.array import PiRGBArray
    from picamera import PiCamera
    import time
    
    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))
    
    # allow the camera to warmup
    time.sleep(0.1)
    
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array
        after = process(image)
        # show the frame
        show(image, after)
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
    
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        if key == ord("c"):
            saveImage(image,after)
        

def openCamera(process, piCamera=False):
	if piCamera:
		try:
			usePiCamera(process)
		except ModuleNotFoundError as e:
			useCvCamera(process)
	else:
		useCvCamera(process)

if __name__ == "__main__":
    openCamera(processFrame)