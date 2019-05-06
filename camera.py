
import cv2
from crop import getCropRect

def processFrame(bgrFrame):
    rgb = cv2.cvtColor(bgrFrame, cv2.COLOR_BGR2RGB)
    cropRect = getCropRect(rgb)
    bgr = cv2.cvtColor(cropRect, cv2.COLOR_RGB2BGR)
    return bgr

def show(before, after):
    cv2.imshow('crop', after)
    cv2.imshow('real', before)

def useCvCamera():
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        bgr = processFrame(frame)
        # Display the resulting frame
        show(frame, bgr)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def usePiCamera():
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
        bgr = processFrame(image)
        # show the frame
        show()
        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
    
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

def openCamera():
    try:
        usePiCamera()
    except ModuleNotFoundError as e:
        useCvCamera()

if __name__ == "__main__":
    openCamera()