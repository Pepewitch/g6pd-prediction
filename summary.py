import numpy as np
import cv2

def getInsideCircle(im , circle , crop=0.5 , debug=False):
    x , y , r = circle
    mask = np.zeros(im.shape , dtype=np.uint8)
    r = int(crop*r)
    cv2.circle(mask , (int(x),int(y)) , r , (255,255,255) , -1)
    if debug:
        plt.figure(figsize=(10,10))
        plt.title('Inside circle')
        plt.imshow(mask & im)
        plt.show()
    selected_r = im[:,:,0][mask[:,:,0] == 255]
    selected_g = im[:,:,1][mask[:,:,1] == 255]
    selected_b = im[:,:,2][mask[:,:,2] == 255]
    return { 
        'avg_r': np.average(selected_r), 
        'std_r': np.std(selected_r),
        'avg_g': np.average(selected_g), 
        'std_g': np.std(selected_g),
        'avg_b': np.average(selected_b), 
        'std_b': np.std(selected_b),
    }

def getOutsideCircles(im , circles , extend=1.2 , border=(40,0,0,0) , debug=False):
    mask = np.zeros(im.shape , dtype=np.uint8)
    for circle in circles:
        x , y , r = circle
        r = int(extend * r)
        cv2.circle(mask , (int(x) , int(y)) , r , (255,255,255) , -1)
    cv2.rectangle(mask , (0,0) , (im.shape[1] , border[0]) , (255,255,255) , -1)
    cv2.rectangle(mask , (im.shape[1]-border[1],0) , im.shape[:2][::-1] , (255,255,255) , -1)
    cv2.rectangle(mask , (0,im.shape[0]-border[2]) , im.shape[:2][::-1] , (255,255,255) , -1)
    cv2.rectangle(mask , (0,0) , (border[3] , im.shape[0]) , (255,255,255) , -1)
    if debug:
        plt.figure(figsize=(10,10))
        plt.title('Background')
        plt.imshow(~mask & im)
        plt.show()
    selected_r = im[:,:,0][mask[:,:,0] == 0]
    selected_g = im[:,:,1][mask[:,:,1] == 0]
    selected_b = im[:,:,2][mask[:,:,2] == 0]
    return { 
        'avg_r': np.average(selected_r), 
        'std_r': np.std(selected_r),
        'avg_g': np.average(selected_g), 
        'std_g': np.std(selected_g),
        'avg_b': np.average(selected_b), 
        'std_b': np.std(selected_b),
    }

def getPixelSummary(img_dict, label, imageTransform=None, debug=False):
    label_dict = label.to_dict()['Ac/Hb']
    summary = {'name': [] , 'label': []}
    for j in ['r','g','b']:
        for i in range(1,4):   
            summary['avg_'+j+str(i)] = []
            summary['std_'+j+str(i)] = []
        summary['background_avg_'+j] = []
        summary['background_std_'+j] = []
    for name , im in img_dict.items():
        if imageTransform is not None:
            im['image'] = imageTransform(im['image'])
        summary['name'].append(name)
        if debug:
            plt.figure(figsize=(10,10))
            plt.title('Original : ' + name)
            plt.imshow(im['image'])
            plt.show()
        outside = getOutsideCircles(im['image'] , im['circles'] , debug=debug)
        for color in ['r' , 'g' , 'b']:
            summary['background_avg_'+ color].append(outside['avg_'+ color])
            summary['background_std_'+ color].append(outside['std_'+ color])
        for index , circle in enumerate(im['circles']):
            inside = getInsideCircle(im['image'] , circle , debug=debug)
            for color in ['r' , 'g' , 'b']:
                summary['avg_'+color+str(index+1)].append(inside['avg_'+color])
                summary['std_'+color+str(index+1)].append(inside['std_'+color])
        label_key = int(name.split('.')[0][6:])
        if label_key in label_dict:
            summary['label'].append(label_dict[label_key])
        else:
            summary['label'].append(np.nan)
    return summary
