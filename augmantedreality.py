import cv2
import numpy as np


cap = cv2.VideoCapture(0)
imgtarget =cv2.imread('img2.jpg')
myvideo = cv2.VideoCapture('video1.mp4')
ht,wt,ct = imgtarget.shape

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(imgtarget,None)
#imgtarget = cv2.drawKeypoints(imgtarget,kp1,None)




while True:
    success1,framewebcam = cap.read()
    success2,framemyvideo = myvideo.read()
    imgaug = framewebcam.copy()
    if success2:
        framemyvideo = cv2.resize(framemyvideo,(wt,ht))
    kp2, des2 = orb.detectAndCompute(framewebcam,None)
    #framewebcam = cv2.drawKeypoints(framewebcam,kp2,None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    print(len(good))
    imgfeatures = cv2.drawMatches(imgtarget,kp1,framewebcam,kp2,good,None,flags = 2)

    if len(good)>25:
        srcpts =np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dstpts =np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        matrix,mask = cv2.findHomography(srcpts,dstpts,cv2.RANSAC,5)
        print(matrix)
        pts = np.float32([[0,0],[0,ht],[wt,ht],[wt,0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,matrix)
        img2 = cv2.polylines(framewebcam,[np.int32(dst)],True,(255,0,255),3)
        cv2.imshow('img2',img2)
        imgwarp =cv2.warpPerspective(framemyvideo,matrix,(framewebcam.shape[1],framewebcam.shape[0]))
        cv2.imshow('imgwarp',imgwarp)
        masknew = np.zeros((framewebcam.shape[0],framewebcam.shape[1]),np.uint8)
        cv2.fillPoly(masknew,[np.int32(dst)],(255,255,255))
        cv2.imshow('masknew',masknew)
        maskInv = cv2.bitwise_not(masknew)
        imgaug = cv2.bitwise_and(imgaug,imgaug,mask=maskInv)
        imgaug = cv2.bitwise_or(imgwarp,imgaug)
        
        cv2.imshow('imgaug',imgaug)
    cv2.imshow('imgfeatures',imgfeatures)
    #cv2.imshow('imgtarget',imgtarget)
    #cv2.imshow('framewebcam',framewebcam)
    #cv2.imshow('framemyvideo',framemyvideo)
    if  cv2.waitKey(1)& 0xFF == ord('q'):
        myvideo.release()
        cap.release()
        cv2.destroyAllWindows()



