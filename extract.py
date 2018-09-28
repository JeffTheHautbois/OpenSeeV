import numpy as np
import cv2

def auto_T(m_img, sigma = 0.33):
    v = np.median(m_img)

    lower_T = int(max(0,(1.0-sigma)*v))
    upper_T = int(min(255,(1.0+sigma)*v))
    imgEdge = cv2.Canny(m_img,lower_T,upper_T)
    
    return imgEdge


cap = cv2.VideoCapture('vtest.avi') 
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()


while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    blurred_mask = cv2.GaussianBlur(fgmask,(3,3),0)
    canny_edge = auto_T(blurred_mask)

    _, contours, _ = cv2.findContours(canny_edge.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

    #cv2.drawContours(fgmask, contours,-1,(0,255,0), 3)
    
    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()