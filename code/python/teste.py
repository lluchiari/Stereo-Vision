mport numpy as np
import cv2
import glob

# Number of corners to be used on the algorithm
chessC1 = 9
chessC2 = 6

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
print("Epsulon + Max Iter, x, y)", criteria)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessC1*chessC2,3), np.float32)
objp[:,:2] = np.mgrid[0:chessC1,0:chessC2].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('/home/lluchiari/Documents/opencv/samples/data/left0*.jpg')

for fname in images:
    img = cv2.imread(fname)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (chessC1,chessC2),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (chessC1,chessC2), corners, ret)
        cv2.imshow('img',img)
        # cv2.waitKey(500)

cv2.destroyAllWindows()

# The calibration parameter itself
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
# print("Ret: ", ret, "\n")
# print("mtx: ", mtx, "\n")
# print("dist: ", dist, "\n")
# print("rvecs: ", rvecs, "\n")
# print("tvecs: ", tvecs, "\n")


# Undistortion
img = cv2.imread('/home/lluchiari/Documents/opencv/samples/data/left12.jpg')
print("opa\n");
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)

# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)

mean_error = 0
tot_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

print "total error: ", mean_error/len(objpoints)
