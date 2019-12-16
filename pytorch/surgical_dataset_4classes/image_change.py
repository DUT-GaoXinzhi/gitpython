import cv2 as cv

img = cv.imread(r"F:\gitpython\pytorch\surgical_dataset_4classes\train\forceps1\IMG_20191014_203948.jpg")
res = cv.resize(img,(512,512))
while True:
    k = cv.waitKey(5)
    cv.imshow("x", res)
    if k == 'q':
        cv.destroyAllWindows()