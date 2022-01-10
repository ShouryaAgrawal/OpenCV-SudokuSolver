import numpy as np
import cv2
import SudokuCode
import mnistNn

def getC(imgf):
    cont, hier = cv2.findContours(imgf,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(i) for i in cont]


    for i in cont:

        if cv2.contourArea(i) == max(areas):
            ind = i
            cv2.drawContours(imgc, i, -1, (0, 0, 255), 3)


    peri = cv2.arcLength(ind,True)

    approx = cv2.approxPolyDP(ind,0.02*peri,True)
    approx = [i[0] for i in approx]
    approx = np.float32(approx)
    return approx

img = cv2.imread("sudokuclean.PNG")
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBl= cv2.GaussianBlur(imgGray, (3,3),1)
imgCanny = cv2.Canny(imgBl,150,200)
imgc = img.copy()
approx = getC(imgCanny)



width = 630
height = 630
pts2 = np.float32([[0,0],[0,height],[width,height], [width,0]])
matrix = cv2.getPerspectiveTransform(approx, pts2)

imgOut = cv2.warpPerspective(img, matrix, (width, height))

#do same procss on above extracted image
imgOGray = cv2.cvtColor(imgOut,cv2.COLOR_BGR2GRAY)

### Separating boxes
def splt(game):

    boxL = []

    r = np.vsplit(game,9)
    for i in r:
        boxL.append(np.hsplit(i, 9))
    return boxL



boxL = splt(imgOGray)  ## indivisual images of eacvh box

recogGrid= []


for i in range(9):
    c = []
    for j in range(9):

        tp = boxL[i][j]
        tp = tp[5:65, 10:60]

        predd = mnistNn.run_model(tp,mnistNn.md)


        if np.max(predd) > 0.60:
            c.append(int(np.argmax(predd)))
        else:

            c.append(0)
    recogGrid.append(c)

# for i in range(9):
#     for j in range(9):
#         print(recogGrid[i][j],end = "    ")
#     print("\n")


#
# print("Solving Sudoku", end= " ")
# print(".............."*100)

# for r in range(9):
#     print("\n")
#     for c in range(9):
#         print(recogGrid[r][c], end="   ")
try:
    ans = SudokuCode.runSS(recogGrid)
except:
    pass

# for r in range(9):
#     print("\n")
#     for c in range(9):
#         print(ans[r][c], end="   ")

empimg = cv2.imread("emptybrd.jpg")
empimg = cv2.resize(empimg,(630,630))

rSpl = int(empimg.shape[0] /9)
cSpl = int(empimg.shape[1] /9)

for x in range(9):
    for y in range(9):
        if recogGrid[x][y] != 0:
            cv2.putText(empimg,text = str(recogGrid[x][y]),org = ( int((y+0.3)*rSpl),x*cSpl+int(cSpl/2)+15),
                        fontFace= cv2.FONT_HERSHEY_COMPLEX, fontScale=1.55, color = (0,0,255),thickness=2)#,bottomLeftOrigin=cv2.LINE_AA)
        else:
            cv2.putText(empimg,text = str(ans[x][y]),org = ( int((y+0.3)*rSpl),x*cSpl+int(cSpl/2)+15),
                        fontFace= cv2.FONT_HERSHEY_COMPLEX, fontScale=1.55, color = (0,255,0),thickness=2)
cv2.imshow("emp",empimg)
cv2.waitKey(0)