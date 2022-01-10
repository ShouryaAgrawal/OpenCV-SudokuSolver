import cv2

import SudokuCode

gg = [[0, 0, 0, 0, 8, 0, 0, 0, 5],
 [0, 0, 1, 9, 0, 0, 0, 0, 0],
 [7, 0, 0, 0, 0, 0, 0, 1, 6],
 [0, 0, 0, 1, 0, 0, 6, 0, 9],
 [3, 0, 5, 0, 0, 0, 1, 0, 4],
 [6, 0, 8, 0, 0, 3, 0, 0, 0],
 [5, 6, 0, 0, 0, 0, 0, 0, 2],
 [0, 0, 0, 0, 0, 5, 4, 0, 0],
 [9, 0, 0, 0, 2, 0, 0, 0, 0]]



a = SudokuCode.runSS(gg.copy())
print(type(a))
for r in range(9):
    print("\n")
    for c in range(9):
        print(a[r][c], end="   ")

empimg = cv2.imread("emptybrd.jpg")
empimg = cv2.resize(empimg,(630,630))

print(empimg.shape)
rSpl = int(empimg.shape[0] /9)
cSpl = int(empimg.shape[1] /9)

for x in range(9):
    for y in range(9):
        if gg[x][y] != 0:
            print(x,y,gg[x][y] )
            cv2.putText(empimg,text = str(gg[x][y]),org = ( int((y+0.3)*rSpl),x*cSpl+int(cSpl/2)+15),
                        fontFace= cv2.FONT_HERSHEY_COMPLEX, fontScale=1.55, color = (0,0,255),thickness=2)#,bottomLeftOrigin=cv2.LINE_AA)
        elif gg[x][y] != 0:
            print(x,y,gg[x][y] )
            cv2.putText(empimg,text = str(gg[x][y]),org = ( int((y+0.3)*rSpl),x*cSpl+int(cSpl/2)+15),
                        fontFace= cv2.FONT_HERSHEY_COMPLEX, fontScale=1.55, color = (0,255,0),thickness=2)
cv2.imshow("emp",empimg)
cv2.waitKey(0)

