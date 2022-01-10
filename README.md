# OpenCV-SudokuSolver
Making a sudoku solver using OpenCV and Pytorch
I had already made a sudoku solver but then I thought, what if i didnt have to put in the work of typing everythign and making a mistake of where 
placing the given numbers in the wrong boxes?
Thus I learned OpenCV and used a Convolutional Neural Network to train data and recognise digits.
I then solved them using my solver and put it all in a beautiful box to help visualise


-------------------------------------------------

### Dataset can be found here : https://github.com/murtazahassan/Digits-Classification


### Image Portion

- Printed a sudoku I took from KrazyDad.com, I then took a picture of it ( CAN BE FOUND BELOW)

- Processed this image for contours and extracted only the sudoku board


### Model portion

- Initally I used the famoous MNIST Dataset but that created issues and written text varies a lot more and is normally very different for a few numbers, while typed is more or less simliar, thus I downloaded these pictures and created a custom dataset with labels.

- Then I used pytorch to make  Convolutional Neural Network and trained it for this Data Set.

- Validated and tested this model which I stored to make it easy in the future

- Ran my image thorugh this model and extracted all the given digits


### Output
- I had already made a sudoku solver which I used to solve the 2D array i passed into it

- Finally I put my solved sudoku on an empty sudoku oard which is out output

---------------------

INPUT

!<img src = "https://github.com/ShouryaAgrawal/OpenCV-SudokuSolver/blob/main/sudopic.jpeg" width = "500" height = "800">

Output 

!<img src = "https://github.com/ShouryaAgrawal/OpenCV-SudokuSolver/blob/main/sudoOut.jpeg" width = "600" height = "600">
-------------

### Credits:

- LEARN OPENCV in 3 HOURS with Python - https://www.youtube.com/watch?v=WQeoO7MI0Bs and other videos of Murtaza Hassan
- Small help from Various Other Youtube Videos
- Stack overflow
- Official Documents for Pytorch, Python, OpenCV 

----------------
