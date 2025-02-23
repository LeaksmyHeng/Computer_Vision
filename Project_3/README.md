# Computer_Vision

Name: Leaksmy Heng
Travel days: 2
Operation system: Window
IDE: Visual Studio Code
Links/URLs: https://drive.google.com/file/d/1zmXUUSit1t403kYVoDV-yO-WCWPcOL5W/view?usp=drive_link
Instructions for running your executables
     I use CMake to create an the executables. If you use Visual Studio Code, you can build the project by pressing Ctrl+c in CMakeLists.txt
     Then you can click run, the program should run fine.
     The project has 2 mode, when you just open it, it is just a regular video frame.
     You can press n | N => This will be the training mode. You will be prompt to put what kind of object is in that frame.
     You can press s | S => This will save what ever labeling you have inserted into the csv.
     You can press q | Q => This will quite the program and if you were in training mode and have not save your label, the program will save it to the csv file that is located in build/debug
     You can press i | I => This is an inferencing mode, since you have trained and provided some features before, when you were doing labeling, the system will perform the calculation (standard euclidean) to classify which object is in the current frame.
