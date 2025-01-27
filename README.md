# Computer_Vision

Name: Leaksmy Heng
Travel days: 3
Operation system: Window
IDE: Visual Studio
Link to Video demo: https://drive.google.com/file/d/1PkoV7ca2gBN_UH9Ptcmgfyv0U0rNFkLd/view?usp=sharing

- Steps by steps on how to run in Window:
   1. Download and install opencv into your operation systems
   2. Go to your Environment Variables by searching for "Edit the systems environment variables" then click on it and go to "Environment Variables..." under Advanced. In "Environment Variables...", navigate to System variables then click "Path" and pressed "Edit". Then copy the bin and lib of your opencv to that path. Mine looks like:
        - C:\opencv\build\x64\vc16\bin
        - C:\opencv\build\x64\vc16\lib
   Then click "OK" and closed the tab
   3. Download and install CMake. Once install successfully, you should see your CMake in the Environment Variables directory described in section 2. Mine looks like:
        - C:\Program Files\CMake\
   4. Download and install Visual Studio.
   5. Once install visual studio successfully, go to "Extensions" then installs all the extensions listed below:
        - C/C++
        - CMake Tools
        - C/C++ Extension Pack
    6. Now you can create project and run your code in Visual Studio Code. If you face any error while running the code like nmake file not found, make sure you install that first by checking the Microsoft Visual Studio directory. Mine is located in:
    C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\bin\Hostx64\x64. 
    If you do not have that, make sure to install nmake. I installed it by downloading Visual Studio (Visual Studio not Visual Studio Code) then install the "Desktop development with C++" as it listed Clang and CMake in there.
