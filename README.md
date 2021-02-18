# CamFilter
This is a script to add a face-mask filter on the face in video feed capture through the web cam.
In the script at first the face in the video feed is detected through opencv. Then using dlib facial landmarks are detected and using this landmark the face-mask image is 
resized and rotated. Then the resized mask is added over the face.
# Install Packages
Required packages to run the script are installed using the required.txt file./n
In terminal run : pip install -r required.txt
# Run Script
In the terminal run the script using:\n
python Filter_App.py
and to close the application use esc button.
