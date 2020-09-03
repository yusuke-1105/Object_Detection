# Object Detection  
このドキュメントは[日本語](https://github.com/yusuke-1105/Object_Detection)もご利用いただけます。  
[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)
[![I like Docker](https://img.shields.io/badge/I%20like-Docker%20-blue)]()
[![Python](https://img.shields.io/badge/Python-%20-red)]()  

## About This  

This is a program for face recognition and object recognition.  
It uses the [OpenVINO™ Toolkit](https://01.org/openvinotoolkit) and [YOLO v3](https://pjreddie.com/darknet/yolo/).  
OpenVINO makes us easier to use image processing programs and interfaces by deep learning.  
Please refer to the [official website](https://www.intel.co.jp/content/www/jp/ja/internet-of-things/solution-briefs/openvino-toolkit-product-brief.html) for more information.   
OpenVINO is available for download at [software.intel.com](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html).  
However, it is a little bit complicated, so I have written a simple guide at [How to Run](#How_to_Run) below.

The program part of this repository consists of the following  

|folder name|contents|
|:---:|:---:|
|`Models`|models and label list|
|`Post_Image`|processed images|
|`Pre_Image`|image before being processed|
|`main1.py`|a program to recognize faces|
|`main2.py`|a program to detect objects|
|`Docker`|files being concerned with docker|


## How to Run  
### METHOD 1  
#### [Intel DevCloud for the Edge](https://devcloud.intel.com/edge/)  
This is a cloud service provided by Intel. OpenVINO is installed beforehand, and you can perform image processing and inference using OpenVINO without building an environment on your own PC.  
First, visit the official website above.  

The figure below shows the procedure.  
If you have not yet registered step 4, you must do so. After registration, you will receive an email like the one below.  
In step 5, <b>check the box</b> for "Use JupyterLab".  
Proceed to step 8. 

<div align="center">
<img src="https://github.com/yusuke-1105/Object_Detection/blob/master/Other/pic1.png">
</div>  

<div align="center">
<img src="https://github.com/yusuke-1105/Object_Detection/blob/master/Other/pic2.png">
</div>  

Enter the following in the terminal. (You can copy and paste, then enter(return))  
```
git clone https://github.com/yusuke-1105/Object_Detection
```  
Then "Object_Detection" will appear in the folder as shown below, and you can click on it to see the contents.  

<div align="center">
<img src="https://github.com/yusuke-1105/Object_Detection/blob/master/Other/pic3.png">
</div>  

Then enter the following command.  
```
cd Object_Detection
wget https://www.dropbox.com/s/6dlv9448ssy8lki/Archive.zip?dl=0 -O Models/Archive.zip
unzip Models/Archive.zip -d Models/
```
Now upload your favorite images to the `Pre_Image` folder and you can upload them all at once from Finder (Mac) or Explorer (Windows) by dragging.  
<div align="center">
<img src="https://github.com/yusuke-1105/Object_Detection/blob/master/Other/Image.GIF">
</div>  

When you have finished uploading your images, return to the terminal and enter the following.  
```
python3 face_recognization.py
```
Then face-recognized image will be saved in `Post_Image`.  
We do as dexcribed above.  
```
python3 object_detection.py
```
Then object-detected image will be saved in `Post_Image`.  

#### Memo  
If you want to delete all the folders in Intel Dev Cloud for the Edge, just type the following command.  
```
rm -rf "folder name"
```  

### METHOD 2  
By using Docker and Visual Studio Code, you can run your program in your local environment. I'll show you the details here.  
if you encounter errors, please try searching.  
- [Docker Desktop](https://www.docker.com/products/docker-desktop)  
To get started, you'll need to install Docker Desktop.  

- [Visual Studio Code](https://code.visualstudio.com/download)(VS Code)  
Next, install VS Code and Docker in Extension. (For checking.)  
<div align="center">
<img src="https://github.com/yusuke-1105/Object_Detection/blob/master/Other/Docker_Extension.png">
</div>  
You should also install Remote Development. This is a very easy to use extension for development with Docker.  
<div align="center">
<img src="https://github.com/yusuke-1105/Object_Detection/blob/master/Other/Dev_Container.png">
</div>  

After installing Remote Development, you do like below.
```
git clone https://github.com/yusuke-1105/Object_Detection
cd Object_Detection
wget https://www.dropbox.com/s/6dlv9448ssy8lki/Archive.zip?dl=0 -O Models/Archive.zip
unzip Models/Archive.zip -d Models/
```
And move the ".devcontainer" under the `Docker` directory to the main directory (where this README and programs are located). `dockerfile` under the `Docker` directory, line 43
```docker
ARG package_url=<your_package_url>
````  
Enter the URL obtained from the [Intel® Distribution of OpenVINO™ Toolkit Download Page](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/choose-download.html) in the `<your_package_url>` area. You don't need "<>".  
After typing, open the VS Code command palette (Ctrl(Command)+shift+P). Then click on "Open Folder in Container".    

<div align="center">
<img src="https://github.com/yusuke-1105/Object_Detection/blob/master/Other/docker_extension_func.png">
</div>  

Select the folder as appropriate. Then the Docker build will start, wait a little (30-90 minutes) and the build will be complete and the program will be ready to use in your PC.  

## About This Model  
These model was converted from YoloV3 to IR to be able to use with OpenVINO. The conversion program is [yolo_to_IR.ipynb](https://github.com/yusuke-1105/Object_Detection/blob/master/Colab/yolo_to_IR.ipynb) under the `Colab` directory, and you can use it in Google Claboratory or something like it.  
There is also a tiny version of yolo, which has a smaller model size and is faster for inference . (However, the accuracy is lower.)  
If you want to use it, you can use and find instructions in the above file.

## References  
#### English  
[YOLO and Tiny-YOLO object detection on the Raspberry Pi and Movidius NCS](https://www.pyimagesearch.com/2020/01/27/yolo-and-tiny-yolo-object-detection-on-the-raspberry-pi-and-movidius-ncs/)  
[open_model_zoo](https://github.com/openvinotoolkit/open_model_zoo)  
[Converting a TensorFlow* Model](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html)  
[Install Intel® Distribution of OpenVINO™ toolkit for Linux* from a Docker* Image](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_docker_linux.html)  
[Introduction to Intel® Deep Learning Deployment Toolkit](https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_Introduction.html)  
#### Japanese  
[YOLOv3 を OpenVINO™ ツールキットで使用する IR 形式へ変換してみよう](https://macnicago.zendesk.com/hc/ja/articles/360042709871-YOLOv3-を-OpenVINO-ツールキットで使用する-IR-形式へ変換してみよう)  