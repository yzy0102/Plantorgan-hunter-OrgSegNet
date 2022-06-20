# Plantorgan-hunter-OrgSegNet
Plantorgan hunter: a deep learning-based framework for quantitative profiling plant subcellular morphology.

Please install requirement by "using pip install -r requirements.txt" first.

# Local Development
If you want to use the local deployment of OrgSegNet, you first need to download the "OrgSegNet_LocalDevlopment" file and the model weights file.

The model weight file download address: https://pan.baidu.com/s/15swgKnIPEfzWPsd72KfQrQ?pwd=m0vz the code is m0vz,
or you can download via google hard drive link: https://drive.google.com/file/d/1WiawifgB5moWc27rvIb7yeammdry7K9D/view?usp=sharing

You can save the downloaded weights file in the model_path folder.

Then, you can run the main.py file to get the result you want.
  
# Train Another model
If you want to train a model yourself, you can do so by using the jupyter notebook file in the file.

# A simple example
The result contains 6 parts, and the segmentation results and morphological metrics of organelle will be included in the folder named after the organelle. For example, the area and electron density parameters of a single organelle are included in Chloroplast_info.csv, and the individual serial number of the organelle is identified in Chloroplast_.png. In the shape_info folder, ShapeResultsTable.csv contains the shape information of the individual chloroplasts, visibilityGraphs.gpickle contains the original graph data of the organelles, and LabeledShapes.pn contains the individual serial numbers of the organelles.

Result folder directory:

|-Result

|----Chloroplast

|----Chloroplast_.png (where organelle number is marked)

|----Chloroplast__.tif (organelle segmentation)

|----Chloroplast_info.csv (including area and electron-density of single organelle)

|----shape_info

|----LabeledShapes.png (where organelle number is marked equal to Chloroplast_.png)

|----ShapeResultsTable.csv (including shape-complexity of single organelle)

|----visibilityGraphs.gpickle (original graph data of organelle)

|----Mitochondrion

|----......

|----Nucleus

|----......

|----Vacuole

|----......

|----1.jpg

|----2.jpg

The input image
![800nm520pixels](https://user-images.githubusercontent.com/54012483/174546721-0073ef64-c456-4acd-8017-0ac49ddc6e74.jpg)
The output segmantation result.
![image](https://user-images.githubusercontent.com/54012483/174546614-3a4ce335-f52e-4de9-946a-7c54178f8fb7.jpg)
Metrics
![image](https://user-images.githubusercontent.com/54012483/174547124-a40395c0-0668-43d7-a917-7da7480caac8.png)
![image](https://user-images.githubusercontent.com/54012483/174547402-6de439c9-802f-4cbd-ad69-c6853ae0cf21.png)


# Test images
You can grab some test images at this link! After opening the link through your browser, right-click to save the image locally.
https://github.com/yzy0102/Plantorgan-hunter-OrgSegNet/raw/main/OrgSegNet_LocalDevlopment/test_img/2000nm460pixels.jpg
https://github.com/yzy0102/Plantorgan-hunter-OrgSegNet/raw/main/OrgSegNet_LocalDevlopment/test_img/2000nm520pixels.jpg
https://github.com/yzy0102/Plantorgan-hunter-OrgSegNet/raw/main/OrgSegNet_LocalDevlopment/test_img/800nm520pixels.jpg
