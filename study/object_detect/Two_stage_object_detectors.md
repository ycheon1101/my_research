### Two-stage object detectors

#### Types of Object detectors

<li> One-stage detectors

    Image -> Feature extraction -> Classfication (class score(cat, dog, person)) + Localization(bounding box(x, y, w, h))

    - Input the entire image, and predict the class and local directly
    - Pro: simple structure, fast
    - con: low accuracity

<li> Two-stage detectors

    Image -> Feature extraction -> Extraction of object proposals -> Classfication + Localization

    - Select the region, and predict object's class and local
    - pro: High accuracity
    - con: slow

#### Localization

<li> Bounding box regression 

    Image -> Feature extraction -> Box coordinates(x, y, w, h)

    Image -> CNN -> Fully connected -> Box Coordinates + Class scores

#### Overfeat

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS691imV-dzp1qkJBCgaUCFlYlQ5uQm7wHAlQ&usqp=CAU" width = 500>

    - Sliding window + box regression + classfication
    - Image -> CNN -> Feature map -> Boxes + Class scores
    - In the feature map, there is the concept of the location
    - Sliding window -> end up with many predictions and have to combine them for a final detection
    - Use many sliding window locations and multiple scales

<img src="https://mblogthumb-phinf.pstatic.net/MjAxODAxMDJfMTUy/MDAxNTE0ODU3NDkwNzA5.rR76tQe2OLlFHmKrSK1A6kIWgvy3r7hNtst80BFmjukg.tYdeaRzpsN-pGvnHsA94h8R5ScNNrkf82seOXmEPBRIg.PNG.infoefficien/cs231n_8%EA%B0%95_Localization%2C_detection.mp4_000650866.png?type=w800" width = 500>

<li> CNN: create the feature map through sliding(convolutional layer) and reduce the size of map (pooling)

#### Multiple Objects in an image

    - Localization: Regression
    - Having a variable sized output is not optimal for Neural Networks
    - Solution: RNN(output -> input), Set prediction

#### Detect multiple objects using classification

    - Instead perform regression and predict the coordinates of bounding boxes, we will follow classification 
    - Problem: Expensive to try all possible positions, scales and aspect ratio
    - Therefore, we can try only on a subset of boxes with most potential (<- Region Proposals)

#### Region Proposals

    - Give us "interesting" regions in an image that potentially contain an object
    - Step1: Obtain region proposals
    - Step2: Classify them

#### R-CNN

<img src="https://production-media.paperswithcode.com/methods/new_splash-method_NaA95zW.jpg">

<li> Training Scheme

    1. Pre-train the CNN on ImageNet
    2. Finetune the CNN on the number of classes the detector is aiming to classify (softmax loss)
    (Finetune: Base on the trained model, change architecture to fit my image data and update it from trained model weights)
    3. Train a linear support vector machine classifier to classify image regions. One SVM per class (hinge loss)
    (SVM: find hyperplane (maximize margin between classes) )
    4. Train the bounding box regressor (L2 loss)

Summary of R-CNN

    Extract region -> merge -> wraping -> getting feature vector (Using CNN) -> predict the region location ( Using SVM(detect),Bounding box regression(control the size and location of the object) )

However, R-CNN is slow -> SPP-Net(one cnn and analysing the feature of a whole image)

#### Fast R-CNN: Rol Pooling

<img src="https://wikidocs.net/images/page/136494/RCNN2.png">

    1. Find the region of interest through selective search
    2. Extract the feature map by input the whole image on CNN
    3. Projection the ROI (fit with feature map size)
    4. ROI pooling -> getting fixed sized feature vector
    5. Feature vector is devided into 2 brunches after passing FC layer(Fully Connected Layer) 
    6. Object classification(soft max)
    7. control the location of the box(Bounding box regression)

#### Faster R-CNN
    - Region Proposal Network(RPN) trained to produce region proposals directly
    - It is same with Fast R-CNN, but when calculate ROI, it use RPN instead of selective search

#### Region Proposal Network

<img src="https://miro.medium.com/v2/resize:fit:750/1*JDQw0RwmnIKeRABw3ZDI7Q.png">

    1. Create anchor boxes
    2. Predict the anchor box
    3. Non-maximum Suppression(NMS): eliminate overlapped bounding boxes











