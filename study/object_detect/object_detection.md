# Object Detection

- Traditional Object detection methods
  
    <br> 1. Template matching + sliding window
   
       Template: a small image describes an object we want to detect

       - Located template to top left corner of the image and sliding it all through the image 
       - Then, for every position, we can evaluate how much do the pixels in the image and template correlate
    
        Problem

          - If the object covered partially with the box, then it matches low correlation with the template
          - If the pose of the object is different, it cannot detect
          - Because objects have an unknown position, scale and aspect ratio, the search space is searched inefficiently with sliding window
        
        Condition to use this method

          - Occulusion: need to see the WHOLE object
          - This works to detect a given instance of an object but not a class of objects
  
    <br> 2. Feature extraction + classification
        
        - Learning multiple weak learners to build a strong classifier
        - Makes many small decisions and combine them for a stronger final decision

        Step for feature extraction

        1. Select Haar-like features
        2. Integral image for fast feature evaluation
        3. AdaBoost(weak learning -> make strong learning model) for find weak learner


        Feature extraction(Haar-like feature)
        
        <img src="data:image/png;base64,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">

    Histogram of Oriented Gradients

    <img src="https://ars.els-cdn.com/content/image/1-s2.0-S1007021411700323-gr1.jpg" width=400 height=300>

    - Gradient: the direction of greatest change of the image

    - Average gradient image over training samples -> gradients provides shape information
    - Compute gradients in dense grids, compute gradients and create histogram based on gradient direction

    Step for classification

      1. Choose your training set of images that contain the object you want to detect
      2. Choose a set of images that do not contain that object
      3. Extract HOG features on both sets
      4. Train an SVM(Support Vector Machine) classifier on the two sets to detect whether a feature vector represents the object of interest or not

  <br> 3. Deformable Part Model

        - Decomposes objects into distinct parts, assuming that each part can undergo independent deformations
        - Structs the part model (including information of shape and position of the parts) and describe how they are relatively arranged within the object
        - can control various size and diformation
        - Sliding Window Detection

    - Example: Detect each body part independently and then combine them. If they fit together, then we can say there is a person in image   

    
  

  <br> 4. <span style="font-size: 100%">Non-Maximum Suppression</span>

    - Many boxes trying to explain one object
    - need a method to keep only the best boxes (Repeat to find reliability box and remove overlaping boxes)

    <br><img src="https://miro.medium.com/v2/resize:fit:1200/1*6d_D0ySg-kOvfrzIRwHIiA.png">

    Algorithm

        1. box i (loop)
        2. box j (nested loop)
        3. If they overlap, discard box i if the score(reliability) is lower than the score j

    Region overlap
    
        - Measure region overlap with the Intersection over Union or Jaccard Index

    <img src="https://b2633864.smushcdn.com/2633864/wp-content/uploads/2016/09/iou_equation.png?lossy=2&strip=1&webp=1">