### Object tracking

#### GoTurn

<img src="https://t1.daumcdn.net/cfile/tistory/995BF13C5BDB0EA203">

previous frame
1. find the bounding box for target and crop
2. set the target
3. extract the feature by using CNN, and create convolutional layers

current frame
1. set the search region near the target we got it on the previous frame and crop. It includes prediction of moving
2. extract the feature by using CNN, and create convolutional layers
3. compare previous convolutional layers and current convolutional layers to track the object
4. predict the location of tracking object by using Fully-Connected Layers

Pros:

- No oline training required
- Tracking is done by comparison. So we do not need to retain or finetune our model for every new object.
- Close to the template matching
- This makes it very fast

Cons:

- If the object moves fast and goes out of our search window, we cannot recover

#### Unsupervised

<img src="https://raw.githubusercontent.com/594422814/UDT/master/UDT.png">

- Forward cycle and backward cycle should be consistent
- Forward cycle: predicting the location of the tracking target in the next frame based on the information from the current frame
- Backward cycle: last frame -> previous frame, utilizing the tracking target's past movements and features to predict its current and future positions
- start point in time t -> backward -> forward with prediction. It should be in the same location with start point. If not, there is loss.

#### MDNet

- Online appearance model learning entails training your CNN at test time
- Slow: not suitable for real-time application
- Solution: train as few layers as possible

<img src="https://cvlab.postech.ac.kr/research/mdnet/images_/mdnet_.png">

- Shared layers + Scene-specific layers
- Backpropagation is independent per sequence
- #k sequence -> #k domain
- At test time, we need to train fc6
- finetuning


Online tracking

1. Detect the object
2. Draw target candidates
3. Find the optimal state
4. Collect training samples
5. Update the CNN(MDNet) if needed
6. Repeat for the next frame

Pros:
- No previous location assumtion, the object can move anywhere in the image
- Fine-tuning step is comparatively cheap

Cons:
- Not as fast as GoTurn

#### ROLO (Recurrent YOLO)
<img src="https://camo.qiitausercontent.com/4b63866cdeedfca9037f48235d61bb404f017cfc/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f3131363730362f39346365353066322d373861642d333635362d373863612d3065316432623537353039312e706e67">

- CNN for appearance + LSTM for motion
- LSTM receives the heatmap for the object's position and descriptor of the image
- Getting visual feature from YOLO

### Multiple object tracking

<li> Online tracking

- Processes two frames at a time
- For real-time application
- Hard to recover from errors or occlusion

<li> Offline tracking

- Processes a batch of frames
- Good to recover from occlusion
- Not suitable for real-time application
- Suitable for video analysis

#### Online tracking

1. Track initialization (At time t, it cannot predict the future, because it is online tracking)
2. At time t + 1, prediction of the next position(motion model)
3. Matching prediction with detections(appearance model)

- Bipartite matching
  - Define distances between boxes(ex. IoU, pixel distance, 3D distance)
  - If prediction and detection matches well, then the pixel distance will be smaller.
  - Need one-to-one matching
  - Problem: 
  - What happens if we are missing a prediction?
  - What happens if no prediction is suitable for the match?
  - To solve these problems, we have to introduce extra nodes with a threshold cost -> no match anything

#### Tracktor

- A method trained as a detector but with tracking capabilities

- There are three objects and we already detect them when frame t. Therefore, we know the locations of objects exactly. And we will use detections of frame t as proposals of frame t + 1. -> Bounding box regression

- Pros:
  - we can reuse an extremely well-trained regressor(getting well-positioned bounding boxes)
  - We can train our model on still images -> easier annotation
  - Tracktor is online method(Fast)

- Cons:
  - There is no notion of "identity" in the model (confusion in crowded spaces)
  - As any online tracker, the track is killed if the target becomes occluded
  - Therefore, to solve the problem 1 & 2, re-identify is needed.
  - The regressor only shifts the box by a small quantity
    - Large camera motions
    - Large displacements due to low framerate

  - Motion model can solve the problem 2 & 3

#### Re-ID(modeling appearance)

- Viewing tracking as a retrieval problem
- Detected person -> probe -> retrieve the top matches
- How to measure the distance between two images?
  - Classification and check whether they are same or not.
    - Similarity Learning: learn a function that measure how similar two objects are(iphone face)
    - Deep Metric Learning: Learning a distance function over objects

#### Similarity learning

<li> Siamese network = shared weighed

- image A -> CNN -> FC -> f(A)
- image B -> CNN -> FC -> f(B)

<li> We use the same network to obtain an encoding of the images, f(A) and f(B)

<li> To be done: compare the encodings

<li> losses

- Distance function d(A,B) = ${||f(A) - f(B)||^2}$
- Training:
  - If A and B depict the same person, d(A,B) is small
    - Loss function: L(A,B) = ${||f(A) - f(B)||^2}$ 
  - If A and B depict a different person, d(A,B) is large
    - Hinge loss: L(A,B) = max(0, ${m^2 - ||f(A) - f(B)||^2}$) (If second value is negative, max will be 0. It means there is any loss)
- Triplet loss(3 images)
  - Anchor, negative, positive
  - allows us to learn a ranking
  - If the distance between anchor and negative is smaller than distance between anchor and positive, we should change it to make distance between anchor and positive is smaller than distance between anchor and negative by training
