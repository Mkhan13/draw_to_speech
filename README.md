# Draw to Speech AAC Device

## Problem
Nonverbal children often rely on augmentative and alternative communication
(ACC) tools to communicate and express themselves. However, most ACC devices require
navigating menus or recognizing text or icons. This can be diLicult for young children who
communicate more easily through visual symbols or drawings. The goal of this project is to
make an interactive sketch to speech communication board that allows children to draw
simple doodles which the system can classify using deep learning and then generate the
corresponding speech output.

---

## Data Source
- **[The Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset)**
  - A collection of 50 million drawings across 345 categories
  - 28x28 grayscale images saved as a numpy array grouped by category

---

## Review of Relevant Previous Efforts and Literature  
Some examples of AAC devices are communication boards and interactive tablets
where the user points or clicks on a word to communicate. Communication boards are
rigid because it limits the user to letters, numbers, and a few key phrases. Interactive
tablets allow for more customization and interaction but still often requires the user to
navigate several menus before locating the desired word. No significant research has been done about AI driven AAC devices.

**My Contribution:**  
My project is unique to this field of study and puts a twist on current AAC devices by allowing the user to doodle the word to be said aloud.

---

## Model Evaluation Process & Metric Selection   
- **Metrics:**  
  - Accuracy
  - Precision
  - Recall
  - F1-score
- **Data Splits:** Stratified 80%/10%/10% split for train/validation/test 

All three approaches (naive, classical ML, and deep learning) are trained and evaluated on the same training, validation, and test sets. The results are compared directly against the naive baseline to quantify performance improvements

---

## Modeling Approach  
1. **Naive Baseline:** Predict the majority class
2. **Classical ML Approach:**  Logistic regression model with PCA
3. **Deep Learning Approach:**  Pretrained EfficientNet CNN


### Data Processing Pipeline  
The raw dataset consists of 345 .npy files, one per class, where each file contains all images for that class as a numpy array. For each class, up to 1500 samples were randomly selected and stored. All classes are combined and each image is reshaped into a 28×28×1 tensor for CNN compatibility. Finally, the dataset is split into training (80%), validation (10%), and test (10%) sets, which are saved as compressed .npz files containing X, y, and the class name list.

The images are then copied into a standardized folder structure under `data/processed/`:  
```
data/processed/
├── train.npz
├── val.npz
├── test.npz
```

### Models Evaluated and Model Selected  
- **Evaluated Models:**  
  | Approach           | Accuracy | Precision | Recall | F1-score | Notes                                       |
  |--------------------|----------|-----------|--------|----------|---------------------------------------------|
  | **Naive Baseline** | 0.0029   | 0.0       | 0.0029 | 0.0      | Predicts the most common class              |
  | **Classical ML**   | 0.2419   | 0.2159    | 0.2419 | 0.2157   | Significant improvement over naive baseline |
  | **Deep Learning**  | 0.7096   | 0.7123    | 0.7096 | 0.7078   | Strongest approach                          |

- **Model Selected:**  Deep Learning

### Comparison to Naive Approach  
The naive baseline predicts the most frequent class and has an accuracy of nearly 0%. This is due to the large amount of classes and balanced sampling strategy during preprocessing. The classical machine learning approach is a logistic regression model on PCA reduced image vectors and has a 24% accuracy. The accuracy was limited by the model's inability to capture the complex visual patterns. The neural network based deep learning approach was based on a pretrained EfficientNet model and significantly outperforms the other approaches. It has an accuracy of 70% with an f1 score of 0.7078 meaning it learned more spacial features that the classical ml appraoch could not capture.

---

## Visual Interface Demo


Video demo of the project can be found here  
Streamlit site can be found here  

---

## Results and Conclusions  
The deep learning approach using a pretrained EfficientNet model was selected as the final model due to its significantly better performance compared to the naive baseline and the classical machine learning approach. The deep learning model achieved an accuracy of approximately 71%, with strong precision, recall, and F1-score. These results demonstrate the model's ability to identify hand-drawn doodles across a large number of classes. In practice, the model performs well at identifying simple shapes and visually distinct objects, as well as some random but clearly drawn categories. However, it can still make incorrect guesses for more complex inputs, which is expected given the high number of classes and the variation in human doodles.

This project demonstrates the possibility of a doodle-to-speech AAC device powered by deep learning. Future improvements to this project could focus on training with a more AAC-oriented dataset that focuses more on daily vocabulary, such as common actions, needs, or emotions, rather than a large set of generic nouns. Training on more meaningful classes would improve usability of this tool. The next step is successfully integrating text-to-speech functionality with would improve the communication abilities of this device.

---

## Ethics Statement  
This project is intended solely for educational and research purposes and is not a production-ready alternative communication device. While the results demonstrate promising accuracy and have potential for improvement, any real-world use of such a tool would require rigorous validation, accessibility testing, and collaboration with AAC users and professionals.

---

## Instructions on How to Run the Code

1. Clone the Repository  
`git clone https://github.com/Mkhan13/draw_to_speech.git`  
`cd draw_to_speech`

3. Install Dependencies  
`pip install -r requirements.txt`

4. Run the Streamlit App  
`streamlit run main.py`  
The app will open in your browser  

6. Select a pen color and size and doodle in the box. Click the button to identify the doodle


