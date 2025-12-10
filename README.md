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
2. **Classical ML Approach:**  Logistic regression model 
3. **Deep Learning Approach:**  CNN trained end-to-end


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
The naive baseline has an accuracy of nearly 0% due to the large amount of classes

---

## Visual Interface Demo


Video demo of the project can be found here  
Streamlit site can be found here  

---

## Results and Conclusions  


---

## Ethics Statement  


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

6. 


