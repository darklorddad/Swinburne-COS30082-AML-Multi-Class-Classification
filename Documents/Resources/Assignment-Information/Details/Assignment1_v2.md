### COS30082 Assignment: Bird Species Classification

**Weighting:** 20%
**Assignment Type:** Individual

---

#### **1. Details**

This assignment requires you to engage in multi-class classification to solve the problem of identifying bird species. You are permitted to use any of the techniques you have learned in the course.

---

#### **2. Dataset**

The dataset to be used is the **Caltech-UCSD Birds 200 (CUB-200)**. This dataset contains images of 200 different bird species, primarily from North America. The training set is comprised of 4829 images. You can download the necessary images and annotation files from the Canvas assignment page.

For more information, you can visit the official dataset page: [http://www.vision.caltech.edu/visipedia/CUB-200.html](http://www.vision.caltech.edu/visipedia/CUB-200.html)

---

#### **3. Image and Annotation Files**

The training images are located in the **Train.zip** folder, with their corresponding annotations in the **train.txt** file. The format for the annotation file is as follows:

`image's name{space}class label`

**Example from train.txt:**
```
Black_footed_Albatross_0019_416160254.jpg 0
Black_footed_Albatross_0005_2755588934.jpg 0
Laysan_Albatross_0014_174432783.jpg 1
Sooty_Albatross_0005_340127050.jpg 2
```

The testing images and their annotations can be found in the **Test.zip** folder and **test.txt** file, respectively, following the same format.

---

#### **4. Evaluation Metrics**

Your results must be reported using the following two metrics:

*   **Top-1 Accuracy:** This evaluates the overall classification performance of your models.
    *   Formula: `Top-1 accuracy = (1/N) * Σ_k=1^N 1{argmax(y) == groundtruth}`
    *   Where `N` is the total number of testing images and `y` is the output probabilities for the 200 classes.

*   **Average Accuracy Per Class:** This assesses the performance for each individual class.
    *   Formula: `Ave = (1/C) * Σ_i=1^C T_i`
    *   Where `T_i` is the average accuracy for all test images of class `C_i`, and `C` is the total number of classes.

You may include other evaluation metrics, but you must provide a proper justification for their use.

---

#### **5. Required Submission**

You are required to submit the following items to the Canvas assignment link:

**Report:**
*   A PDF document with two main sections: **Methodology** and **Result and Discussion**.
*   The report should address potential overfitting and how your model design aims to minimize it.
*   Detail your models' architecture, loss function, hyperparameters, and other relevant information.
*   Discuss and compare the performance of your models, justifying which one produced the best results.
*   The report must be limited to **4 pages** and include your **name** and **ID**.

**Python Program Source Code:**
*   A single zip file containing all your code.
*   The code should be well-commented to demonstrate your understanding.

**Video Presentation:**
Your presentation should:
*   Explain the key concepts you are using and your model design. Slides are recommended.
*   Present and explain your code.
*   Demonstrate the training process and the results generated.
*   Present your AI model web application via Hugging Face.
*   The presentation should be a maximum of **10 minutes**.
*   Include the **YouTube link** for your video and the **Hugging Face link** in your report.