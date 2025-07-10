# AI ML Developer Test Anosh

This repository contains solutions to a machine learning task provided by the company. It's organized into three distinct tasks, task_4 is contained in another repository,the link to which will be attached along with this.
As the first and second tasks involve model training and ouput visualization at every step, instead of doing it in a normal python file, I've used a Jupyter Notebook to do the data_analysis and representation on task1_symptom_analysis, and task2_image_detection. This will be intuitive for the person inspecting the code as well, as it shows my thinking process in every step of solving the process, which I have done with Markdown texts and comments throughout the code, so that anyone can understand it.
A README.md file is there for every task, and this README file is just to give some insights to the way I have structured and solved the questions

Task 4: 
Experience the ChestX-Pneumonia-CNN live application here:

[ChestX-Pneumonia Live Demo](https://chestx-pneumonia.streamlit.app/)

GitHub repo link:

https://github.com/anoshandrews/ChestX-Pneumonia-CNN

---

## Folder Structure

You'll find the repository structured like this:
```
AI_ML_Developer_Test_Anosh
├── README.md
├── requirements.txt
├── task_1_symptom_analysis
│   ├── confusion_matrix_scaled.png
│   ├── README.md
│   ├── symptom_analysis.ipynb
│   └── Symptom2Disease.csv
├── task_2_image_detection
│   ├── image_detection.ipynb
│   └── README.md
└── task_3_rag_pipeline
    ├── 3_questions_output.png
    ├── docs
    │   ├── diabetes.txt
    │   └── hypertension.txt
    ├── rag_pipeline.py
    └── README.md
```


* `task_1/`: Holds the solution for the first task, implemented in a Python **notebook** for clear data portrayal and intuition.
* `task_2/`: Contains the solution for the second task, also in a Python **notebook**, focusing on deeper analysis and insights.
* `task_3/`: Houses the third task's solution, provided as a standalone Python **script**.
* `requirements/`: This crucial folder contains **precise instructions and/or files** needed to set up the project's environment.

---

## Tasks Overview

### Task 1: Data Portrayal & Intuition (Python Notebook)

This initial task focuses on exploring the dataset, visualizing key features, and building foundational intuitions about the data. The notebook format allows for interactive analysis and clear presentation of findings.

### Task 2: Data Analysis & Insights (Python Notebook)

Building on the first task, this section dives deeper into data analysis. You might find more complex visualizations, statistical analyses, or preliminary model explorations here. The notebook ensures a clear and reproducible workflow for conveying insights.

### Task 3: Solution Implementation (Python Script)

The third task involves implementing a specific solution using a Python script. This demonstrates the ability to develop a self-contained and executable program for a defined machine learning problem.

---

## Setup & Installation

To get started and run the tasks, you'll need to set up your environment by following the instructions within the **`requirements/`** folder. Here’s a typical workflow:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/anoshandrews/AI_ML_Developer_Test_Anosh.git](https://github.com/anoshandrews/AI_ML_Developer_Test_Anosh.git)
    cd AI_ML_Developer_Test_Anosh
    
2.  **Recreate the environment:** Navigate into the folder and follow the specific instructions there. This will likely involve installing dependencies. For example, if there's a `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

Once your environment is set up:

* To explore **Task 1** and **Task 2**, open their respective Python notebooks (`.ipynb` files) using Jupyter Notebook or JupyterLab.
* To run **Task 3**, simply execute the Python script from your terminal:
    ```bash
    python task_3/task_3_script.py
