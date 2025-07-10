# StarX - Machine Learning Task Repository

This repository contains solutions to a machine learning task provided by a company. It's organized into three distinct tasks, demonstrating data portrayal, intuition building, and script-based solution implementation using Python.

---

## Project Structure

You'll find the repository structured like this:
```
.
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
    ```
2.  **Recreate the environment:** Navigate into the `requirements/` folder and follow the specific instructions there. This will likely involve installing dependencies. For example, if there's a `requirements.txt` file:
    ```bash
    pip install -r requirements/requirements.txt
    ```

---

## Usage

Once your environment is set up:

* To explore **Task 1** and **Task 2**, open their respective Python notebooks (`.ipynb` files) using Jupyter Notebook or JupyterLab.
* To run **Task 3**, simply execute the Python script from your terminal:
    ```bash
    python task_3/task_3_script.py
