# User-Based Collaborative Filtering (UserKNN) from Scratch

This repository contains a pure Python implementation of the **User-based Collaborative Filtering** algorithm. The project is designed as a foundational component for research in **Explainable AI (xAI)** and **Recommender Systems** using the MovieLens 100k dataset.

## Project Overview
The system predicts user ratings for unobserved items by identifying "neighbors" with similar tastes. Unlike library-dependent implementations, this code utilizes **Vectorized Operations** with `numpy` and `pandas` to ensure high performance while maintaining mathematical transparency.

* **Dataset:** MovieLens 100k (943 users, 1682 movies).
* **Key Techniques:** Cosine Similarity, K-Nearest Neighbors (KNN), Mean Centering, Matrix Factorization Pre-processing.
* **Performance Metric:** Root Mean Squared Error (RMSE).

## Mathematical Background

[Image of User-based Collaborative Filtering mechanism]

### 1. Cosine Similarity
The similarity between two users $u$ and $v$ is calculated as the cosine of the angle between their rating vectors:

$$sim(u, v) = \frac{\mathbf{r}_u \cdot \mathbf{r}_v}{\|\mathbf{r}_u\| \|\mathbf{r}_v\|} = \frac{\sum r_{ui} r_{vi}}{\sqrt{\sum r_{ui}^2} \sqrt{\sum r_{vi}^2}}$$

### 2. Normalization (Mean Centering)
To mitigate "rating bias" (where some users are naturally more generous than others), we normalize the data by subtracting the user's mean rating $\bar{r}_u$:

$$h_{ui} = r_{ui} - \bar{r}_u$$

### 3. Rating Prediction
The predicted rating $\hat{r}_{u,i}$ is a weighted average of the ratings from the top $k$ neighbors ($N_k$):

$$\hat{r}_{u,i} = \bar{r}_u + \frac{\sum_{v \in N_k(u)} sim(u, v) \cdot (r_{vi} - \bar{r}_v)}{\sum_{v \in N_k(u)} |sim(u, v)| + \epsilon}$$

## Features
* **Optimized Similarity Computation:** Uses matrix multiplication ($R^T \cdot R$) to compute all-pair similarities efficiently.
* **Sparse Data Handling:** Custom `train_test_split_matrix` function designed to handle `NaN` values in sparse user-item matrices.
* **Top-N Recommendations:** Generates a ranked list of movies for any specific user based on predicted scores.

## Results
The model's accuracy is evaluated using **RMSE** on a 20% hold-out test set.

| Parameter | Setting |
| :--- | :--- |
| **K-Neighbors** | 100 |
| **Test Size** | 20% |
| **Similarity Metric** | Adjusted Cosine |
| **Rating Bounds** | Clipped [1.0, 5.0] |

## Explainable AI (xAI) Integration
This code serves as a baseline for xAI research. By maintaining the movie genre metadata (Action, Sci-Fi, etc.), the system allows for:
1.  **Neighbor-based Explanations:** "Because users similar to you liked this."
2.  **Content-based Reasoning:** Analyzing if the recommended movies align with the user's historical genre preferences via Association Rules.

---
**Author:** Khoa (Student @ HCMUS)
