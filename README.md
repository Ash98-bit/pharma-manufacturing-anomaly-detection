# pharma-manufacturing-anomaly-detection

**Authors**: Aditi Shilke  
**Course**: Master's Thesis  
**Title**: Harnessing Data Science in Pharma: Improving Process Verification with Machine Learning  
**Supervised by**: Dr. Peter Roßbach & M.Sc Daniele Gadler  
**Core Focus**: Anomaly detection using Unsupervised Machine Learning, data science and business analytics for the MS&T team at STADA AG  

## Overview: 

Welcome to the next chapter of my journey at the intersection of data science, business analytics, and machine learning for business impact! This project captures the highlights from my Master’s thesis, conducted in collaboration with STADA. Because the work involved proprietary data, the complete code cannot be shared here. Instead, this repository showcases some of the core methods, workflows, and insights that shaped the project.

The thesis focused on developing and engineering clustering and anomaly detection models to identify component batches that showed statistically deviant Input Material Attributes. Beyond simply flagging anomalies, the models also aimed to highlight which of these attributes might be driving these deviations. At a higher level, the project explored correlations between the input material attributes and the output quality attributes, with the ambition to contribute toward real-time quality assurance applications. Imagine automated alerts for QA teams when unusual production patterns emerge — that’s the kind of value the research sought to unlock.

By equipping subject-matter experts with novel, machine learning–driven insights, the thesis aligned with STADA’s broader Pharma 4.0 vision: enabling data-driven process optimization and supporting the company’s mission of caring for people’s health.

## Highlights:

This repository captures a few key components and analyses from the thesis, showcasing the workflow and insights without exposing any proprietary data.

**Data ETL** – This chapter outlines the first phase of data preparation, focusing on refining Ongoing Process Verification (OPV) data into a structured, interpretable dataset ready for anomaly detection via machine learning. Domain-informed data curation was central to this step, as the performance of ML models is strongly influenced by the quality and relevance of the input data. The ETL process finalized a clean, interpretable dataset tailored for unsupervised anomaly detection, carefully preserving statistically and operationally meaningful input material attributes. An example of custom logic developed to protect data integrity from rounding issues inherent to binary floating-point representation is also illustrated in Fig 5.4.

**Principal Component Analysis (PCA)** – This chapter focuses on transforming the dataset to maximize the effectiveness of unsupervised anomaly detection. The dual objectives were to apply PCA for dimensionality reduction while preserving key variance signals, and to standard scale the data both before and after PCA. Results were visualized through a scree plot and a component loadings heatmap, ensuring that dimensionality reduction remained interpretable, aligned with OPV process understanding, and preserved important patterns for downstream analysis.

**One-Class SVM (OCSVM) Trial** – This chapter explores the application of the One-Class Support Vector Machine (OCSVM) model for identifying anomalous batches within the feature space. To define what constitutes a “normal” profile for the model, a frequency distribution analysis of unique profiles was conducted to determine the gamma (γ) parameter, as illustrated in Fig 12.1. Further observations were made to assess whether OCSVM was suitable for the dataset at hand, ensuring the approach was grounded in both statistical rigor and domain relevance.

**Isolation Forest** – This chapter details how the Isolation Forest pipeline was adapted to the specific characteristics of the dataset to surface statistically meaningful outliers. The first step in tuning the hyperparameters of the model involved evaluating its decision function, while the contamination parameter was explored with attention to the OPV context. To support model assessment, the detected outliers were visualized in PCA 3D plots (Fig 11.4 and 11.5), providing a deeper understanding of the model’s suitability for potential large-scale deployment.

## Outcomes:

Through a structured modeling framework, unsupervised machine learning models were developed, evaluated, and validated. Each approach successfully flagged outliers within the dataset, directly delivering the insights requested by stakeholders. These anomalies were not only statistically meaningful but also visually separable in both PCA projections and the raw feature space, reinforcing their structural distinctiveness.

Post-hoc validation with historical data revealed that a significant portion of the flagged outliers overlapped with batches that had previously failed regulatory checks. This overlap provided a strong operational justification for the pipeline, highlighting its value in real-world quality assurance contexts.

The project was awarded a grade of 1.3 (on a scale of 1 to 5, 1 being the best) :D
