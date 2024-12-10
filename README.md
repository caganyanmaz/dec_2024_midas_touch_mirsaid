# Vela Partners - Investor Scoring and Feature Engineering
### Research Internship Work
Author: Mirsaid Abdullaev, Oxford University

Supervisor: Yigit Ihlamur, Vela Partners

### Abstract

The venture capital landscape is often influenced by prominent early-stage investors who are seen as predictors of startup success. However, focusing solely on well-known names like Elad Gil or Andreessen Horowitz (a16z) limits our predictive capabilities due to data sparsity and overlooks the nuanced features that contribute to an investor's success. This project aims to abstract investor names by creating "investor personas" based on quantifiable features. By doing so, we can enhance our ability to predict startup success and understand how different investor types contribute to it.

### Goals

- **Primary goal**: Define investor personas using quantitative features extracted from investment data, thereby reducing data sparsity and improving prediction models for startup success.
- **Secondary goal**: Analyze the impact of co-investments between different investor personas on the probability of startup success.

### Expected Outputs

**(Primary objective): Calculate the random probabilities**

Understand what random picking probability is. Calculate the average outlier rate, interim outlier rate and recent graduation rate of the whole dataset. In this way, we’ll understand which personas outperform or underperform the index. 

**(Primary objective): Investor Personas**

Determine what personas you’ll develop. Develop investor personas such as L1 Specialist, L3 Angel, or L5 Generalist, and calculate their corresponding success rates against the index. 
Each persona should be characterized based on features derived from the data. If you want to keep it simple, you can initially focus on developing five different levels: L1 - L5. If you want to add more sophistication, you can add specialization/generalization of the investor via categories AND if the investor is a person or VC firm (person vs company) AND many more personas like this. 

**(Secondary objective): Multi-Investor Persona Analysis**

Analyze combinations of investor personas to determine how co-investments affect the probability of startup success. Initially, focus on two-pair combinations, and increase it to three-pair combinations. 

### Plan/Methodology

1. I will make a reusable, scalable data processing project with Python scripts utilising the Pandas, Numpy, OpenAI and MatPlotLib libraries to streamline the data engineering.

2. My initial aim is to create a dataset preparation script which takes in datasets of the form given to me originally, and cleans the dataset to make it easier to work with.

3. I will follow this with a feature engineering script where I calculate and save new datasets which have many more useful derived features from the raw data provided, for use in later parts of my project.

4. Using this feature engineering script, I will aim to categorise all investors by their success rates in investing into startups, their investing experience, their breadth of investing (i.e. how many sectors/industries they tend to invest across). I want to create two separate bucket classes initially, one which buckets investors by their success rates and the other which will bucket investors by their breadth of investing.

5. The two independent bucketing strategies will then be combined to create pair-buckets, which will have certain thresholds and allow us to classify new investors into these buckets given data on them. This will also give us their expected success rate and recent graduation rate based on this. This allows us to maximise our own success rates by choosing the "follow the best picker" strategy.

6. After this, I plan on using knowledge graph methods to investigate coinvestor relationships. The first aim with this methodology will be to define an efficient way to design the knowledge graph by nodes and edges in order to capture the most important information from my now-extensive list of features. Something that I need to think about is how to weight information based on factors such as investor experience, like lowering the impact of high-success-rate/short-investor-experience compared to high-success-rate/long-investor-experience investors for example.


### Progress update on the plan/methodology action points

1) I have developed a Python-based data processing framework using the Pandas, Numpy, Matplotlib libraries.S This framework is modular and reusable, allowing me to clean, process, and derive features from the provided datasets efficiently. The scripts have been developed to be able to be run sequentially, *"dataset_prep.py -> feature_eng.py -> knowledge_graph.py -> ... -> functions.py"*, where I hope the functions.py should have the final use-case functions implemented to be able to pull out data on investors provided some input data.

2) A complete dataset preparation script has been implemented. This script cleans the raw datasets, removes inconsistencies such as missing values and duplicate indices, and standardizes formats across all tables. It outputs cleaned datasets ready for downstream processing. We start with a coinvestor relationships table, a long term performance, and a short term performance dataset, and my script cleans all the data, and creates a new *coinvestor_clean.csv* table, an *investors.csv* table, a *"startups.csv"* table for mapping startup uuid's to names, and new *long_term_clean.csv* and *short_term_clean.csv* tables. 

3) I have created a feature engineering script that calculates various investor-specific features such as annualized investments, success rates (250M+, 100M+, 25M+ outlier rates), recent graduation rates, breadth of investing (specialist to universalist categories), and outlier scores (my own weighted sum of the 250M+, 100M+, and 25M+ success rates in the respective weighting 0.85, 0.10, 0.05). These features are saved into the *investors.csv* dataset for analysis and modelling in tandem with the *coinvestor_clean.csv*.

4) Investors have successfully been categorized into two distinct bucket classifications:
    - Outlier Success Buckets (*L1–L5*): Investors are grouped by success rates in their investments.
    - Focus Classification (*Specialist–Universalist*): Investors are grouped by the breadth of industries/sectors they invest in. 

    Both classifications are derived from the engineered features and allow for detailed segmentation and analysis.

5) I have created combined buckets (e.g., *"L1 Specialist"*) by merging the two independent bucketing strategies. The combined buckets have been analyzed for counts, mean, median, and standard deviation of outlier scores and recent graduation rates, stored in a new *bucket_stats.csv* table. This enables predicting investor performance based on these pair classifications.

6) The foundation for a knowledge graph to analyze coinvestor relationships has been laid. Investors are represented as nodes with features like annualized investments, outlier scores, focus buckets, outlier buckets and investor experience. Edges represent coinvestor relationships with attributes such as shared investment counts, combined outlier scores, average investor experience, focus bucket pairs, and outlier bucket pairs. This design supports advanced modeling and analysis of interrelationships between investors.

