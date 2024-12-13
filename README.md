# Vela Partners - Investor Scoring and Feature Engineering

## Research Internship Work

### Author: Mirsaid Abdullaev, Oxford University

### Supervisor: Yigit Ihlamur, Vela Partners

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

1. **Create a scalable data processing project**: Build Python scripts utilizing libraries such as Pandas, NumPy, OpenAI, and Matplotlib to clean and process the data efficiently.

2. **Data preparation**: Develop a dataset preparation script to clean raw datasets, resolve inconsistencies, and standardize formats for downstream analysis.

3. **Feature engineering**: Create a feature engineering script to calculate investor-specific features like annualized investments, success rates, and breadth of investing. Store the results in a consolidated dataset.

4. **Investor categorization**: Categorize investors into two distinct buckets—outlier success buckets (L1-L5) and focus classification (Specialist–Universalist)—and analyze their distributions.

5. **Combined bucketing**: Combine the two classifications to create "pair-buckets" (e.g., L1 Specialist) and compute aggregated statistics such as success rates and recent graduation rates for these groups.

6. **Knowledge graph construction**: Design a knowledge graph to represent investors as nodes with attributes and coinvestor relationships as weighted edges. Identify key attributes to include and optimize weighting schemes for relationships.

7. **Graph analysis**: Apply centrality measures such as degree and eigenvector centrality to identify influential investors and partnerships.

8. **Investor ranking**: Implement a custom page-rank-inspired algorithm, **InvestorRank**, to determine the most influential investors in the graph using scaled shared outlier rates as edge weights.

9. **Community detection**: Use the Louvain algorithm to detect communities in the graph and analyze their aggregated statistics, focusing on the most impactful groups.

10. **Final deliverables**: Save the results of community analysis and investor ranking into structured datasets, including a community composition summary and individual community files for detailed analysis.

---

### Development

1. **Created a scalable data processing project**: Built modular Python scripts for data cleaning, feature engineering, and graph construction. These scripts can be run sequentially: `dataset_prep.py -> feature_eng.py -> knowledge_graph.py` or just run the helper script `run_analysis.py` script on its own, which will run these scripts in order.

2. **Data preparation**: Cleaned raw datasets to remove inconsistencies and standardize formats. Generated cleaned datasets: `coinvestor_clean.csv`, `all_investors.csv`, `startups.csv`, `long_term_clean.csv`, and `short_term_clean.csv`.

3. **Feature engineering**: Calculated investor-specific features, including annualized investments, success rates (250M+, 100M+, 25M+), breadth of investing, and outlier scores. Stored these features in `all_investors.csv`.

4. **Investor categorization**: Grouped investors into:
    - **Outlier Success Buckets**: Categorized by success rates (L1-L5).
    - **Focus Classification**: Based on breadth of investing (Specialist–Universalist).  
   These classifications allow for nuanced segmentation.

5. **Combined bucketing**: Merged the two classifications into pair-buckets (e.g., L1 Specialist). Computed and analyzed statistics for these buckets, saved in `bucket_stats.csv`.

6. **Knowledge graph construction**: Built a knowledge graph with:
    - **Nodes**: Investors with attributes like annualized investments and outlier scores.
    - **Edges**: Representing coinvestor relationships with attributes like shared outlier scores and total coinvestments.

7. **Graph analysis**: Applied degree centrality to identify well-connected investors and eigenvector centrality to highlight those connected to influential peers. These measures provided insights into key investors and partnerships.

8. **Investor ranking**: Developed **InvestorRank**, a custom ranking algorithm inspired by PageRank. This algorithm ranks investors by influence, using scaled shared outlier rates between coinvestors as edge weights.

9. **Community detection**: Used the Louvain algorithm to detect and analyze communities. Summarized community statistics (e.g., average outlier rate, investor experience) in `community_composition.csv`. Saved individual community details into separate files (`community_{rank}.csv`).

10. **Final deliverables**: Provided structured datasets summarizing the graph, including adjacency matrices, investor rankings, and community compositions.

### Results

Below are visual representations of the processed datasets and analysis results. Each image showcases key data points or insights extracted during the project.

#### 1. Adjacency Matrix
![Adjacency Matrix](results/Adjacency%20Matrix.png)  
*Sample rows from the adjacency matrix, showing total coinvestments, shared outlier scores, and other edge attributes between investors.*

#### 2. Bucket Stats
![Bucket Stats](results/Bucket%20Stats.png)  
*Summary statistics for investor buckets (L1-L5, Specialist–Universalist), including counts and success rates.*

#### 3. Centrality Analysis
![Centrality Analysis](results/Centrality%20Analysis.png)  
*Output from centrality analysis, highlighting key investors based on degree and eigenvector centrality measures.*

#### 4. Community 1 - Sample
![Community 1 - Sample](results/Community%201%20-%20Sample.png)  
*Example of a community detected using the Louvain algorithm, showing nodes (investors) and their attributes.*

#### 5. Community Composition
![Community Composition](results/Community%20Composition.png)  
*Aggregated statistics for each detected community, including average outlier rate, experience, and node counts.*

#### 6. Knowledge Graph Investors
![Knowledge Graph Investors](results/Knowledge%20Graph%20Investors.png)  
*Investor data used as nodes in the knowledge graph, including annualized investments, outlier scores, and classifications.*



---
### Evaluation and Future Work

This project achieved its primary goals of constructing a robust knowledge graph of investors and analyzing their influence within venture capital networks. However, several extensions and enhancements could be made to deepen the insights and broaden the scope of the research. 

#### **Key Achievements**
1. **Comprehensive Data Pipeline**: A well-structured data pipeline was implemented, enabling efficient data preparation, feature engineering, and graph construction. 
2. **Custom Ranking Algorithm**: The **InvestorRank** algorithm successfully quantified investor influence by integrating shared success rates and coinvestment data.
3. **Community Analysis**: Louvain clustering provided significant insights into the relationships and structures within the network, identifying key communities and their characteristics.

#### **Potential Extensions**
1. **Visual Exploration of the Graph**: While the project focused on constructing and analyzing the graph quantitatively, an intuitive visual representation would allow for identifying patterns and anomalies directly. Tools like Gephi or Plotly could be used to map the graph, highlight influential nodes, and visually depict community structures. Visual exploration could also facilitate storytelling around investor behaviors and their contributions to startup success.

2. **Feature Correlations**: The project primarily segmented investors and analyzed their influence. A natural extension would involve studying correlations between investor features (e.g., focus classification and outlier score, or annualized investments and success rates). This analysis could reveal relationships that are not immediately apparent and could inform predictive models or strategies for matching investors with startups.

3. **Predictive Modeling**: Given the rich dataset and constructed features, building machine learning models to predict the success probabilities of startups based on their investors’ profiles and communities could be a valuable addition. This could extend the practical applications of the research, enabling startups to identify optimal investors and vice versa.

4. **Temporal Analysis**: The project largely considers static snapshots of investor performance and relationships. Incorporating temporal elements, such as how investor influence or community structures evolve over time, could provide insights into emerging trends and changing dynamics in venture capital.

5. **Sector-Specific Analysis**: While the current analysis is generalized across all industries, creating sector-specific knowledge graphs could allow for a more targeted understanding of investor success and behavior in areas like fintech, healthcare, or artificial intelligence.

#### **Time Constraints and Challenges**
This project was completed independently within just four days. While the primary objectives were met, the time constraints limited further exploration into the aforementioned extensions. For instance, a detailed exploration of community interactions or fine-tuning the **InvestorRank** algorithm for specific industry contexts could have added depth to the analysis.

#### **Reflection**
This research internship with Vela Partners was an incredible learning opportunity. It gave me the chance to explore the intersection of data engineering, graph theory, and venture capital in a meaningful way. Collaborating with such a talented and insightful mentor as Yigit Ihlamur has been an inspiring experience. His guidance not only helped shape the direction of this project but also gave me a deeper appreciation for the analytical and strategic thinking required in this field. As a first-year undergraduate, this opportunity has greatly enriched my technical skills and broadened my understanding of real-world applications in venture capital.


### **Author**: Mirsaid Abdullaev, Oxford University

### **Supervisor**: Yigit Ihlamur, Vela Partners

### Contact Details

- **LinkedIn**: [www.linkedin.com/in/mirsaid-abdullaev-6a4ab5242/](https://www.linkedin.com/in/mirsaid-abdullaev-6a4ab5242/)
- **Email**: [mirsaid.abdullaev@cs.ox.ac.uk](mailto:mirsaid.abdullaev@cs.ox.ac.uk)
- **Phone**: +447 498 663301
