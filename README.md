# Predicting Startup Success Through Investor Analysis: A Graph-Based Approach

## Abstract

This research presents a novel approach to predicting startup success by analyzing investor behavior and relationships using graph-based methods. We focus on the coinvestor dataset to examine how different investors' historical performance and network characteristics can predict future startup success. By splitting investments before and after 2021 into training and test sets, we develop several prediction models combining past success rates, linear regression, and neural networks. The models analyze three types of data: basic investor features, coinvestor pairs, and graph-enhanced investor features including PageRank and centrality metrics. Our results show that simpler models often outperform more complex ones, with the basic method achieving up to 28.4x outlier detection performance than random selection when picking the top 30 investments.

## Introduction

Predicting startup success is a critical challenge in venture capital, with significant implications for investors, entrepreneurs, and the innovation ecosystem. Traditional approaches often rely heavily on qualitative assessments and brand recognition of prominent investors. This research takes a more quantitative approach, leveraging historical investors data and network analysis to develop predictive models.

Our research makes several key contributions to the field. We develop a comprehensive methodology to predict startup success using investor historical performance and other features, creating and analyzing different model types across various data representations. We integrate graph-based features to capture network effects in the startup ecosystem and evaluate different combination methods for aggregating multiple investors' signals. Importantly, we demonstrate that simpler models can outperform more complex approaches in this domain, providing practical insights for implementation.

## Related Work

While our research builds upon previous studies in startup success prediction, it differentiates itself by focusing specifically on investor behavior and network effects. The "Midas Touch" paper explored similar themes using graph-based methods to analyze startup investors, but our approach differs in its focus on prediction rather than classification. The field has seen significant development in recent years, with various approaches attempting to quantify and predict startup success through different lenses.

## Dataset Description

For our research we only use the Co-investor Relationships dataset. This dataset contains detailed information about startups and their associated investors, including columns of startup UUID, UUIDs and names of the investors of the startup, category lists of the startup, and total funding it had in 2024, which is used to assess the success of the startup in our research.

## Data Preparation

Our data preparation pipeline processes raw investment data through several cleaning and structuring steps. We begin by extracting and cleaning the core dataset containing startup investments and their investors. The raw data includes organization UUIDs, investor UUIDs, investor names, founding years, funding amounts and other information about the startup. The preprocessing begins with creating a mapping table between startup UUIDs and names for reference purposes. We then remove startups with missing category information, which affects 463 out of over 37,000 startups, ensuring complete categorical analysis for all remaining entries. Investor information is extracted into a separate table that maps UUIDs to names, maintaining consistency across the dataset. The data is then split into pre-2021 (input) and post-2021 (output) sets based on founding year.

The resulting cleaned dataset contains 36,932 startups and 33,726 investors, with complete categorical and temporal information for all entries. The data is separated into training (pre-2021) and testing (post-2021) sets, providing a robust foundation for our analysis.

## Feature Engineering

We engineer a comprehensive set of features across three main categories: basic investor features, coinvestor pair features, and graph-enhanced features. Each category captures different aspects of investor behavior and relationships within the startup ecosystem.

### Basic Investor Features

The fundamental investor features begin with investment activity metrics. We calculate the annual investment count by normalizing total investments by years active between 2013-2024, alongside maintaining the raw count of all investments made. The earliest investment year is recorded to calculate investor experience.

- Investment Counts: investment_count stores the number of startups invested by the investment, annualised_investment_count stores the investment count divided by the years to get an annualised quantity.
- Investment Experience: earliest_investment_experience stores the earliest investment made by the investor in the dataset, experience stores the years passed since the first investment of the investor (since we train based on 2021 data, the end year is 2021).
- Diversity of Portfolio: There are several metrics to measure how specialized the investor is. specific_category_count keeps track of the total number of different categories of the startups invested by the investor. broad_category_count does the same for category groups. proportional_specific_diversity and proportional_broad_diversity are the respective category counts of the investor divided by the total number of investments made. This is to detect specialized investors that invest in a higher number of startups. specific_diversity and broad_diversity represent the normalized values of square of respective category counts divided by the number of total investments made. This is made to have a higher weight on total category count than the proportional values while determining the diversity of the portfolio.
- Success Rates: We created multiple metrics to determine the success of the investors. 25m_rate represents the proportion of startups invested by the investor that exceeded 25 million dollars in funding and that was founded after 2019, 100m_rate represents the proportion of startups invested by the investor that exceeded 100 million dollars in funding and that was founded after 2017, 250m_rate represents the proportion of startups invested by the investor that exceeded 250 million dollars in funding and that was founded after 2013.

The weighted_success_rate gives the weighted sum of these three rates so that we can obtain a single value to determine the company's total success from its long term and short term success. The weights are 0.625 for 250m_rate, 0.25 for 100m_rate, and 0.125 for 25m_rate.
- Buckets: Finally, in order to categorize the investors, we assign them two seperate bucket attributes depending on their success rate and diversity rate. The startups are sorted by their weighted_success_rate and broad_diversity respectively, and then assigned 5 buckets depending on their quintile position. For success_bucket, the buckets are named "L1, L2, L3, L4, L5" from the least successful to the most. For diversity_bucket, the buckets are named "Specialized, Focused, Balanced, Diverse,Universalist" from least diverse to most. Although the bucket metrics are not used for the models directly, they're used for the graph algorithms and to extract graph features.

### Partner Investor Features

For pairs of investors that have co-invested together, we compute several relationship-based metrics including shared investments, joint success rates, and combined diversity metrics. The relationship features capture both the quantity and quality of investment partnerships.


- Co-invested metrics: investment_count, 25m_rate, 100m_rate, 250m_rate, weighted_success_rate are very similar to the features with the same name for investors. But for the co-investor features, the set of investments is the startups in which both of the co-investors invested.
- Scaled success: Since the co-investor pairs make a sparse dataset, with many co-invesot pairs investing in only one or two similar startups without a meaningful partnership, we employ the scaled_success_rate metric, which is a version of weighted_success_rate that penalizes the small number of investments.
- Average metrics: average_experience, average_success_rate, average_specific_diversity, average_broad_diversity and average_annualised_investment_count metrics are the averages of the respective metrics of both co-investors.
- Combined bucket pairs: success_bucket_pair, diversity_bucket_pair are the concatanation of respective buckets of both co-investors.


### Investor Graph Features

InvestorRank_score,degree_centrality,eigenvector_centrality,Our graph-based features derive from a knowledge graph where investors are nodes and co-investments are edges. We use several algorithms to extract information from the structure of the knowledge graph.

Centrality metrics: Centrality metrics capture different aspects of investor influence through degree_centrality and eigenvector_centrality. degree_centrality measures the number of co-investment relationships of each investor, calculating their connectivity with other investors. eigenvector_centrality is calculated through the degree_centrality of co-investors of the investors, measuring the quality of the connections instead of just the quantity like degree_centrality.

PageRank algorithm (Investor Rank): Evaluating the influence of investors in a similar manner to Google's Page Rank algorithm.

Communities: We find community of investors using Louvain algorithm on our knowledge graph, then append total size of the community, average statistics of investors and bucket counts (number of investors in the community with a specific success bucket or a diversity bucket) of the community to every investor in the community, so that our models can use the community statistics on decision making. 


The graph algorithms are taken from the Midas Paper, so you can check that to get additional information on exact implementations, and how the knowledge graph is constructed.

## Methodology

### Models

Our approach to model development focuses on creating practical, interpretable predictions. We implement three distinct model types for each data representation: a basic model that predicts future success rate equal to past success rate, linear regression that learns feature relationships, and neural networks that capture potential non-linear patterns.

- Basic Model: For all three datatypes, it predicts the success of a investor / investor pair equal to its past weighted_success_rate
- Linear Model: Using every quantifiable feature in the given datatypes, it creates a linear regression model to predict investor success
- Neural Model: Similar to Linear Model, we only use a dense Neural Network with three hidden layers and appropriate number of neurons in each layer. 

We don't need any training for basic model. For linear and neural models, we use the data of invetors before 2021 to predict weighted_success_rate of data after 2021, and train using those metrics.

To combine individual investor success predictions into final startup success predictions, we developed several aggregation methods. These include taking the maximum score among all investors, computing various averages and medians, and implementing a least squares optimization approach. Each method provides different trade-offs between stability and sensitivity to strong individual signals.

- Best: Returns the highest prediction from every investor / investor pair of a startup
- Mean: Returns the mean prediction from every investor / investor pair of a startup
- Median: Returns the median prediction from every investor / investor pair of a startup
- Top Mean: Returns the mean prediction from 5 investor / investor pairs with highest predictions of a startup
- Top Mean: Returns the median prediction from 5 investor / investor pairs with highest predictions of a startup
- LSQ: Given the predicted success rates and investment counts of investors / investor pairs, and all of the investments, it constructs a system of linear equations where the variables are success status of investments (if we had known it before, they would've been 0 if fail, 1 if success) and the result are total expectd succeess counts of investors / investor pairs (success rate * investment count). Then it finds the solution with least square error and assigns the results of each investment as the predicted score.


## Results

### Investors Dataset

#### Simple Model
| Method | Prediction (25M/100M/250M) | Recall (25M/100M/250M) | Counts (25M/100M/250M) | Improvement (25M/100M/250M) |
|--------|---------------------------|------------------------|----------------------|---------------------------|
| Best | 33.3%/10.0%/3.3% | 5.6%/7.9%/16.7% | 10/3/1 | 9.5x/13.5x/28.4x |
| Mean | 26.7%/13.3%/6.7% | 4.5%/10.5%/33.3% | 8/4/2 | 7.6x/17.9x/56.8x |
| Median | 33.3%/20.0%/6.7% | 5.6%/15.8%/33.3% | 10/6/2 | 9.5x/26.9x/56.8x |
| Mean Top | 30.0%/13.3%/6.7% | 5.0%/10.5%/33.3% | 9/4/2 | 8.6x/17.9x/56.8x |
| Median Top | 26.7%/13.3%/6.7% | 4.5%/10.5%/33.3% | 8/4/2 | 7.6x/17.9x/56.8x |
| LSQ | 13.3%/6.7%/3.3% | 2.2%/5.3%/16.7% | 4/2/1 | 3.8x/9.0x/28.4x |

#### Linear Model
| Method | Prediction (25M/100M/250M) | Recall (25M/100M/250M) | Counts (25M/100M/250M) | Improvement (25M/100M/250M) |
|--------|---------------------------|------------------------|----------------------|---------------------------|
| Best | 33.3%/13.3%/3.3% | 5.6%/10.5%/16.7% | 10/4/1 | 9.5x/17.9x/28.4x |
| Mean | 30.0%/13.3%/6.7% | 5.0%/10.5%/33.3% | 9/4/2 | 8.6x/17.9x/56.8x |
| Median | 36.7%/20.0%/6.7% | 6.1%/15.8%/33.3% | 11/6/2 | 10.5x/26.9x/56.8x |
| Mean Top | 30.0%/13.3%/6.7% | 5.0%/10.5%/33.3% | 9/4/2 | 8.6x/17.9x/56.8x |
| Median Top | 23.3%/10.0%/6.7% | 3.9%/7.9%/33.3% | 7/3/2 | 6.7x/13.5x/56.8x |
| LSQ | 16.7%/6.7%/3.3% | 2.8%/5.3%/16.7% | 5/2/1 | 4.8x/9.0x/28.4x |

#### Neural Model
| Method | Prediction (25M/100M/250M) | Recall (25M/100M/250M) | Counts (25M/100M/250M) | Improvement (25M/100M/250M) |
|--------|---------------------------|------------------------|----------------------|---------------------------|
| Best | 0.0%/0.0%/0.0% | 0.0%/0.0%/0.0% | 0/0/0 | 0.0x/0.0x/0.0x |
| Mean | 0.0%/0.0%/0.0% | 0.0%/0.0%/0.0% | 0/0/0 | 0.0x/0.0x/0.0x |
| Median | 0.0%/0.0%/0.0% | 0.0%/0.0%/0.0% | 0/0/0 | 0.0x/0.0x/0.0x |
| Mean Top | 0.0%/0.0%/0.0% | 0.0%/0.0%/0.0% | 0/0/0 | 0.0x/0.0x/0.0x |
| Median Top | 0.0%/0.0%/0.0% | 0.0%/0.0%/0.0% | 0/0/0 | 0.0x/0.0x/0.0x |
| LSQ | 3.3%/0.0%/0.0% | 0.6%/0.0%/0.0% | 1/0/0 | 1.0x/0.0x/0.0x |

### Coinvestors Dataset

#### Simple Model
| Method | Prediction (25M/100M/250M) | Recall (25M/100M/250M) | Counts (25M/100M/250M) | Improvement (25M/100M/250M) |
|--------|---------------------------|------------------------|----------------------|---------------------------|
| Best | 23.3%/10.0%/3.3% | 3.9%/7.9%/16.7% | 7/3/1 | 6.7x/13.5x/28.4x |
| Mean | 20.0%/6.7%/0.0% | 3.4%/5.3%/0.0% | 6/2/0 | 5.7x/9.0x/0.0x |
| Median | 23.3%/10.0%/3.3% | 3.9%/7.9%/16.7% | 7/3/1 | 6.7x/13.5x/28.4x |
| Mean Top | 20.0%/10.0%/0.0% | 3.4%/7.9%/0.0% | 6/3/0 | 5.7x/13.5x/0.0x |
| Median Top | 16.7%/13.3%/3.3% | 2.8%/10.5%/16.7% | 5/4/1 | 4.8x/17.9x/28.4x |
| LSQ | 6.7%/6.7%/0.0% | 1.1%/5.3%/0.0% | 2/2/0 | 1.9x/9.0x/0.0x |

#### Linear Model
| Method | Prediction (25M/100M/250M) | Recall (25M/100M/250M) | Counts (25M/100M/250M) | Improvement (25M/100M/250M) |
|--------|---------------------------|------------------------|----------------------|---------------------------|
| Best | 16.7%/6.7%/3.3% | 2.8%/5.3%/16.7% | 5/2/1 | 4.8x/9.0x/28.4x |
| Mean | 13.3%/10.0%/0.0% | 2.2%/7.9%/0.0% | 4/3/0 | 3.8x/13.5x/0.0x |
| Median | 16.7%/10.0%/0.0% | 2.8%/7.9%/0.0% | 5/3/0 | 4.8x/13.5x/0.0x |
| Mean Top | 20.0%/16.7%/3.3% | 3.4%/13.2%/16.7% | 6/5/1 | 5.7x/22.4x/28.4x |
| Median Top | 20.0%/16.7%/3.3% | 3.4%/13.2%/16.7% | 6/5/1 | 5.7x/22.4x/28.4x |
| LSQ | 0.0%/0.0%/0.0% | 0.0%/0.0%/0.0% | 0/0/0 | 0.0x/0.0x/0.0x |

#### Neural Model
| Method | Prediction (25M/100M/250M) | Recall (25M/100M/250M) | Counts (25M/100M/250M) | Improvement (25M/100M/250M) |
|--------|---------------------------|------------------------|----------------------|---------------------------|
| Best | 10.0%/10.0%/3.3% | 1.7%/7.9%/16.7% | 3/3/1 | 2.9x/13.5x/28.4x |
| Mean | 16.7%/6.7%/0.0% | 2.8%/5.3%/0.0% | 5/2/0 | 4.8x/9.0x/0.0x |
| Median | 16.7%/6.7%/0.0% | 2.8%/5.3%/0.0% | 5/2/0 | 4.8x/9.0x/0.0x |
| Mean Top | 20.0%/10.0%/0.0% | 3.4%/7.9%/0.0% | 6/3/0 | 5.7x/13.5x/0.0x |
| Median Top | 13.3%/10.0%/0.0% | 2.2%/7.9%/0.0% | 4/3/0 | 3.8x/13.5x/0.0x |
| LSQ | 0.0%/0.0%/0.0% | 0.0%/0.0%/0.0% | 0/0/0 | 0.0x/0.0x/0.0x |

### Investors with Graph Data

#### Simple Model
| Method | Prediction (25M/100M/250M) | Recall (25M/100M/250M) | Counts (25M/100M/250M) | Improvement (25M/100M/250M) |
|--------|---------------------------|------------------------|----------------------|---------------------------|
| Best | 33.3%/10.0%/3.3% | 5.6%/7.9%/16.7% | 10/3/1 | 9.5x/13.5x/28.4x |
| Mean | 26.7%/13.3%/6.7% | 4.5%/10.5%/33.3% | 8/4/2 | 7.6x/17.9x/56.8x |
| Median | 33.3%/20.0%/6.7% | 5.6%/15.8%/33.3% | 10/6/2 | 9.5x/26.9x/56.8x |
| Mean Top | 30.0%/13.3%/6.7% | 5.0%/10.5%/33.3% | 9/4/2 | 8.6x/17.9x/56.8x |
| Median Top | 26.7%/13.3%/6.7% | 4.5%/10.5%/33.3% | 8/4/2 | 7.6x/17.9x/56.8x |
| LSQ | 13.3%/6.7%/3.3% | 2.2%/5.3%/16.7% | 4/2/1 | 3.8x/9.0x/28.4x |

#### Linear Model
| Method | Prediction (25M/100M/250M) | Recall (25M/100M/250M) | Counts (25M/100M/250M) | Improvement (25M/100M/250M) |
|--------|---------------------------|------------------------|----------------------|---------------------------|
| Best | 30.0%/16.7%/3.3% | 5.0%/13.2%/16.7% | 9/5/1 | 8.6x/22.4x/28.4x |
| Mean | 30.0%/16.7%/6.7% | 5.0%/13.2%/33.3% | 9/5/2 | 8.6x/22.4x/56.8x |
| Median | 36.7%/23.3%/6.7% | 6.1%/18.4%/33.3% | 11/7/2 | 10.5x/31.4x/56.8x |
| Mean Top | 30.0%/16.7%/6.7% | 5.0%/13.2%/33.3% | 9/5/2 | 8.6x/22.4x/56.8x |
| Median Top | 23.3%/13.3%/6.7% | 3.9%/10.5%/33.3% | 7/4/2 | 6.7x/17.9x/56.8x |
| LSQ | 16.7%/6.7%/3.3% | 2.8%/5.3%/16.7% | 5/2/1 | 4.8x/9.0x/28.4x |

#### Neural Model
| Method | Prediction (25M/100M/250M) | Recall (25M/100M/250M) | Counts (25M/100M/250M) | Improvement (25M/100M/250M) |
|--------|---------------------------|------------------------|----------------------|---------------------------|
| Best | 0.0%/0.0%/0.0% | 0.0%/0.0%/0.0% | 0/0/0 | 0.0x/0.0x/0.0x |
| Mean | 13.3%/6.7%/3.3% | 2.2%/5.3%/16.7% | 4/2/1 | 3.8x/9.0x/28.4x |
| Median | 0.0%/0.0%/0.0% | 0.0%/0.0%/0.0% | 0/0/0 | 0.0x/0.0x/0.0x |
| Mean Top | 0.0%/0.0%/0.0% | 0.0%/0.0%/0.0% | 0/0/0 | 0.0x/0.0x/0.0x |
| Median Top | 0.0%/0.0%/0.0% | 0.0%/0.0%/0.0% | 0/0/0 |

The analysis reveals several significant patterns in startup success prediction. Our basic models generally performed best, achieving up to 9.5x improvement over random selection. Linear regression models showed similar strength, occasionally outperforming basic models, while neural networks surprisingly underperformed across all scenarios.

When comparing data types, we found that basic investor features and graph-enhanced features showed slightly better performance than coinvestor pairs. The addition of graph features provided marginal improvements in certain scenarios, particularly when identifying highly promising investments.

The "Best" combination method typically showed strongest performance, though differences between methods were not dramatic. Results were most significant when selecting top 30-100 investments, with performance declining for larger sets. For top 30 picks, the basic model achieved a 33.3% success rate, linear regression reached 30-36%, while neural networks struggled to exceed 5%.

## Conclusion

The superior performance of simpler models suggests that past success rates are strong predictors of future performance in venture capital. The failure of neural networks to match simpler approaches indicates that the relationship between investor characteristics and startup success may be more straightforward than initially hypothesized.

The strong performance when selecting smaller sets of investments suggests these methods are most valuable for identifying the most promising opportunities rather than providing broad market predictions. This aligns with the practical needs of venture capital firms, which typically make relatively few investments from a large pool of opportunities.

This research demonstrates that quantitative analysis of investor behavior and relationships can significantly improve startup success prediction, particularly for identifying the most promising investments. While simpler models often outperform more complex approaches, the integration of graph-based features and careful combination methods can provide additional predictive power. These findings have practical implications for venture capital decision-making and suggest promising directions for future research in startup success prediction.

## Limitations and Future Work

Our research faces several important limitations. Training and test split challenges arise from limited data availability and the interconnected nature of investor-investment relationships. Since we train the models using investor data, but we have to conduct our tests using startups data, we couldn't seperate the data after 2021 into test and training data. So the Linear and Neural models might have an overfitting problem and the results might be overstating their capabilities. But since the basic models, which don't require any training, mostly outperform the linear and neural models, this doesn't change our findings that basic models are a good method for predicting startup success and are better than more sophisticated methods.

The other problem is due to lack of data, since the Co-investor dataset doesn't contain every startup since 2013, the incomplete data might be giving inaccurate findings. Also, since there's no historical funding data, we evaluate the success of the past startups using the total funding they have today, which we wouldn't have access to if we wanted to use the model to predict future investments. So a timestamp of total fundings of startups in 2021 would give us more accurate findings of the potential success of our models. A future work might be addressing these issues and using more accurate and extensive data to analyse these models.


## References

[1] Abdullaev, M., Ihlamur, y., Alican, F., Hendrick, K., Darwazah, O. (2024). Midas Touch: Revealing Hidden Patterns Among Startup Investors with Graph-based Methods 

[2] Xiong, S., Ihlamur, Y., Alican, F., & Yin, A. (2024). GPTree: Towards Explainable Decision-Making via LLM-powered Decision Trees. arXiv:2411.08257.

[3] Xiong, S., & Ihlamur, Y. (2023). FounderGPT: Self-play to evaluate the Founder-Idea fit. arXiv:2312.12037.

[4] Ozince, E., & Ihlamur, Y. (2024). Automating Venture Capital: Founder assessment using LLM-powered segmentation, feature engineering and automated labeling techniques. arXiv:2407.04885.

[5] Gastaud, C., Carniel, T., & Dalle, J.-M. (2024). The varying importance of extrinsic factors in the success of startup fundraising: competition at early-stage and networks at growth-stage. arXiv:1906.03210.

[6] Lyu, S., et al. (2021). Cryptocurrency co-investment network: token returns reflect investment patterns. arXiv:2301.02027.

[7] Oravkin, E., & Ihlamur, Y. (2021). Midas Touch: Graph-based Investor Analysis. Retrieved from https://github.com/velapartners/midastouch-v1

[8] Piskorz, J., & Ihlamur, Y. (2022). Midas Touch v2: Enhanced Graph Analytics for Venture Capital. Retrieved from https://github.com/velapartners/midas_touch_v2

# For API Usage

In order to train the models, you have to run_analysis.py, then train_models.py and then test_models.py. You can check if anything went wrong by checking the printed results with the results in the repository (if you use the same dataset). While training the models, the INPUT_OUTPUT_SPLIT_YEAR must be set to 2021 (or some other past year) because there needs to be some output investments for the model to train on.

If you want to use the model with more accurate past data, update the past data you can change the dataset/original.csv with the more accurate data, change the INPUT_OUTPUT_SPLIT_YEAR to 2025 (so that all of the data will be used as input instead of reserving some for training the model) in constants.py file and then execute run_analysis.py (make sure that it doesn't execute train_models.py and test_models.py as they'll not work as intended with no testing data).

After training the models and providing the past investment data, using the API becomes simple. You just need to import the api file, load the strategy you want with `api.load_strat(strat_name)`, and then to predict any startup's success based on investors, you just have to use `api.get_prediction(investors)`, where investors are a python list of investors' uuids.

To specify a strategy with strat name you have to write it in the form datatype-name-model-type-name-aggregation-method-name.

Datatype names:
    investors: Investors without graph data
    coinvestors: Investor pair data
    investors-with-graph-data: Investors with graph data

Model type names:
    linear
    neural

Aggragation method names:
    best
    mean
    median
    mean_top
    median_top

So an example strat name is: investors-linear-best and a python code to load it up would be 
`api.load("investors-linear-best")`
`(res, p) = api.prediction(investor_uuids)`
`if res and p > 0.5:`
`   print('Buy')`
`else:`
`   print('Don\'t buy')`
