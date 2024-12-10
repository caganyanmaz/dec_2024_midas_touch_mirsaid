import networkx as nx

# Notes:
# Nodes: investor_uuid with {annualised_investments,
#                            investing_experience,
#                            outlier_bucket,
#                            focus_classification,
#                            outlier_score}
# 
# Edges: vector {num_coinvestments,
#                outlier_rate (weighted outlier score for shared startups),
#                avg_investor_exp, 
#                focus_buckets,
#                outlier_buckets}


# TODO : Implement KG here