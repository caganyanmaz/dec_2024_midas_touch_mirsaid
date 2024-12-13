import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
from collections import defaultdict

kg_investors_df = pd.read_csv("dataset/kg_investors.csv")
coinvestor_df = pd.read_csv("dataset/coinvestor_clean.csv")

# weights to sum the outlier_rates for each investor (TWEAK THIS TO PLAY WITH THE OUTLIER SCORE FOR EACH NODE IN THE GRAPH)
WEIGHTS = {
    "250m": 0.85,
    "100m": 0.1,
    "25m": 0.05
}
# Threshold for annualized investments to penalise coinvestor outlier rates (no scale up occurs, only scale down happens if annualised investments are below this threshold for each investor)
ANN_INVESTMENT_THRESHOLD = 1.5  


# STEP 1: begin by creating an adjacency matrix of all coinvestors from the coinvestor_df table

# Filter valid investors from kg_investors_df
valid_investors = set(kg_investors_df['investor_uuid'])

# Filter coinvestor_df to only include rows where at least one investor is valid
filtered_coinvestor_df = coinvestor_df[
    coinvestor_df['investor_uuids'].apply(
        lambda uuids: any(uuid in valid_investors for uuid in uuids.split(','))
    )
]

# Initialize an empty dictionary for adjacency matrix
adjacency_accumulator = defaultdict(int)

# Iterate through kg_investors_df to build the adjacency matrix
for investor in kg_investors_df['investor_uuid']:
    # Filter rows from coinvestor_df that involve the current investor
    investor_rows = filtered_coinvestor_df[filtered_coinvestor_df['investor_uuids'].str.contains(investor, na=False)]

    # Iterate through the startups these rows represent
    for _, row in investor_rows.iterrows():
        # Extract all investors in the same startup
        investors_in_startup = row['investor_uuids'].split(',')

        # Update the adjacency accumulator for all pairs involving the current investor
        for co_investor in investors_in_startup:
            if co_investor != investor and co_investor in valid_investors:
                # Ensure pairs are always stored in a consistent order to avoid duplicate pairs in the opposite configuration
                pair = tuple(sorted([investor, co_investor]))
                adjacency_accumulator[pair] += 1

# Convert the accumulator to a DataFrame
adjacency_data = [
    {'investor_a': pair[0], 'investor_b': pair[1], 'total_coinvestments': count}
    for pair, count in adjacency_accumulator.items()
]
adjacency_df = pd.DataFrame(adjacency_data)
# Save the adjacency matrix as a DataFrame
adjacency_df.to_csv("dataset/adjacency_matrix.csv", index=False)


def calculate_edge_attributes(row):
    investor_a, investor_b = row['investor_a'], row['investor_b']
    
    # Get rows in coinvestor_df related to both investors
    related_rows = coinvestor_df[
        coinvestor_df['investor_uuids'].str.contains(investor_a) &
        coinvestor_df['investor_uuids'].str.contains(investor_b)
    ]
    
    # Get investors' data
    inv_a_data = kg_investors_df.loc[kg_investors_df['investor_uuid'] == investor_a]
    inv_b_data = kg_investors_df.loc[kg_investors_df['investor_uuid'] == investor_b]
            
    # Calculate average experience
    avg_investor_experience = (
        inv_a_data['investing_experience'].iloc[0] +
        inv_b_data['investing_experience'].iloc[0]) / 2
    
    # Get focus and outlier bucket pairs
    focus_bucket_pair = f"{inv_a_data['focus_classification'].iloc[0]},{inv_b_data['focus_classification'].iloc[0]}"
    outlier_bucket_pair = f"{inv_a_data['outlier_bucket'].iloc[0]},{inv_b_data['outlier_bucket'].iloc[0]}"
    
    return {
        'avg_investor_experience': avg_investor_experience,
        'focus_bucket_pair': focus_bucket_pair,
        'outlier_bucket_pair': outlier_bucket_pair
    }

# Apply the function to calculate edge attributes
edge_attributes = adjacency_df.apply(calculate_edge_attributes, axis=1)

# Convert the dictionary results to a DataFrame
edge_attributes_df = pd.DataFrame(edge_attributes.tolist())

# Concatenate the edge attributes with the adjacency matrix
adjacency_df = pd.concat([adjacency_df, edge_attributes_df], axis=1)

# Save the updated adjacency matrix
adjacency_df.to_csv("dataset/adjacency_matrix.csv", index=False)


# Initialize the raw_outlier_rate column
adjacency_df['raw_outlier_rate'] = 0.0

# Iterate over the adjacency matrix
for idx, row in adjacency_df.iterrows():
    investor_a = row['investor_a']
    investor_b = row['investor_b']
    
    # Filter coinvestor_df for shared investments between investor_a and investor_b
    shared_investments = coinvestor_df[
        coinvestor_df['investor_uuids'].apply(lambda x: investor_a in x and investor_b in x)
    ]
    
    # Calculate the weighted outlier rate for shared investments
    if not shared_investments.empty:
        raw_outlier_rate = (
            WEIGHTS['250m'] * shared_investments['ultimate_outlier_success_250_mil_raise'].mean() +
            WEIGHTS['100m'] * shared_investments['interim_success_100_mil_founded_year_2019_or_above'].mean() +
            WEIGHTS['25m'] * shared_investments['recent_success_25_mil_raise_founded_year_2022_or_above'].mean()
        ) * 100
        adjacency_df.loc[idx, 'raw_outlier_rate'] = raw_outlier_rate

# Save the updated adjacency matrix
adjacency_df.to_csv("dataset/adjacency_matrix.csv", index=False)


# Pull annualized investments from kg_investors_df
investor_annual_investments = kg_investors_df.set_index('investor_uuid')['annualised_investments_2013'].to_dict()


# Define the harsher penalty scaling function
def penalize_low_values(raw_outlier_rate, avg_ann_investments):
    penalty = 1.0  # Start with no penalty
    
    # Penalize more harshly for low average annualized investments
    if avg_ann_investments < ANN_INVESTMENT_THRESHOLD:
        penalty -= 1.5 * (ANN_INVESTMENT_THRESHOLD - avg_ann_investments) / ANN_INVESTMENT_THRESHOLD
        
    # Apply the penalty but ensure it's not negative
    penalty = max(penalty, 0.2)  # Minimum penalty to avoid extreme reductions
    
    # Scale down the raw_outlier_rate by the harsher penalty
    return raw_outlier_rate * penalty

# Apply the penalization logic to the adjacency matrix
def compute_penalized_outlier_rate(adjacency_row):
    raw_rate = adjacency_row['raw_outlier_rate']
    
    # Retrieve annualized investments for both investors
    inv_a = adjacency_row['investor_a']
    inv_b = adjacency_row['investor_b']
    ann_invest_a = investor_annual_investments.get(inv_a, 0)
    ann_invest_b = investor_annual_investments.get(inv_b, 0)
    
    # Compute average annualized investments
    avg_ann_investments = (ann_invest_a + ann_invest_b) / 2
    
    # Apply harsher penalization
    return penalize_low_values(raw_rate, avg_ann_investments)

# Add harsher scaled_outlier_rate column to adjacency_df
adjacency_df['scaled_outlier_rate'] = adjacency_df.apply(compute_penalized_outlier_rate, axis=1)

adjacency_df.to_csv("dataset/adjacency_matrix.csv", index=False)


# Initialize an undirected graph
G = nx.Graph()
# Add nodes with attributes from kg_investors_df
for _, row in kg_investors_df.iterrows():
    G.add_node(
        row['investor_uuid'],
        annualised_investments=row['annualised_investments_2013'],
        investing_experience=row['investing_experience'],
        outlier_score=row['outlier_score'],
        focus_classification=row['focus_classification'],
        outlier_bucket=row['outlier_bucket']
    )

# Add edges with attributes
for _, row in adjacency_df.iterrows():
    inv_a = row['investor_a']
    inv_b = row['investor_b']
    
    # Add an edge with attributes
    G.add_edge(
        inv_a,
        inv_b,
        total_coinvestments=row['total_coinvestments'],
        avg_investor_experience=row['avg_investor_experience'],
        focus_bucket_pair=row['focus_bucket_pair'],
        outlier_bucket_pair=row['outlier_bucket_pair'],
        raw_outlier_rate=row['raw_outlier_rate'],
        scaled_outlier_rate=row['scaled_outlier_rate']
    )
for node in G.nodes():
    incident_edges = G.edges(node, data=True)
    rates = [d['scaled_outlier_rate'] for _, _, d in incident_edges if 'scaled_outlier_rate' in d]
    G.nodes[node]['avg_outlier_rate'] = np.mean(rates) if rates else 0

# Save the graph
nx.write_gexf(G, "dataset/knowledge_graph.gexf")


# ANALYSIS 1: Degree Centrality
degree_centrality = nx.degree_centrality(G)

# ANALYSIS 2: Eigenvector Centrality
eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)

# Combine centrality metrics into a DataFrame for analysis
centrality_df = pd.DataFrame({
    'investor_uuid': list(degree_centrality.keys()),
    'degree_centrality': list(degree_centrality.values()),
    'eigenvector_centrality': [eigenvector_centrality.get(node, 0) for node in degree_centrality.keys()]
})

# Sort by centrality metrics to identify top coinvestors
centrality_df = centrality_df.sort_values(by=['degree_centrality', 'eigenvector_centrality'], ascending=[False, False])
# Save the results to CSV for further inspection
centrality_df.to_csv("dataset/centrality_analysis.csv", index=False)

# ANALYSIS 3: Filter for successful partnerships
threshold_coinvestments = 5  # At least 5 shared startups
threshold_scaled_outlier_rate = 10  # Shared scaled outlier rate above this set threshold

successful_partnerships = adjacency_df[
    (adjacency_df['total_coinvestments'] >= threshold_coinvestments) &
    (adjacency_df['scaled_outlier_rate'] >= threshold_scaled_outlier_rate)
]

# Rank partnerships by scaled_outlier_rate and total_coinvestments
ranked_partnerships = successful_partnerships.sort_values(
    by=['scaled_outlier_rate', 'total_coinvestments'], ascending=[False, False]
)

# Summaries
print("Total partnerships analyzed:", len(adjacency_df))
print("Number of successful partnerships:", len(successful_partnerships))
print("Top Successful Partnerships:")
print(ranked_partnerships)


# ANALYSIS 4 - InvestorRank algorithm (modified PageRank algorithm for finding the most influential investors in the graph)

# Define the InvestorRank algorithm for undirected graphs
def investor_rank(graph, alpha=0.85, max_iter=100, tol=1e-6):
    # Compute InvestorRank scores for an undirected graph
    # Initialize scores equally for all nodes
    num_nodes = len(graph.nodes)
    scores = {node: 1 / num_nodes for node in graph.nodes}

    # Precompute the sum of weights for each node
    weight_sums = {node: sum(data['scaled_outlier_rate'] for neighbor, data in graph[node].items()) for node in graph.nodes}

    for iteration in range(max_iter):
        new_scores = {}
        max_diff = 0

        for node in graph.nodes:
            # Calculate rank contributions from neighbors
            rank_sum = sum(
                scores[neighbor] * data['scaled_outlier_rate'] / weight_sums[neighbor]
                for neighbor, data in graph[node].items()
                if weight_sums[neighbor] > 0
            )

            # Apply damping factor and handle teleportation
            new_scores[node] = (1 - alpha) / num_nodes + alpha * rank_sum

            # Track the maximum change in scores for convergence check
            max_diff = max(max_diff, abs(new_scores[node] - scores[node]))

        # Update scores
        scores = new_scores

        # Check for convergence
        if max_diff < tol:
            print(f"InvestorRank converged after {iteration + 1} iterations.")
            break
    else:
        print("InvestorRank did not converge within the maximum number of iterations.")

    return scores

# Run the InvestorRank algorithm
investor_rank_scores = investor_rank(G)

# Sort and display the top investors by rank
sorted_investors = sorted(investor_rank_scores.items(), key=lambda x: x[1], reverse=True)
print("Top 10 Investors by InvestorRank:")
for investor, score in sorted_investors[:50]:
    print(f"Investor: {investor}, Rank Score: {score:.6f}")

scores_df = pd.DataFrame(sorted_investors, columns=['investor_uuid', 'InvestorRank_score'])
scores_df.to_csv("dataset/investor_rank_scores.csv", index=False)


# ANALYSIS 5: Louvain Algorithm for Community Detection
num_runs = 20
results = []

for run in range(num_runs):
    # Run Louvain
    partition = community_louvain.best_partition(G, weight='scaled_outlier_rate')
    modularity = community_louvain.modularity(partition, G)
    
    # Compute community stats
    community_groups = defaultdict(list)
    for node, community_id in partition.items():
        community_groups[community_id].append(node)

    # Calculate average scaled_outlier_rate for each community
    community_stats = []
    for community_id, nodes in community_groups.items():
        scaled_outlier_rates = [
            G.nodes[node].get('avg_outlier_rate', 0) for node in nodes if node in G.nodes
        ]
        avg_outlier_rate = sum(scaled_outlier_rates) / len(scaled_outlier_rates) if scaled_outlier_rates else 0
        community_stats.append({
            'community_id': community_id,
            'num_nodes': len(nodes),
            'avg_outlier_rate': avg_outlier_rate
        })
    
    # Store results
    results.append({
        'run': run,
        'modularity': modularity,
        'partition': partition,
        'community_stats': sorted(community_stats, key=lambda x: x['avg_outlier_rate'], reverse=True)
    })

# Find the strongest run based on modularity
strongest_run = max(results, key=lambda x: x['modularity'])
print(f"Strongest Run: {strongest_run['run']} with Modularity: {strongest_run['modularity']}")

# Get the top communities for the strongest run
partition = strongest_run['partition']
community_groups = defaultdict(list)
for node, community_id in partition.items():
    community_groups[community_id].append(node)

# Filter and sort communities by avg_outlier_rate, descending
community_stats = []
for community_id, nodes in community_groups.items():
    scaled_outlier_rates = [
        G.nodes[node].get('avg_outlier_rate', 0) for node in nodes if node in G.nodes
    ]
    avg_outlier_rate = sum(scaled_outlier_rates) / len(scaled_outlier_rates) if scaled_outlier_rates else 0
    if len(nodes) >= 5:
        community_stats.append({
            'community_id': community_id,
            'num_nodes': len(nodes),
            'avg_outlier_rate': avg_outlier_rate,
            'nodes': nodes
        })

# Sort by avg_outlier_rate descending
community_stats = sorted(community_stats, key=lambda x: x['avg_outlier_rate'], reverse=True)

# Save community compositions
community_compositions = []
for rank, community in enumerate(community_stats, start=1):
    community_id = community['community_id']
    num_nodes = community['num_nodes']
    avg_outlier_rate = community['avg_outlier_rate']
    nodes = community['nodes']

    # Extract node data for the community
    community_data = pd.DataFrame([
        {
            'investor_uuid': node,
            'annualised_investments': G.nodes[node].get('annualised_investments', 0),
            'investing_experience': G.nodes[node].get('investing_experience', 0),
            'outlier_score': G.nodes[node].get('outlier_score', 0),
            'focus_classification': G.nodes[node].get('focus_classification', ''),
            'outlier_bucket': G.nodes[node].get('outlier_bucket', '')
        }
        for node in nodes
    ])

    # Save to a separate CSV file
    filename = f"dataset/communities/community_{rank}.csv"
    community_data.to_csv(filename, index=False)

    # Add to compositions
    focus_counts = community_data['focus_classification'].value_counts().to_dict()
    bucket_counts = community_data['outlier_bucket'].value_counts().to_dict()

    community_compositions.append({
        'Rank': rank,
        'Community ID': community_id,
        'Avg Outlier Rate': avg_outlier_rate,
        'Node Count': num_nodes,
        'Avg Experience': community_data['investing_experience'].mean(),
        'Balanced': focus_counts.get('Balanced', 0),
        'Diverse': focus_counts.get('Diverse', 0),
        'Focused': focus_counts.get('Focused', 0),
        'Specialist': focus_counts.get('Specialist', 0),
        'Universalist': focus_counts.get('Universalist', 0),
        'L1': bucket_counts.get('L1', 0),
        'L2': bucket_counts.get('L2', 0),
        'L3': bucket_counts.get('L3', 0),
        'L4': bucket_counts.get('L4', 0),
        'L5': bucket_counts.get('L5', 0)
    })

# Save community composition summary
community_compositions_df = pd.DataFrame(community_compositions)
community_compositions_df.to_csv("dataset/communities/community_composition.csv", index=False)

print("Community analysis completed. Files saved.")