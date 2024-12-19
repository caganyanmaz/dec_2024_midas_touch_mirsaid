import numpy as np
import pandas as pd
import networkx as nx
import community as community_louvain
from collections import defaultdict
from constants import *


def main():
    investors_df = pd.read_csv(INPUT_INVESTORS_FILENAME)
    investments_df = pd.read_csv(INPUT_INVESTMENTS_FILENAME)
    coinvestors_df = pd.read_csv(INPUT_INVESTOR_PAIRS_FILENAME)
    #coinvestors_df.set_index('pair', inplace=True)

    print('A')
    adjacency_df = coinvestors_df.copy()

    # Split the 'pair' column into 'investor_a' and 'investor_b'
    adjacency_df[['investor_a', 'investor_b']] = adjacency_df['pair'].str.split(',', expand=True)

    # Create a new DataFrame with swapped 'investor_a' and 'investor_b'
    swapped_df = adjacency_df.copy()
    swapped_df['pair'] = swapped_df['investor_b'] + ',' + swapped_df['investor_a']

    # Concatenate the original and swapped DataFrames
    adjacency_df = pd.concat([adjacency_df, swapped_df])

    # Drop the temporary 'investor_a' and 'investor_b' columns
    adjacency_df = adjacency_df.drop(columns=['investor_a', 'investor_b'])
    adjacency_df.set_index('pair', inplace=True)



    # Initialize an undirected graph
    G = nx.Graph()
    # Add nodes with attributes from investors_df
    for _, row in investors_df.iterrows():
        G.add_node(
            row['investor_uuid'],
            annualised_investments=row['annualised_investment_count'],
            experience=row['experience'],
            weighted_success_rate=row['weighted_success_rate'],
            diversity_bucket=row['diversity_bucket'],
            success_bucket=row['success_bucket']
        )

    # Add edges with attributes
    for row in adjacency_df.itertuples():
        inv_a, inv_b = row.Index.split(',')
        
        # Add an edge with attributes
        G.add_edge(
            inv_a,
            inv_b,
            total_coinvestments=row.investment_count,
            avg_investor_experience=row.average_experience,
            focus_bucket_pair=row.diversity_bucket_pair,
            success_bucket_pair=row.success_bucket_pair,
            raw_outlier_rate=row.weighted_success_rate,
            scaled_success_rate=row.scaled_success_rate
        )
    for node in G.nodes():
        incident_edges = G.edges(node, data=True)
        rates = [d['scaled_success_rate'] for _, _, d in incident_edges if 'scaled_success_rate' in d]
        G.nodes[node]['average_outlier_rate'] = np.mean(rates) if rates else 0

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
    threshold_scaled_success_rate = 10  # Shared scaled outlier rate above this set threshold

    successful_partnerships = adjacency_df[
        (adjacency_df['investment_count'] >= threshold_coinvestments) &
        (adjacency_df['scaled_success_rate'] >= threshold_scaled_success_rate)
    ]

    # Rank partnerships by scaled_success_rate and total_coinvestments
    ranked_partnerships = successful_partnerships.sort_values(
        by=['scaled_success_rate', 'investment_count'], ascending=[False, False]
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
        weight_sums = {node: sum(data['scaled_success_rate'] for neighbor, data in graph[node].items()) for node in graph.nodes}

        for iteration in range(max_iter):
            new_scores = {}
            max_diff = 0

            for node in graph.nodes:
                # Calculate rank contributions from neighbors
                rank_sum = sum(
                    scores[neighbor] * data['scaled_success_rate'] / weight_sums[neighbor]
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
        partition = community_louvain.best_partition(G, weight='scaled_success_rate')
        modularity = community_louvain.modularity(partition, G)
        
        # Compute community stats
        community_groups = defaultdict(list)
        for node, community_id in partition.items():
            community_groups[community_id].append(node)

        # Calculate average scaled_success_rate for each community
        community_stats = []
        for community_id, nodes in community_groups.items():
            scaled_success_rates = [
                G.nodes[node].get('average_outlier_rate', 0) for node in nodes if node in G.nodes
            ]
            average_outlier_rate = sum(scaled_success_rates) / len(scaled_success_rates) if scaled_success_rates else 0
            community_stats.append({
                'community_id': community_id,
                'num_nodes': len(nodes),
                'average_outlier_rate': average_outlier_rate
            })
        
        # Store results
        results.append({
            'run': run,
            'modularity': modularity,
            'partition': partition,
            'community_stats': sorted(community_stats, key=lambda x: x['average_outlier_rate'], reverse=True)
        })

    # Find the strongest run based on modularity
    strongest_run = max(results, key=lambda x: x['modularity'])
    print(f"Strongest Run: {strongest_run['run']} with Modularity: {strongest_run['modularity']}")

    # Get the top communities for the strongest run
    partition = strongest_run['partition']
    community_groups = defaultdict(list)
    for node, community_id in partition.items():
        community_groups[community_id].append(node)

    # Filter and sort communities by average_outlier_rate, descending
    community_stats = []
    for community_id, nodes in community_groups.items():
        scaled_success_rates = [
            G.nodes[node].get('average_outlier_rate', 0) for node in nodes if node in G.nodes
        ]
        average_outlier_rate = sum(scaled_success_rates) / len(scaled_success_rates) if scaled_success_rates else 0
        if len(nodes) >= 5:
            community_stats.append({
                'community_id': community_id,
                'num_nodes': len(nodes),
                'average_outlier_rate': average_outlier_rate,
                'nodes': nodes
            })

    # Sort by average_outlier_rate descending
    community_stats = sorted(community_stats, key=lambda x: x['average_outlier_rate'], reverse=True)

    # Save community compositions
    community_compositions = []
    for rank, community in enumerate(community_stats, start=1):
        community_id = community['community_id']
        num_nodes = community['num_nodes']
        average_outlier_rate = community['average_outlier_rate']
        nodes = community['nodes']

        # Extract node data for the community
        community_data = pd.DataFrame([
            {
                'investor_uuid': node,
                'annualised_investments': G.nodes[node].get('annualised_investments', 0),
                'experience': G.nodes[node].get('experience', 0),
                'weighted_success_rate': G.nodes[node].get('weighted_success_rate', 0),
                'diversity_bucket': G.nodes[node].get('diversity_bucket', ''),
                'success_bucket': G.nodes[node].get('success_bucket', '')
            }
            for node in nodes
        ])

        # Save to a separate CSV file
        filename = f"dataset/communities/community_{rank}.csv"
        community_data.to_csv(filename, index=False)

        # Add to compositions
        focus_counts = community_data['diversity_bucket'].value_counts().to_dict()
        bucket_counts = community_data['success_bucket'].value_counts().to_dict()

        community_compositions.append({
            'Rank': rank,
            'Community_ID': community_id,
            'Avg_Outlier_Rate': average_outlier_rate,
            'Node_Count': num_nodes,
            'Avg_Experience': community_data['experience'].mean(),
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

if __name__ == "__main__":
    main()
