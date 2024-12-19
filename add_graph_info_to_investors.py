import pandas as pd
from constants import *


def main():
    input_investors_df = pd.read_csv(INPUT_INVESTORS_FILENAME)
    investor_ranks_df = pd.read_csv(INVESTOR_RANK_SCORES_FILENAME)
    centrality_df = pd.read_csv(CENTRALITY_ANALYSIS_FILENAME)

    input_investors_df = input_investors_df.merge(investor_ranks_df, on='investor_uuid')
    input_investors_df = input_investors_df.merge(centrality_df, on='investor_uuid')
    input_investors_df.set_index('investor_uuid', inplace=True) 
    

    # Rank,Community_ID,Avg_Outlier_Rate,Node_Count,Avg_Experience,Balanced,Diverse,Focused,Specialist,Universalist,L1,L2,L3,L4,L5
    community_info_df = {
            'community_average_outlier_rate': dict(), 
            'community_node_count': dict(), 
            'community_average_experience': dict(), 
            'community_balanced_count': dict(),
            'community_diverse_count': dict(),
            'community_focused_count': dict(),
            'community_specialist_count': dict(),
            'community_universalist_count': dict(),
            'community_L1_count': dict(),
            'community_L2_count': dict(),
            'community_L3_count': dict(),
            'community_L4_count': dict(),
            'community_L5_count': dict()
    }
    communities_df = pd.read_csv(COMMUNITY_COMPOSITION_FILENAME)
    for community in communities_df.itertuples():
        community_df = pd.read_csv(f'{COMMUNITY_DIRECTORY}/community_{community.Rank}.csv')
        for member in community_df.itertuples():
            if member.investor_uuid not in input_investors_df.index:
                continue
            investor_uuid = member.investor_uuid
            community_info_df['community_average_outlier_rate'][investor_uuid] = community.Avg_Outlier_Rate
            community_info_df['community_node_count'][investor_uuid] = community.Node_Count
            community_info_df['community_average_experience'][investor_uuid] = community.Avg_Experience
            community_info_df['community_balanced_count'][investor_uuid] = community.Balanced
            community_info_df['community_diverse_count'][investor_uuid] = community.Diverse
            community_info_df['community_focused_count'][investor_uuid] = community.Focused
            community_info_df['community_specialist_count'][investor_uuid] = community.Specialist
            community_info_df['community_universalist_count'][investor_uuid] = community.Universalist
            community_info_df['community_L1_count'][investor_uuid] = community.L1
            community_info_df['community_L2_count'][investor_uuid] = community.L2
            community_info_df['community_L3_count'][investor_uuid] = community.L3
            community_info_df['community_L4_count'][investor_uuid] = community.L4
            community_info_df['community_L5_count'][investor_uuid] = community.L5
    for key, value in community_info_df.items():
        input_investors_df[key] = value
    input_investors_df.fillna(0)
    
    input_investors_df.to_csv(INPUT_INVESTORS_WITH_GRAPH_DATA_FILENAME)





if __name__ == '__main__':
    main()
