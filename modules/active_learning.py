import pandas as pd 
from scipy.spatial.distance import cdist

def random_selection(current_sample, not_labelled_df, step, seed):
    # Get new sample
    new_labels = not_labelled_df.sample(n=step, random_state = seed)

    # Update datasets
    not_labelled_df_new = not_labelled_df.drop(new_labels.index)
    current_sample_new = pd.concat([current_sample, new_labels])
    
    return current_sample_new, not_labelled_df_new

def uncertainty_selection(current_sample, non_labelled_df, step, model, full_X_train_scaled):
    """
    Select points based on maximum uncertainty
    """
    predicted_proba = model.predict_proba(full_X_train_scaled)
    predicted_proba_df = pd.DataFrame(predicted_proba[:,1], index=non_labelled_df.index)
    predicted_proba_df['uncertainty_diff'] = (0.5 - predicted_proba_df[0]).abs()
    top_50_uncertain_points = predicted_proba_df.nsmallest(step, 'uncertainty_diff')

    to_be_labelled = non_labelled_df.loc[top_50_uncertain_points.index]

    # Update labelled_indexes with the indexes of the top 50 uncertain points
    current_sample_new = pd.concat([current_sample, to_be_labelled])
    non_labelled_df_new = non_labelled_df.drop(top_50_uncertain_points.index)
    
    return current_sample_new, non_labelled_df_new

def distribution_selection(current_sample, non_labelled_df, step):
    """
    Selects points based on maximum dissimilarity between the already labelled
    """

    # Dissimilarity based selection
    distances = cdist(non_labelled_df, current_sample, metric='euclidean')
    mean_distances = distances.mean(axis=1)
    dissimilarity_df = pd.DataFrame(mean_distances, index=non_labelled_df.index, columns=['dissimilarity'])
    dissimilar_points = dissimilarity_df.nlargest(step, 'dissimilarity')
    to_be_labelled_dissimilarity = non_labelled_df.loc[dissimilar_points.index]

    current_sample_new = pd.concat([current_sample, to_be_labelled_dissimilarity])
    non_labelled_df_new = non_labelled_df.drop(dissimilar_points.index)

    return current_sample_new, non_labelled_df_new

def duality_selection(dual_df, df_Y, z_hat_df, current_sample, not_labelled_df, step, cost_2_df):
    """"
    Selects points twice: once based on duality, the other based on the network output costs
    """
    ### O SET - DUAL VARIABLES
    dual_df_temp = pd.merge(dual_df, z_hat_df[['z_hat_adjusted']], left_index=True, right_index=True, how='left')
    O_set = dual_df_temp[(dual_df_temp['z_hat_adjusted']==0) & (dual_df_temp['Dual Value']!=0)].sort_values(by='Dual Value', ascending=True)
    O_set_without_labelled = O_set[~O_set.index.isin(current_sample.index)]
    indexes = O_set_without_labelled.head(int(step/2)).index
    new_labels = not_labelled_df.loc[indexes]
    
    not_labelled_df_new = not_labelled_df.drop(new_labels.index)
    current_sample_new = pd.concat([current_sample, new_labels])

    ### I SET - COST
    merged_cost = pd.merge(df_Y, cost_2_df, on=['j', 'k'], how='left')
    I_set = merged_cost.groupby('k')[['Value']].sum().sort_values(by='Value', ascending=False).reset_index()
    I_set_without_labelled = I_set[~I_set['k'].isin(current_sample_new.index)]
    indexes = I_set_without_labelled.head(int(step/2))['k'].tolist()
    new_labels = not_labelled_df_new.loc[indexes]
    
    not_labelled_df_return = not_labelled_df_new.drop(new_labels.index)
    current_sample_return = pd.concat([current_sample_new, new_labels])

    return current_sample_return, not_labelled_df_return

