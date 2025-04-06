# Helper functions to make recommendations

from code import interact
import pandas as pd
import numpy as np
import pickle
import json

def get_top_items(model, user_id: str, item_ids, interactions_matrix):
    """Predict ranking of items for a user given a model and a matrix of interactions. 

    Args:
        model (object): trained recommender system
        user_id (str): user id
        item_ids (list): list of items
        interactions_matrix (array): matrix n_users x n_items containing interactions
    """

    _, n_items = interactions_matrix.shape
    scores = model.predict(user_id, np.arange(n_items))

    # Careful here item ids are partner ids
    top_item_ids = item_ids[np.argsort(-scores)]

    return(top_item_ids)

def is_tagged_item(item_id, tag_ids, items_dict):
    """Check if an item is tagged according to tag ids

    Args:
        item_id (str): id of the item
        tag_ids (list): list of tags
        items_dict (dictionary): mapping from an item id to a list of corresponding tag ids

    Returns:
        bool: whether or not the item is tagged according to all of the tags
    """

    # Careful here item_id and tag_ids are ids
    item_tags = items_dict[item_id]
    return all(tag in item_tags for tag in tag_ids)

def find_relevant_recos(top_item_ids, tag_ids, items2tags, top_k = 10):
    """In a ranked list of items, find the ones that match the provided tags.

    Args:
        top_item_ids (list): tanked list of item ids
        tag_ids (list): list of tags to filter the items on
        items2tags (dict): mapping from an item id to a list of corresponding tags
        top_k (int, optional): number of relevant items to output. Defaults to 10.

    Returns:
        list: list of item ids relevant to the tags provided in tag_ids
    """
    
    relevant_items = [item for item in top_item_ids if \
                        is_tagged_item(item, tag_ids, items2tags)]
    if top_k is None:
        return relevant_items
    else:
        return(relevant_items[:min(top_k, len(relevant_items))])

def sample_recommendation(model, member_id, tag_ids, interactions_matrix, item_ids, items2tags, 
items2labels, labels2items, tags2names, user_id_map, top_k = 10):
    """Make recommendations given a member id, tags to filter the partners on, and a trained model

    Args:
        model (obj): trained model for recommendation
        member_id (str): user id
        tag_ids (list): list of tag ids to filter recommendations on
        interactions_matrix (array): array of interactions
        item_ids (list): list of item ids
        items2tags (dict): mapping of item ids to their corresponding tags
        items2labels (dict): mapping of item ids to their labels
        labels2items (dict): mapping of item labels to their ids
        tags2names (dict): mapping of tag ids to their label
        user_id_map (dict): mapping of member ids to the model's internal user ids
        top_k (int, optional): number of recommended items to output. Defaults to 10.
    
    Returns:
        json: recommendation json
    """

    _, n_items = interactions_matrix.shape
    user_id = user_id_map[member_id]

    # Recommendations
    top_item_ids = get_top_items(model, user_id, item_ids, interactions_matrix)
    relevant_item_ids = find_relevant_recos(top_item_ids, tag_ids, items2tags, top_k=top_k)
    # top_relevant_items = [items2labels[item_id] for item_id in relevant_item_ids]

    # Known positives
    known_positives_ids = item_ids[interactions_matrix.tocsr()[user_id].indices]
    relevant_known_positives_ids = find_relevant_recos(known_positives_ids, tag_ids, items2tags, top_k=None)
    # relevant_known_positives = [items2labels[item_id] for item_id in relevant_known_positives_ids]
    
    true_positive_ids = [item_id for item_id in relevant_item_ids \
                        if item_id in relevant_known_positives_ids]

    # Tag Labels
    tag_labels = [tags2names[tag] for tag in tag_ids]
    tag_labels = ', '.join(tag_labels)

    # Json output
    reco = {"user": member_id,
            "tags": tag_labels,
            "recommended": [
                {"PartnerId": item_id,
                "PartnerName":items2labels[item_id] } for item_id in relevant_item_ids
                ],
            "positives": [
                {"PartnerId": item_id,
                "PartnerName":items2labels[item_id] } for item_id in relevant_known_positives_ids
                ],
            "common": [
                {"PartnerId": item_id,
                "PartnerName":items2labels[item_id] } for item_id in true_positive_ids
                ]
            }
    
    return(reco)

def display_reco(reco):
    # Function to display the recommendation

    print("User %s" % reco["user"])
    print("Criteria: %s" % reco["tags"])

    print("     Recommended:")

    for partner in reco["recommended"]:
        print("        {} (PartnerId: {})".format(partner["PartnerName"], partner["PartnerId"]))

    print("     Known positives:")

    for partner in reco["positives"]:
        print("        {} (PartnerId: {})".format(partner["PartnerName"], partner["PartnerId"]))

    print("     In common:")

    for partner in reco["common"]:
        print("        {} (PartnerId: {})".format(partner["PartnerName"], partner["PartnerId"]))

def recommend(recsys_model, user_id, dataset, tag_ids, tag_names = None, top_k = 5):
    """Function to make recommendations based on a user and tags. 

    Args:
        recsys_model (LightFM object): trained recommendation model
        user_id (str): user id to make recommendation for
        dataset (UserItemData object): dataset containing user item data
        tag_ids (list): list of tag ids to filter items on
        tag_names (list, optional): list of tag names to be provided instead of tag ids. Defaults to None.
        If tag_ids is provided and tag_names is provided, the tag_names argument will be ignored. 
        top_k (int, optional): maximal number of recommendations to make. Defaults to 5.

    Returns:
        json: recommendation json
    """

    # If tag names are provided and no tag ids are provided
    # Convert those names to ids
    # If both are provided, tag_ids argument takes precedence over tag_names argument, which is ignored
    if tag_ids is None and tag_names is not None:
        tag_ids = [dataset.tags_labels2ids[tag] for tag in tag_names]
    reco = sample_recommendation(recsys_model,
                            user_id, 
                            tag_ids, 
                            interactions_matrix= dataset.interactions, 
                            item_ids = dataset.item_ids, 
                            items2tags = dataset.items_ids2tags, 
                            items2labels = dataset.items_ids2labels, 
                            labels2items = dataset.items_labels2ids,
                            tags2names = dataset.tags_ids2labels, 
                            user_id_map = dataset.user_id_map,
                            top_k = top_k)
    return reco


def save_model(model, pkl_filename):
    # Helper function to save a recommender model
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
        print('Model saved in {}'.format(pkl_filename))

def load_model(pkl_filename):
    # Helper function to load a recommender model
    with open(pkl_filename, 'rb') as file:
        pickle_model = pickle.load(file)
    
    return pickle_model

def save_reco(reco, file_name):
    # Helper function to save a recommendation to a json
    with open(file_name, "w") as f:
        json.dump(reco, f, indent = 4, ensure_ascii=False)
    
    f.close()
