# Creating a data set class to process and save user-item data and interactions
# Data set will enable training of a recommendation system
import sys
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from lightfm.data import Dataset

class UserItemData(Dataset):
    def __init__(self, experiences2partners, item_features = None, tf_idf_features = False):
        """Initialization method for the data set.

        Args:
            experiences2partners (pd.DataFrame): table of experiences associated to partners.
            item_features (pd.DataFrame, optional): table of items associated to features. Defaults to None.
            tf_idf_features (bool, optional): if set to True, item features will be processed and weighted 
            according to TF-IDF. Defaults to False.
        """
        super().__init__()

        # Create interactions dictionary
        self.interactions = self.process_experiences(experiences2partners)
        # Create user and item ID mappings
        self.create_mappings(experiences2partners)

        if item_features is not None:
            # If item features are provided, encode them to enable content-based recommendations
            self.encode_features(item_features, tf_idf_features = tf_idf_features)

        self.fit()

        # Compute interactions and weight of interactions
        self.interactions, self.weights = self.build_interactions()

        # Define mappings
        self.user_id_map, self.user_feature_map, \
        self.item_id_map, self.item_feature_map = self.mapping()

        if not tf_idf_features:
            self.item_features = self.build_item_features((
                (x["PartnerId"], x["feature_list"]) for x in self.item_features_dict))  


    def process_experiences(self, experiences2partners):
        """ Process table to get a table n_users x n_items containing, in each cell
        the number of interactions between a user and an item. """
        
        interactions = pd.pivot_table(
            experiences2partners[['ExperienceName', 
                                'MemberId', 
                                'PartnerId']], 
            values = 'ExperienceName',
            index = 'MemberId',
            columns = 'PartnerId',
            aggfunc = 'count',
            fill_value = 0)
        
        return interactions
    
    def create_mappings(self, experiences2partners):
        """ Create dictionaries (mappings) of item ids to their labels and vice versa.""" 

        ##################### ITEM ID - ITEM LABEL MAPPING ####################
        partners2names_dict = experiences2partners[['PartnerId','PartnerName']].to_dict(orient = 'records')
        self.items_ids2labels = {str(partner['PartnerId']): partner['PartnerName'] for partner in partners2names_dict}
        self.items_labels2ids = {v:k for k,v in self.items_ids2labels.items()}

        self.interactions_dict = self.interactions_to_dict()
        # Retrieve ids and labels
        self.user_ids = np.unique([x["MemberId"] for x in self.interactions_dict])
        self.item_ids = np.unique([x["PartnerId"] for x in self.interactions_dict])
        self.item_labels = np.unique([x["PartnerName"] for x in self.interactions_dict])

    def interactions_to_dict(self):
        """ Build dictionary of interactions """
        interactions_json = self.interactions.to_json(orient = 'index')
        interactions_json = json.loads(interactions_json)

        interactions_dict = [{"MemberId": member,
                            "PartnerId": partner,
                            "PartnerName": self.items_ids2labels[partner],
                            "n_interactions": interactions_json[member][partner]} 
                            for member in interactions_json.keys()
                                for partner in  interactions_json[member].keys()]
        return interactions_dict

    def encode_features(self, items_table, tf_idf_features):
        """Encode item features into a matrix (here, partner tags)

        Args:
            items_table (_type_): items data frame
            tf_idf_features (bool): whether or not to use tf-idf weighting for item features
        """
        # One-Hot encode items (partners)
        self.one_hot_items = pd.get_dummies(
            items_table[items_table.PartnerId.astype(str).isin(self.item_ids)][['TagId', 'PartnerId']], 
            columns = ['TagId'],
            prefix = '',
            prefix_sep = ''
            ).groupby('PartnerId').agg('sum')
        
        # Drop meaningless feature (0 TagId corresponds to no tag)
        self.one_hot_items.drop('0', axis = 1, inplace = True)
    
        # Create dictionaries (mappings) for tags
        ################## TAG ID - TAG LABEL MAPPING #########################
        tags2names_dict = items_table[['TagId', 'TagName_FR']].to_dict(orient = 'records')
        self.tags_ids2labels = {str(tag['TagId']): tag['TagName_FR'] for tag in tags2names_dict}
        self.tags_labels2ids = {v:k for k,v in self.tags_ids2labels.items()}

        self.item_feature_ids = list(self.one_hot_items.columns)

        self.item_features_dict = self.features_to_dict(self.one_hot_items)

        self.items_ids2tags = {x["PartnerId"]: x["feature_list"] for x in self.item_features_dict}

        if(tf_idf_features): # If true, transform features using IDF weighting
            tf_idf_vec = TfidfTransformer(use_idf = True, smooth_idf = True)
            self.item_features = tf_idf_vec.fit_transform(self.one_hot_items)
    
    def features_to_dict(self, one_hot_items):
        """ Save item features to dictionary for greater ease of use """
        features_json = one_hot_items.to_json(orient = 'index')
        features_json = json.loads(features_json)

        item_features_dict = []
        for partner in features_json.keys():
            partner_tags = []
            for tag in features_json[partner].keys():
                if features_json[partner][tag]:
                    partner_tags.append(tag)
            item_features_dict.append({"PartnerId": partner,
                                "PartnerName": self.items_ids2labels[partner],
                                "feature_list": partner_tags})

        return item_features_dict

    def fit(self):
        """ Fit the data set """
        super().fit(self.user_ids, self.item_ids, None, self.item_feature_ids)

    def build_interactions(self):
        """ Build sparse matrix of interactions """
        interactions, weights = super().build_interactions((
                    (x["MemberId"], x["PartnerId"], x["n_interactions"]) for x in self.interactions_dict
                    if x["n_interactions"]!=0))
        return interactions, weights
    
    def build_item_features(self, data):
        """ Build sparse matrix of item features """
        return super().build_item_features(data)


def is_valid(element, element_list, exception_msg):
    """Check that provided tag is valid: it belongs to tag_list."""
    try:
        if element in element_list:
            pass
        else:
            raise Exception(exception_msg)
    except Exception as e:
        print(e)
        sys.exit(1)

