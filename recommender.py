import os
import sys
import glob
import warnings
import numpy as np
warnings.filterwarnings("ignore")

import argparse
import pickle

import src.recsys as recsys
from src.user_item_dataset import UserItemData, is_valid

from conf.core import PACKAGE_ROOT, SAVED_MODEL_DIR, USER_ITEM_DATASET_PATH, RECO_DIR

def make_parser():
    """ Create parser to process command line arguments for the recommender system """
    
    parser = argparse.ArgumentParser(description= 'A hybrid recommender system')

    parser.add_argument("-m",
                        "--member", 
                        type = str,
                        required = True,
                        help = "ID of the member to whom we wish to make recommendations")
    
    parser.add_argument("--tag_ids",
                        nargs = "+",
                        type = str,
                        required = False,
                        default = None,
                        help = "List of partner tag ids to select the partner on. \
                        For example, for (Piscine Spa Marrakech) , provide the following : 499 695 146")
    
    parser.add_argument("--tag_names",
                        nargs = "+",
                        type = str,
                        required = False,
                        default = None,
                        help = "OPTIONAL - List of criteria (partner tag labels) to select the partner on. \
                        If tag_ids is provided, this argument is ignored. Example : Piscine Spa Marrakech")
    
    parser.add_argument("-n",
                        "--n_reco",
                        type = int, 
                        required = False,
                        default = 7,
                        help = "OPTIONAL - Number of recommendations to display. Default is 7")

    parser.add_argument("-r",
                        "--recsys", 
                        type = str, 
                        required = False, 
                        default = None, 
                        help = "OPTIONAL - Choice of recommender system: \
                                Provide the name of the model as saved in the saved_models/ directory. \
                                For example: LightFMHybrid.pkl\
                                Default is the latest saved models in the saved_models/ directory.")
    
    parser.add_argument("-f",
                        "--file",
                        nargs = '*',
                        type = str, 
                        required = False,
                        default = None,
                        help = "OPTIONAL - Path to save the recommendation as json. \
                        If nothing is provided (-f), default file name is \
                        *reco_member_{member_id}_tags_{tag_ids}.json*. \
                        If file name is provided, recommendation will be saved as \
                        *{filename}.json*. \
                        If -f is not entered in the command, the recommendation will be printed \
                        in the output in the command line.")

    return parser

def check_tags(dataset, tag_ids = None, tag_names = None):
    """ Check that provided tags are valid (e.g. that they belong to the user-item 
    dataset tags). Tags can be provided as ids or labels. """

    try:
        if (tag_ids is not None and len(tag_ids)>0) or (tag_names is not None and len(tag_names)>0):
            pass
        else:
            raise Exception("Please provide tag ids (--tag_ids) or tag names (--tag_names).")
    except Exception as e:
        print(e)
        sys.exit(1)
    
    if tag_ids is not None:
        if len(tag_ids)>0:
            for tag_id in tag_ids:
                is_valid(tag_id, 
                dataset.item_feature_ids,
                exception_msg = "Non-valid tag provided: {}. Please provide a valid partner tag.".format(
                    tag_id))

    else:
        if tag_names is not None:
            if len(tag_names)>0:
                tag_labels = list(dataset.tags_labels2ids.keys())
                for tag_name in tag_names:
                    is_valid(tag_name, 
                            tag_labels,
                            exception_msg = "Non-valid tag provided: {}. Please provide a valid partner tag.".format(
                                            tag_name))


def main():
    parser = make_parser()

    args = parser.parse_args()

    # Parse command line arguments
    member_id = args.member
    tag_ids = args.tag_ids
    tag_names = args.tag_names
    top_k = args.n_reco
    model_name = args.recsys
    file_name = args.file


    # Load data set
    with open(USER_ITEM_DATASET_PATH, 'rb') as file:
        dataset = pickle.load(file) 
    
    # Check if the arguments provided are valid

    # Valid member ID
    is_valid(member_id, 
            dataset.user_ids,
            exception_msg = "Non-valid member ID provided: {}. Please provide a valid member ID.".format(
                member_id))
    # Valid tags
    check_tags(dataset, tag_ids, tag_names)


    # Retrieve model to make recommendations
    if model_name is not None:
        model_name = SAVED_MODEL_DIR / model_name
    else:
        model_list = glob.glob(os.path.join(SAVED_MODEL_DIR, '*'))
        try:
            # Pick latest model saved in the saved_models/ directory
            model_name = max(model_list, key = os.path.getctime)
        except ValueError as err:
            print("There is no saved recommender system in the 'saved_models/' directory. \
                Please train a model before using the recommender system.")
            sys.exit(1)

    try: 
        model = recsys.load_model(model_name)
    except FileNotFoundError as err:
        print("Model file name provided does not exist in the saved_models/ directory. Please make sure you provide an existing model file name.")
        sys.exit(1)

    # Make recommendation

    reco = recsys.recommend(model, 
                     member_id,
                     dataset, 
                     tag_ids,
                     tag_names, 
                     top_k = top_k)

    # Save recommendation to json or output in shell
    if file_name is not None:
        if len(file_name)>0:
            file_name = args.file[0]
            _, tail = os.path.splitext(file_name)
            if not tail:
                file_name = file_name + '.json'
                file_name = RECO_DIR / file_name 
        else:
            file_name = "reco_member_{}_tags_{}.json".format(member_id, tag_names) if tag_names is not None \
            else "reco_member_{}_tags_{}.json".format(member_id, tag_ids)
            file_name = RECO_DIR / file_name
        recsys.save_reco(reco, file_name)
    else:
        recsys.display_reco(reco)


if __name__ == '__main__':

    main()