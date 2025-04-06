# IMPORTS 
# ===========================================================================================
import warnings
warnings.filterwarnings("ignore")

import os
import itertools 
import argparse
import pickle
import json
from tqdm import tqdm
from datetime import date
from dotenv import load_dotenv

import src.recsys as recsys

from data.process_data import process_data
from src.user_item_dataset import UserItemData
from conf.core import config, SAVED_MODEL_DIR, USER_ITEM_DATASET_PATH, TRAINING_EXP_DIR

from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score
# ===========================================================================================


def find_best_params(X_train, X_test, w_train, item_features, verbose = False):
    """Find the best parameters for the LightFM model using a gridsearch method. 

    Args:
        X_train (_type_): training interactions, obtained by splitting the interactions matrix (n_users x n_items)
        X_test (_type_): test interactions
        w_train (_type_): weights of the training interactions
        epochs (int, optional): number of epochs to train the model for. Defaults to 5.
        verbose (bool, optional): level of verbosity required for parameter optimization. Defaults to False.
    
    Returns:
        best_params (tuple): tuple of the parameters yielding the best validation score for the LightFM model: 
        (nb of components for the LightFM model, loss function type, 
        parameter alpha, learning schedule, random state, number of epochs)
    """
    param_dict = config.param_grid.dict()
    param_combs = list(itertools.product(*(param_dict[name] for name in param_dict.keys())))
    param_combs = [
                    {'no_components': comb[0],
                    'loss':comb[1],
                    'item_alpha':comb[2],
                    'learning_schedule':comb[3],
                    'random_state':comb[4],
                    'epochs': comb[5]} for comb in param_combs]

    best_train_auc, best_test_auc = 0, 0
    print("\n")
    print(f"Optimizing model parameters: trying out {len(param_combs)} parameter combinations from param_grid...")
    print("\n")

    with tqdm(param_combs, position = 0, leave = True) as progress_bar:
        for param_comb in param_combs:
            hybrid_model = LightFM(no_components = param_comb['no_components'],
                                    loss = param_comb['loss'],
                                    item_alpha = param_comb['item_alpha'],
                                    learning_schedule = param_comb['learning_schedule'],
                                    random_state = param_comb['random_state'])

            hybrid_model.fit(X_train,
                            sample_weight = w_train,
                            epochs = param_comb['epochs'],
                            item_features = item_features,
                            num_threads=2,
                            verbose=False)

            train_auc = auc_score(hybrid_model, 
                                X_train,
                                item_features = item_features).mean()

            test_auc = auc_score(hybrid_model, 
                                X_test,
                                train_interactions = X_train,
                                item_features = item_features,
                                check_intersections=True).mean()
            
            if verbose: 
                print("\n")
                print("Param Comb: ")
                print(param_comb)
                print("Train Auc : {:.2f}, Test AUC: {:.2f}".format(train_auc, test_auc))
                print("\n")
                
            if test_auc>best_test_auc:
                best_params = param_comb
                best_train_auc, best_test_auc = train_auc, test_auc
            
            progress_bar.update()

    if verbose:
        print("\n")
        print("Best param Comb: ", best_params)
        print("\n")
    
    return(best_params)

def save_experiment(config, train_auc, test_auc, params):
    """Save experiment details into json for tracking. 

    Args:
        config (ExperimentConfig): configuration loaded from config.yml
        train_auc (float): train score
        test_auc (float): test score
        params (dict): model parameters
    """

    exp_file = config.experiment_name + '.json'
    exp_json = {"Experiment": config.experiment_name,
                "date": str(date.today()),
                "Data set": os.getenv('MYSQL_DB'),
                "Model type": config.model_type,
                "Test size": str(config.test_size),
                "Parameter optimization": config.param_opt,
                "Train AUC": str(train_auc),
                "Test AUC": str(test_auc)}
    if config.param_opt:
        exp_json["Best parameters"] = str(params)
        exp_json["Param Grid"] = str(config.param_grid)
    else:
        exp_json["Parameters"] = str(params)


    with open(TRAINING_EXP_DIR / exp_file, "w") as f:
        json.dump(exp_json, f, indent = 4)
    
    f.close()


def main(mysql_creds, verbose = 1):
    """Run the parameter optimization and the training phase for a LightFM recommender system

    Args:
        mysql_creds (dict): Credentials to connect to a data base
        verbose (int, optional): Level of verbosity required for the training phase. Defaults to 1.
        If set to 0, minimal information will be given about training. If set to any value >1, maximal details
        will be given.     
    """
    experiences2partners, tagged_partners = process_data(mysql_creds["mysql_host"], 
                                                        mysql_creds["mysql_user"], 
                                                        mysql_creds["mysql_passwd"], 
                                                        mysql_creds["mysql_db"],
                                                        verbose = False)

    # Create interactions matrix
    # Map users and items to ids
    dataset = UserItemData(experiences2partners, 
                        item_features = tagged_partners,
                        tf_idf_features= False)
    
    with open(USER_ITEM_DATASET_PATH, 'wb') as file:
        pickle.dump(dataset, file)
        print('Dataset saved in {}'.format(USER_ITEM_DATASET_PATH))
    
    # If the model type is hybrid, use item features
    use_item_features = dataset.item_features if config.model_type =='hybrid' else None

    # Split the data set into train and test sets
    X_train, X_test = random_train_test_split(dataset.interactions, 
                                            test_percentage=config.test_size, 
                                            random_state=42)

    w_train, _ = random_train_test_split(dataset.weights, 
                                            test_percentage=config.test_size, 
                                            random_state=42)

    # Perform parameter optimization if required
    param_opt_verbose = True if verbose >1 else False
    if config.param_opt: 
        param_config = find_best_params(X_train, 
                                        X_test, 
                                        w_train, 
                                        item_features = use_item_features, 
                                        verbose = param_opt_verbose)
    else:
        param_config = config.model_config.dict()

    # Define final model and train it
    model = LightFM(no_components = param_config['no_components'],
                    loss = param_config['loss'],
                    item_alpha = param_config['item_alpha'],
                    learning_schedule = param_config['learning_schedule'],
                    random_state = param_config['random_state'])

    print("\n")
    print("Training model ...")
    print("\n")

    model.fit(X_train, 
            sample_weight = w_train,
            item_features = use_item_features,
            epochs = param_config['epochs'],
            num_threads = 2,
            verbose = verbose)

    # Evaluate the model on train and test
    train_auc = auc_score(model, 
                            X_train,
                            item_features=use_item_features).mean()

    test_auc = auc_score(model, 
                        X_test,
                        train_interactions = X_train,
                        item_features = use_item_features,
                        check_intersections=True).mean()

    if verbose:
        print("\n")
        print("Train AUC : {:.2f}, Test AUC: {:.2f}".format(train_auc, test_auc))
        print("\n")
    
    # Save model and training configuration
    save_to = config.model_config.model_file_name
    if save_to is None:
        today = date.today()
        save_to =  f"{today}_model.pkl"

    recsys.save_model(model, SAVED_MODEL_DIR / save_to)

    save_experiment(config, train_auc, test_auc, param_config)



def make_parser():
    """ Create parser to process command line arguments for training the recommender system """

    parser = argparse.ArgumentParser(description= 'A hybrid recommender system')
    
    parser.add_argument("-v",
                        "--verbose",
                        type = int, 
                        required = False,
                        default = 1,
                        help = "OPTIONAL - Level of verbosity for the training phase.\
                            For minimal details, set verbose = 0. For maximal details, \
                            set verbose to any integer strictly greater than 1. Default verbosity is 1.")

    return parser

if __name__=='__main__':

    # Load environment variables
    load_dotenv()

    mysql_creds = {
    "mysql_host" : os.getenv('MYSQL_HOST'),
    "mysql_user" : os.getenv('MYSQL_USER'),
    "mysql_passwd" : os.getenv('MYSQL_PASSWD'),
    "mysql_db" : os.getenv('MYSQL_DB')
    }
    
    parser = make_parser()
    args = parser.parse_args()

    # Run main to train the model
    main(mysql_creds,
         verbose = args.verbose)
