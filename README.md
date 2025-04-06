# Recommendation system #

* This repository contains the code for a recommendation system aimed at recommending venues (hotels, restaurants, shops...) to users. 
* The model was trained on a (private) data base of around 1000 users (called "Members") and 300 venues (called "Partners"). The data set contained information about each partner (tags that characterize the partner - "hotel", "5-star", "swimming pool"), and interaction history between the members and the partners. 
* The model is based on collaborative filtering and content, and makes use of the LightFM package. 
* Two recommender models are available : 
    - a hybrid model (based on content and collaborative filtering)
    - a pure collaborative filtering model


## Project Structure ##
```
.
├── conf
│   ├── core.py
├── data
├── recommendations
├── experiments
│   ├── test_experiment.json
├── saved_models
├── src
│   ├── recsys.py
│   ├── user_item_dataset.py
├── recommender.py
├── train.py
├── config.yml
├── .env
├── requirements.txt
└── README.md

```

## Recommender System ##

### About the model ###

Two models were trained: 

- a ***collaborative filtering based model*** : partners are recommended to a member if they were visited by similar members. 

- a ***hybrid model*** based on ***content-based*** and ***collaborative filtering*** : partners are recommended to a member if they were visited by similar members and if they are similar to partners previously visited by the member. 

The models were trained on a processed data set extracted from the data base.

The pre-processing phase included : 

- removal of "generic" partners or partners tagged as 'train' or 'avion' (such as 'La Poste', 'AirBnb', 'Ticketmaster', 'Eurostar',
'booking.com', 'Vueling', 'Uber', 'EasyJet', 'Transavia', 'SNCF', 'Air France'... )

- removal of all experiences which were not associated to a partner (and consequently, of members who only had experiences which were not associated to any parter). For such members, the recommendation system will not be able to make a recommendation. 


### How the model is built

Our model makes recommendations from **implicit feedback**. Indeed, in the data set, we only have the number of interactions of each user with each item. We therefore assume that the number of times a user interacted with an item is an indication of the users' likes/dislikes. 

The model learns a ranking of the items, so all items can still be recommended to a user, even if the user has never interacted with the items. 

In our setting, provided a list of tags, a ranking of all items tagged as such will be given as output by the system (items which don't have the right tags will not be displayed). For example, a recommendation of partner for the tags "familial" and "restaurant" will give a ranking of all partners tagged with at least both tags "familial" and "restaurant". 

The recommendation system is built as follows : it learns a vector representation of items (partners) and users (members) which encodes the preferences of users over the items. Learning this vector representation is done so that users and items can be represented in the same latent space and thus be "compared". Those vector representations are called **embeddings**. 

In our data set : 

- a user (member) is represented by a vector *X* of length *n_items* (number of partners) with each coordinate being the number of times the user visited the partner. 

- an item (partner) is represented by a vector *Y* of length *n_tags* with each coordinate a 1 if the item was tagged as such or 0 if not. 

We optimize the model to find new representations of *X* and *Y*, called *U* and *V* (of same smaller dimensionality *no_components*) such that the product *U.V* will be maximal if user *U* and item *V* are "compatible", and that we want to recommend the item *V* to user *U*.

The embeddings *U* and *V* are based on both the interactions between users and items, and on the item features (the tags for the partners). Thus, a user will be recommended partners similar to partners he has previously interacted with, but also partners that users with a similar interaction profile visited. 

Since our model takes into account item features (i.e. partner tags), a new partner can be recommended by the system (even if no member has had an interaction with it). 
However, our model does not take into account any user features (characteristics of the members), so for a new member to be given a recommendation, the training data set must contain at least one interaction for the given member. 

More details about how the system works can be found in the documentation here : https://making.lyst.com/lightfm/docs/index.html 

### Future work

Improvements could be made to the model in the following ways: 

- Providing user features (i.e. member characteristics), in order to give recommendations to new users with no previous interactions, based on similar users. Up to now, we only provide item features to the model (i.e. partner tags), which enables the model to recommend new items. 

- Developping a tailored recommender system based on operational requirements: this system is based off the *LightFM* Python package for simplicity of use. However, in the future, a custom recommender system could be developed according to business needs (custom weighting content/collaborative filtering, custom ranking function...)

- Partner tags: in our work we noticed item tags could sometimes lead to wrong recommendations by the model. Tagging the items more accurately could help in making relevant item recommendations to users. 

- Adding more interactions data could be useful to improve the robustness of the model. 


## Code Set Up ##

### Prerequisites ###

* Python version : 3.7.13
* Virtual environment
* Libraries to install : requirements.txt

### Installations ###

* Download Python (version 3.7.13): https://www.python.org/downloads/ 

* Install virtualenv using  pip (for better package management) :
    - Install virtualenv : ```pip install virtualenv```
    - Create a virtual environment with desired name : ```virtualenv {virtualenv_name}```
    - Activate the virtual environment : ```source virtualenv {virtualenv_name}``` (OS/Linux) or ```{virtualenv_name}\Scripts\activate``` (Windows)

* Navigate to the project directory : *RecSys/*
* Install the libraries required to run the recommender system : ```pip install -r requirements.txt```

## Running the recommendation system ##

### Input to the model ###

The recommendation system can take in :

- a member id (only one member id is supported at the moment)

- a list of tag ids (tag_ids) or tag names (tag_names): ***only partners matching all of the tags can be recommended by the system***

- a number of recommendations to output (n_reco)

- a model to choose from - hybrid or pure collaborative filtering model (model_name)

- a file name where the recommendation will be saved as a json, in the *recommendations/* folder. 

### Output of the model

The recommender system saves the recommendation to a json file if the flag ```--file``` is used. 
If not, the recommendation is simply displayed in the shell. 

The json has the following fields: 

- "user" (the id of the member)

- "tags" (the tags provided for the recommendation request)

- "recommended" (the partners recommended by the system), 

- "positives" (the partners already visited by the member)

- "common" (the partners already visited by the member and found by the recommendation system)

Below is the structure of the recommendation json:

```
{
    "user": member_id,
    "tags": "tag1, tag2, ..., tagN" ,
    "recommended": [# list of recommended partners
        {"PartnerId": partner_id1,
        "PartnerName": partner_name1}
        ...
        {"PartnerId": partner_id_p,
        "PartnerName": partner_name_p}
    ],
    "positives": [# list of partners visited by the member tagged according to tags
        {"PartnerId": partner_id_k,
        "PartnerName": partner_name_k}
        ...
        {"PartnerId": partner_id_j,
        "PartnerName": partner_name_j}
    ],
    "common": [# list of partners visited by the member and recommended by the system
        {"PartnerId": partner_id_k,
        "PartnerName": partner_name_k}
        {"PartnerId": partner_id_j,
        "PartnerName": partner_name_j}
    ]
}
```

### How To
In the command line, within the virtual environment:

- Navigate to the *RecSys/* folder

- Run the following, with the desired argument values (an example is provided further down): 

```python recommender.py --member {member_id} --tag_ids {tag_id1} {tag_id2} ... {tag_idN} --tag_names {tag_name1} {tag_name2} ... {tag_nameN} --n_reco {n_reco} --recsys {model_name} --file {file_name}```

OR (equivalent)

```python recommender.py -m {member_id} --tag_ids {tag_id1} {tag_id2} ... {tag_idN} -n {n_reco} -r {model_name} -f {file_name}```


### Arguments description

 ``` -h, --help  ``` Show help for this function

  ``` -m MEMBER, --member MEMBER  ```
                        ID of the member to whom we wish to make
                        recommendations

``` --tag_ids TAG_IDS [TAG_IDS ...]  ```
                        List of partner tag ids to select the partner on. 
                        For example, for (Piscine Spa Marrakech) , provide the following : 499 695 146

``` --tag_names TAG_NAMES [TAG_NAMES ...]  ```
                        OPTIONAL - List of criteria (partner tag labels) to select the partner on. 
                        If tag ids are provided, this argument is ignored. Example : Piscine Spa Marrakech

``` -n N_RECO, --n_reco N_RECO  ```
                        OPTIONAL - Number of recommendations to display. Default is 7

``` -r {model_file_name}, --recsys {model_file_name}  ```
                        OPTIONAL - Choice of recommender system: provide the name of the model as saved in the saved_models/ directory. For example: LightFMHybrid.pkl. **IMPORTANT : The default model used is the latest saved model in the *saved_models/* directory.** 

``` -f [FILE [FILE ...]], --file [FILE [FILE ...]]  ```
                        OPTIONAL - Path to save the recommendation as json. 
                        If the flag -f is provided without any file name, a default file name will be set depending on the member id and the tags provided. 
                        If file name is provided, recommendation will be saved as 
                        *{filename}.json*. 
                        If the flag -f is not used in the command, the recommendation will be printed 
                        in the output in the command line. 

### ***Example : Get 5 recommendations for the member 0 of partners that match "Marrakech", "Piscine", and "Spa", and save it in a file called test.json*** ###

The following lines are equivalent: 

```python recommender.py --member 0 --tag_names Marrakech Piscine Spa --n_reco 5 --file test```

```python recommender.py --member 0 --tag_ids 146 499 695 --n_reco 5 --file test```

To print output directly in the command line, simply remove the --file flag: 

```python recommender.py --member 0 --tag_ids 146 499 695 --n_reco 5```

NB : If a tag name contains several words, such as "Hôtel de charme", it should be provided within double quotes. 
Example : 
```python recommender.py --member 0 --tag_names Marrakech Piscine "Hôtel de charme" --n_reco 5 --file test```

## Training the recommendation system ##

### Preliminary : connect to the data base

The script retrieves the data using a Python connector to MySQL Workbench, where the data base should be stored. 
**Provide the necessary SQL credentials in the .env file.**

### How To
In the command line, within the virtual environment:

- Navigate to the *RecSys/* folder

- **Supply the required values** in the *config.yml* file (see explanations below)

- Run the following, with the desired level of verbosity: 

```python train.py --verbose {verbosity_level}```

OR (equivalent): 

```python train.py -v {verbosity_level}```

### Arguments description

 ``` -h, --help  ``` Show help for this function 

``` -v VERBOSITY_LEVEL, --verbose VERBOSITY_LEVEL  ```
                        OPTIONAL - Level of verbosity for the training phase.
                        For minimal details, set verbose to 0. For maximal details, 
                        set verbose to any integer strictly greater than 1. Default verbosity is 1.

### ***Example : Train*** ###

To launch training, the following lines are equivalent: 

```python train.py --verbose 1```

```python train.py -v 1 ```

```python train.py```

### Supplying parameters in config.yml

The file *config.yml* sets up the parameter configuration for the model. Parameters should be supplied every time a new training is launched. 

This configuration can be saved in the *experiments/* folder (see *Tracking training experiments* section below). 

- *experiment_file_name* : name of the experiment to save the configuration and the results of training as a json in the *experiments/* folder. 

- *model_type* :  choice of the type of recommender system: 
                'hybrid' for a system based on both collaborative filtering and content
                or 'cf' for a collaborative filtering system only.

- *test_size*: proportion of samples to keep aside for evaluating the model (they will not be used for training the model). 

- *model_file_name* : file name to save the trained model (ex : *hybrid_model.pkl*).
                    If not supplied, the model will be saved as *{todays_date}_model.pkl*

- *param_opt* : boolean, whether or not to run **parameter optimization** to maximize performance of the model. This is **recommended** if a **new data set is supplied** to the model (in this case, set to *True*). 
If set to *True*, supply various values of parameters to try in *param_grid*. The *model_config* right below will be ignored. If set to *False*, no parameter optimization will be carried out, and the parameters supplied in *model_config* below will be used to configure the model. 

- *model_config* (choice of model parameters, to supply if *param_opt* is *False*): 
    - *no_components*: dimensionality of the latent embeddings for users and items. 
    - *loss*: loss function optimized during training. One of (‘logistic’, ‘bpr’, ‘warp’, ‘warp-kos’). 
    - *item_alpha*: L2 penalty on item features. Setting this number too high will slow down training.
    - *learning_schedule*: optimization strategy used for gradient descent. One of (‘adagrad’, ‘adadelta’).
    - *random_state*: random seed set to get reproducible results. 
    - *epochs*: number of passes over the full training set. If the size of the data set gets significantly larger, model performance will benefit from training for more epochs. 

More details can be found in the documentation here : https://making.lyst.com/lightfm/docs/lightfm.html 

### Tracking training experiments

Every time the training script is run, the training configuration and results are saved in a .*json* file in the *experiments/* directory. The experiment name can be set up in the *config.yml* file.  