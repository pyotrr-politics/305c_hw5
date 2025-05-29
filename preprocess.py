import pandas as pd

torch.manual_seed(305)
device = 'cuda' if torch.cuda.is_available() else 'cpu'



### Import data
votes = pd.read_csv("https://raw.githubusercontent.com/slinderman/stats305b/winter2025/assignments/hw2/votes.csv")
votes.fips = votes.fips.apply(lambda x: str(int(x)).zfill(5))

demo = pd.read_csv("https://raw.githubusercontent.com/slinderman/stats305b/winter2025/assignments/hw2/demographics.csv")
demo.fips = demo.fips.apply(lambda x: str(int(x)).zfill(5))





### helper function
def preprocess_covariates(demo):
    """
    Preprocess the demographic data to extract the covariates for the regression,
    as described above.
    """
    # Sort the demographic data by FIPS code
    demo = demo.sort_values("fips")

    # Extract the scalar features and standardize them
    scalar_feat_names = ["white_pct",
                        "black_pct",
                        "hispanic_pct",
                        "foreignborn_pct",
                        "female_pct",
                        "age29andunder_pct",
                        "age65andolder_pct",
                        "median_hh_inc",
                        "clf_unemploy_pct",
                        "lesshs_pct",
                        "lesscollege_pct",
                        "lesshs_whites_pct",
                        "lesscollege_whites_pct",
                        "rural_pct"]

    X_scalar = demo[scalar_feat_names]
    X_scalar -= X_scalar.mean(axis=0)
    X_scalar /= X_scalar.std(axis=0)

    # Extract the categorical features and one-hot encode them
    cat_feat_names = ["ruralurban_cc"]
    X_cat = demo[cat_feat_names]
    X_cat = pd.get_dummies(X_cat, columns=cat_feat_names, drop_first=True)

    # Combine the features and add a bias term
    X = pd.concat([X_scalar, X_cat, demo["fips"]], axis=1)
    return X

covariates = preprocess_covariates(demo)


def extract_responses(votes, year):
    """
    Extract the responses for a given year from the votes dataframe.
    """
    # Extract the democratic and republican votes for the given year
    dem_votes = votes[(votes["party"] == "DEMOCRAT") & (votes["year"].isin(year))]
    rep_votes = votes[(votes["party"] == "REPUBLICAN") & (votes["year"].isin(year))]

    # Sort by fips codes
    dem_votes = dem_votes.sort_values("fips")
    rep_votes = rep_votes.sort_values("fips")
    assert (rep_votes.fips.values == dem_votes.fips.values).all()

    # Extract the states and code them as integers
    state_names = sorted(dem_votes.state.unique())
    s = pd.Categorical(dem_votes.state, categories=state_names).codes

    # Make a dataframe of responses
    responses = pd.DataFrame({
        "fips": dem_votes.fips.values,
        "state": dem_votes.state.values,
        "s": s,
        "share": dem_votes.candidatevotes.values / \
            (dem_votes.candidatevotes.values + rep_votes.candidatevotes.values),
        "year": dem_votes.year.values
        # "N": dem_votes.totalvotes.values
        })
    return responses, state_names


state_order = ["INDIANA", 
               "KENTUCKY", 
               "VERMONT", 
               "WEST VIRGINIA", 
               "ALABAMA", 
               "MISSISSIPPI",
               "OKLAHOMA", 
               "TENNESSEE", 
               "CONNECTICUT", 
               "MARYLAND", 
               "MASSACHUSETTS", 
               "RHODE ISLAND", 
               "FLORIDA", 
               "SOUTH CAROLINA", 
               "ARKANSAS", 
               "NEW JERSEY", 
               "DELAWARE", 
               "ILLINOIS", 
               "LOUISIANA", 
               "NEBRASKA", 
               "NORTH DAKOTA", 
               "SOUTH DAKOTA", 
               "WYOMING", 
               "NEW YORK", 
               "OHIO", 
               "TEXAS", 
               "MISSOURI", 
               "MONTANA", 
               "UTAH", 
               "COLORADO", 
               "DISTRICT OF COLUMBIA", 
               "KANSAS", 
               "IOWA", 
               "MAINE", 
               "IDAHO", 
               "CALIFORNIA", 
               "WASHINGTON", 
               "NORTH CAROLINA", 
               "OREGON", 
               "NEW MEXICO", 
               "VIRGINIA", 
               "HAWAII", 
               "GEORGIA", 
               "NEW HAMPSHIRE", 
               "PENNSYLVANIA", 
               "MINNESOTA", 
               "WISCONSIN", 
               "MICHIGAN", 
               "NEVADA", 
               "ARIZONA"]


responses, state_names = extract_responses(votes, [2012, 2016, 2020])

target_states = [state_names.index(s) for s in state_order[25:]]
target_rows = (responses["s"].isin(target_states)) & (responses["year"] == 2020)

responses = pd.get_dummies(responses, columns=["year", "s"], drop_first=True)
responses_test = responses[target_rows].reset_index(drop=True)
responses_train = responses[~target_rows].reset_index(drop=True)

target_rows = [covariates[covariates["fips"] == fips].index.item() \
               for fips in responses_train["fips"]]
covariates_train = pd.concat([covariates.loc[target_rows].reset_index(drop=True),
                              responses_train.iloc[:, 3:]], axis=1)

target_rows = [covariates[covariates["fips"] == fips].index.item() \
               for fips in responses_test["fips"]]
covariates_test = pd.concat([covariates.loc[target_rows].reset_index(drop=True),
                              responses_test.iloc[:, 3:]], axis=1)
