import numpy as np
import pandas as pd

from scipy.sparse import csc_matrix
import scipy
from sklearn.neighbors import NearestNeighbors

import pickle
from scipy.sparse import save_npz, load_npz
from joblib import dump, load


class RecommendationSys:
    """
    A class used to represent an Item-Based Collaborative Filtering Recommendation System.

    This system is designed to recommend items to users based on the similarity between items,
    which is computed using user-item interaction data. The system uses a Nearest Neighbors
    model with cosine similarity to find similar items.

    Attributes
    ----------
    indic : str
        The name of the column in the input data that indicates the rating or interaction value.
    csr_mat_X : scipy.sparse.csc_matrix
        The sparse matrix representing the user-item interactions.
    user_mapper : dict
        A dictionary mapping user IDs to matrix indices.
    item_mapper : dict
        A dictionary mapping item IDs to matrix indices.
    item_inv_mapper : dict
        A dictionary mapping matrix indices back to item IDs.
    model : sklearn.neighbors.NearestNeighbors
        The nearest neighbors model used to find similar items.
    is_model_data_loaded : bool
        A flag indicating whether the model data has been loaded from disk.
    all_rec_items_sorted_ids_dict : dict[int, list[str]]
        A dictionary storing sorted recommended item IDs for each user's previous Items.
    user_prev_data : set[str]
        A set containing the IDs of items the user has previously interacted with.

    Methods
    -------
    train_model(...) -> None
        Trains the recommendation system using the provided user-item interaction data.

    recommend_items(...) -> ...
        Recommends items to a user based on their previous interactions.

    print_pretty_recommendation(...) -> None
        Prints the recommended items in a user-friendly format.

    __save_model_data() -> None
        Saves the trained model and associated data to disk.

    __load_model_data() -> None
        Loads the model and associated data from disk.

    __create_csr_mat(...) -> None
        Creates a sparse matrix from the user-item interaction data.

    __find_similar_item(...) -> ...
        Finds similar items to a given item using the trained model.

    __combine_and_filter_rec_ids(...) -> ...
        Combines and filters recommended item IDs, removing duplicates.

    __iid_is_found_in_rec(...) -> str
        Checks if a given item ID is found in the user's previous data or in the recommended items.

    __add_new_idd_to_rec(...) -> None
        Adds new recommended items IDs to the final recommendation list.
    """

    def __init__(self, indic: str = 'rate') -> None:
        """
        Initializes the RecommendationSys with a specified indicator column name.

        Parameters
        ----------
        indic : str, optional
            The name of the column in the input data that indicates the rating or interaction value default is 'rate'
        """

        self.__indic: str = indic
        self.__csr_mat_X: scipy.sparse.csc_matrix = None

        self.__user_mapper = dict()
        self.__item_mapper = dict()
        self.__item_inv_mapper = dict()

        self.__model = None
        self.__is_model_data_loaded = False

        self.__all_rec_items_sorted_ids_dict: dict[int, list[str]] = dict()
        self.__user_prev_data: set[str] = set()

    def train_model(self, df: pd.DataFrame) -> None:
        """
        Trains the recommendation system using the provided user-item interaction data.

        This method creates a sparse matrix from the input DataFrame, where each row represents an item,
        each column represents a user, and the values represent interaction ratings. It then trains a
        Nearest Neighbors model using cosine similarity.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing user-item interaction data. It must contain 'uid' for user IDs,
            'iid' for item IDs, and the interaction values in the column specified by `indic`.
        """

        self.__create_csr_mat(df)

        self.__model = NearestNeighbors(
            algorithm='brute',
            metric='cosine'
        )

        if self.__csr_mat_X is not None:
            self.__model.fit(self.__csr_mat_X)

        dump(self.__model, 'model_data/item_recommendations_model.joblib')

    def recommend_items(self,
                        user_prev_data: pd.DataFrame,
                        n_most_important_items: int = 10,
                        n_for_each_item: int = 5,
                        print_results: bool = True):
        """
        Recommends items to a user based on their previous interactions.

        This method identifies the most important items the user has interacted with and finds similar items
        using the trained model. The results are optionally printed and returned as a list of recommended item IDs.

        Parameters
        ----------
        user_prev_data : pd.DataFrame
            A DataFrame containing the user's previous interactions with items. It must contain 'iid' for item IDs
            and the interaction values in the column specified by `indic`.
        n_most_important_items : int, optional
            The number of most important items from the user's previous data to consider for recommendations
             (default is 10).
        n_for_each_item : int, optional
            The number of similar items to find for each important item (default is 5).
        print_results : bool, optional
            Whether to print the recommended items (default is True).

        Returns
        -------
        tuple[list[str], dict[str, list[str]]]
            A tuple containing a list of recommended item IDs and a dictionary mapping each important item ID
            to a list of similar recommended item IDs.
        """

        if user_prev_data.empty:
            print('User Id does not prefer something.')
            return []

        most_important_items = user_prev_data.nlargest(
            n_most_important_items, self.__indic)['iid']

        self.__user_prev_data = set(user_prev_data['iid'].values)
        self.__all_rec_items_sorted_ids_dict = dict()
        item_rec_dict = dict()

        if print_results:
            print(
                f'this userId have {len(most_important_items)} Interested in products.\n')

        for item_idx, item_id in enumerate(most_important_items):

            curr_similar_items_ids = self.__find_similar_item(
                item_id,
                n_neighbors=n_for_each_item
            )
            item_rec_dict[item_id] = curr_similar_items_ids

            if print_results:
                print(f"{item_idx + 1}. Since you prefer productId "
                      f"\"{item_id}\", you might also like this products Ids:")

                for rec_item_idx, recommended_item_id in enumerate(curr_similar_items_ids):
                    found_before = self.__iid_is_found_in_rec(
                        recommended_item_id)
                    print(
                        f'{item_idx + 1}.{(rec_item_idx + 1):>2}. {recommended_item_id}\t{found_before}')

                print()

            self.__add_new_idd_to_rec(curr_similar_items_ids)

        return (
            self.__combine_and_filter_rec_ids(
                self.__all_rec_items_sorted_ids_dict),
            item_rec_dict
        )

    @staticmethod
    def print_pretty_recommendation(items_ids: dict[str, str],
                                    item_rec_dict: dict[str, list[str]]) -> None:
        """
        Prints the recommended items in a user-friendly format.

        This method takes a dictionary of item IDs to item names and a dictionary of recommended item IDs
        and prints them in a structured format.

        Parameters
        ----------
        items_ids : dict[str, str]
            A dictionary mapping item IDs to item names.
        item_rec_dict : dict[str, list[str]]
            A dictionary mapping important item IDs to lists of similar recommended item IDs.
        """

        print("Recommendation:")
        for pref_item, res_rec_items in item_rec_dict.items():

            print(f"Once you Prefer: \"{items_ids[pref_item]}\" you may prefer:")

            for i, rec_item in enumerate(res_rec_items):
                print(f"  {i + 1}. {items_ids[rec_item]}")
            print()

    def __save_model_data(self) -> None:
        """
        Saves the trained model and associated data to disk.

        This method serializes the item mappings, the sparse matrix, and the trained Nearest Neighbors model
        to disk for later use.
        """

        try:

            with open("model_data/item_mapper.pkl", 'wb') as file:
                pickle.dump(self.__item_mapper, file)

            with open("model_data/item_inv_mapper.pkl", 'wb') as file:
                pickle.dump(self.__item_inv_mapper, file)

            save_npz('model_data/csr_mat_X.npz', self.__csr_mat_X)

        except Exception as e:
            print(e.args)

    def __load_model_data(self) -> None:
        """
        Loads the model and associated data from disk.

        This method deserializes the item mappings, the sparse matrix, and the trained Nearest Neighbors model
        from disk. This allows the system to make recommendations without retraining the model.
        """

        try:
            with open("rec_sys\\model_data\\item_mapper.pkl", 'rb') as file:
                self.__item_mapper = pickle.load(file)

            with open("rec_sys\\model_data/item_inv_mapper.pkl", 'rb') as file:
                self.__item_inv_mapper = pickle.load(file)

            self.__csr_mat_X = load_npz('rec_sys\\model_data/csr_mat_X.npz')

            self.__model = load('rec_sys\\model_data/item_recommendations_model.joblib')
            self.__is_model_data_loaded = True

        except Exception as e:
            print(e)

    def __create_csr_mat(self, df: pd.DataFrame) -> None:
        """
        Creates a sparse matrix from the user-item interaction data.

        This method transforms the input DataFrame into a sparse matrix, mapping users and items to matrix indices.
        The sparse matrix is then used to train the recommendation model.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing user-item interaction data. It must contain 'uid' for user IDs,
            'iid' for item IDs, and the interaction values in the column specified by `indic`.
        """

        unique_users = np.unique(df['uid'])
        unique_items = np.unique(df['iid'])

        N = len(unique_users)  # number if unique users
        M = len(unique_items)  # number if unique items

        # making list of range to the number of the users
        list_N = np.arange(N)
        list_M = np.arange(M)  # making same but for items

        # making dict hold a key of the user or item id and its "Index".
        self.__user_mapper = dict(zip(unique_users, list_N))
        self.__item_mapper = dict(zip(unique_items, list_M))

        # making the dict but to get the item id from its index
        self.__item_inv_mapper = dict(zip(list_M, unique_items))

        # get all the indexes corresponding to user or item ids
        # from the original user-item Interaction.
        user_index = [self.__user_mapper[i] for i in df['uid']]
        item_index = [self.__item_mapper[i] for i in df['iid']]

        # making sparse matrix with items-id in rows and the users-id in columns
        # and the value is the indicator from user-item interaction data.
        self.__csr_mat_X = csc_matrix(
            (
                df[self.__indic],
                (
                    item_index,
                    user_index
                )
            ),
            shape=(M, N)
        )

        self.__save_model_data()

    def __find_similar_item(self, iid: str, n_neighbors: int, show_distance: bool = False):
        """
        Finds similar items to a given item using the trained model.

        This method takes an item ID and finds the top N similar items based on the cosine similarity measure.
        It can optionally return the distances to the similar items.

        Parameters
        ----------
        iid : str
          The item ID for which similar items are to be found.
        n_neighbors : int
          The number of similar items to return.
        show_distance : bool, optional
          Whether to return the distances along with the similar item IDs (default is False).

        Returns
        -------
        list[str]
          A list of similar item IDs.
        """

        # load the model if not loaded.
        if not self.__is_model_data_loaded:
            self.__load_model_data()

        # get all users interaction with this item.
        item_index = self.__item_mapper[iid]
        item_vector = self.__csr_mat_X[item_index]  # type: ignore
        item_vector = item_vector.reshape(1, -1)

        n_neighbors = min(n_neighbors, len(self.__item_mapper)) + 1

        neighbors = self.__model.kneighbors(  # type: ignore
            item_vector,
            n_neighbors=n_neighbors,
            return_distance=show_distance
        )

        neighbors_ids = list()
        for i in range(1, n_neighbors):
            n = neighbors.item(i)  # type: ignore

            neighbors_ids.append(
                self.__item_inv_mapper[n]
            )

        return neighbors_ids

    @staticmethod
    def __combine_and_filter_rec_ids(all_rec_items_sorted_ids_dict: dict[int, list[str]]):
        """
        Combines and filters recommended item IDs, removing duplicates.

        This method processes the dictionary of recommended items, ensuring that each item appears only once
        in the final recommendation list.

        Parameters
        ----------
        all_rec_items_sorted_ids_dict : dict[int, list[str]]
            A dictionary storing sorted recommended item IDs for each user.

        Returns
        -------
        list[str]
            A list of unique recommended item IDs.
        """

        res_list = []
        sorted_idx_rate = list(all_rec_items_sorted_ids_dict.keys())
        for idx in sorted_idx_rate:

            rec_items = all_rec_items_sorted_ids_dict[idx]

            for rec_item in rec_items:
                if rec_item not in res_list:
                    res_list.append(rec_item)
        return res_list

    def __iid_is_found_in_rec(self, iid: str):
        """
        Checks if a given item ID is found in the user's previous data or in the current recommended items.

        This method checks whether the item ID has been previously interacted with by the user or if it
        is already present in the current recommendations.

        Parameters
        ----------
        iid : str
            The item ID to check.

        Returns
        -------
        str
            A string indicating where the item ID was found, if at all.
        """

        if iid in self.__user_prev_data:
            return 'Found: User Prev Data'

        for iid_s in self.__all_rec_items_sorted_ids_dict.values():
            if iid in iid_s:
                return 'Found: RecSys Res'

        return ""

    def __add_new_idd_to_rec(self, curr_similar_items_ids: list[str]):
        """
        Adds new recommended item IDs to the recommendation list.

        This method updates the dictionary of recommended items by adding new IDs that have been found
        as similar to the user's previously interacted items.

        Parameters
        ----------
        curr_similar_items_ids : list[str]
        A list of newly recommended item IDs.
        """

        for rec_item_idx, recommended_item_id in enumerate(curr_similar_items_ids):

            if recommended_item_id not in self.__user_prev_data:
                if self.__all_rec_items_sorted_ids_dict.get(rec_item_idx, None) is None:

                    self.__all_rec_items_sorted_ids_dict[rec_item_idx] = [
                        recommended_item_id]
                else:
                    if recommended_item_id not in self.__all_rec_items_sorted_ids_dict[rec_item_idx]:
                        self.__all_rec_items_sorted_ids_dict[rec_item_idx] \
                            .append(recommended_item_id)
