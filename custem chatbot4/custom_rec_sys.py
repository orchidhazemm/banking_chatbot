from pandas import DataFrame
from rec_sys.recommendation_sys import *


class CustomRecSys(RecommendationSys):

    def __init__(self, indic):
        super().__init__(indic)

        self.product_names_dict: dict[str, str] | None = None
        self.rec_data: DataFrame | None = None
        self.load_custom_data()

    def load_custom_data(self):

        self.rec_data = pd.read_csv('rec_sys/rec_user_product_qty_data.csv')

        product_names = pd.read_csv("rec_sys/products_type_names.csv")
        product_names = product_names.set_index('iid')['name']
        self.product_names_dict = product_names.to_dict()

    def recommend_products_types(self, user_id) -> list[str]:

        user_preferred_products = self.rec_data[  # type: ignore
            self.rec_data['uid'] == user_id][['iid', 'qty']]  # type: ignore

        recommendations_ids = self.recommend_items(
            user_prev_data=user_preferred_products,
            print_results=False,
        )[0]

        return [self.product_names_dict[iid]  # type: ignore
                for iid in recommendations_ids]
