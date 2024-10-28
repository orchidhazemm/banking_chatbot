from geopy.distance import distance
from pandas import read_csv


def get_user_location(user_id):
    return (30.1388403, 31.2396769)


def get_nearby_banque_branches(
    location: tuple[float, float] = (30.1388403, 31.2396769)
) -> str:

    branches_info = read_csv('branch_locations\\branches_locations.csv')

    branches_info['distance'] = branches_info.apply(
        lambda row: round(distance(location, eval(row['location'])).km, 2),
        axis=1)

    nearby_branches = branches_info.nsmallest(3, 'distance')

    str_res = '### Nearest Banque Misr Branches to You:'
    for i, branch in enumerate(nearby_branches.values):
        str_res += \
            f"""
        \r##### Branch Name: {branch[0]}
        \r- **Address:** {branch[2]}
        \r- **Google Maps:** [View on Google Maps]({branch[3]})
        \r- **Distance:** {branch[-1]} km
        """

    return str_res
