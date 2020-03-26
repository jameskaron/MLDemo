from scipy.spatial import distance
import pandas as pd
from sklearn.preprocessing import StandardScaler

features = ['Age', 'Weight', 'Salary']
dc_listings = pd.read_csv('nba.csv')
dc_listings = dc_listings[features]

dc_listings = dc_listings.dropna()

dc_listings[features] = StandardScaler().fit_transform(dc_listings[features])
normalized_listings = dc_listings
# print(dc_listings.shape)

norm_train_df = normalized_listings.copy().iloc[0:457]
norm_test_df = normalized_listings.copy().iloc[457:]
print(norm_train_df.shape)
print(norm_test_df)


def predict_price_multivariate(new_listing_value,feature_columns):
    temp_df = norm_train_df
    temp_df['distance'] = distance.cdist(temp_df[feature_columns],[new_listing_value[feature_columns]])
    temp_df = temp_df.sort_values('distance')
    knn_5 = temp_df.price.iloc[:5]
    predicted_price = knn_5.mean()
    return predicted_price


cols = ['Age', 'Weight']
norm_test_df['predicted_price'] = norm_test_df[cols].apply(predict_price_multivariate,feature_columns=cols,axis=1)
norm_test_df['squared_error'] = (norm_test_df['predicted_price'] - norm_test_df['Salary'])**(2)
mse = norm_test_df['squared_error'].mean()
rmse = mse ** (1/2)
print(rmse)