import pandas as pd
import numpy as np



def train_test_split_matrix(df_matrix, test_size = 0.2, random_state = 42):
    np.random.seed(random_state)
    matrix_values = df_matrix.values
    train_matrix = matrix_values.copy()
    test_matrix = np.full(matrix_values.shape, np.nan)
    row_idx, col_idx = np.where(~np.isnan(matrix_values))
    total_rating = len(row_idx)
    num_test = int(total_rating * test_size)
    random_indices = np.random.choice(total_rating, num_test, replace = False)
    test_rows = row_idx[random_indices]
    test_cols = col_idx[random_indices]
    test_matrix[test_rows, test_cols] = train_matrix[test_rows, test_cols]
    train_matrix[test_rows, test_cols] = np.nan
    train_df = pd.DataFrame(train_matrix, index = df_matrix.index, columns = df_matrix.columns)
    test_df = pd.DataFrame(test_matrix, index = df_matrix.index, columns = df_matrix.columns)
    return train_df, test_df


def get_top_n_recommendations(user_id, n = 5):
    user_predicted_ratings = recommendations_knn[user_id]
    user_original_ratings = user_item_matrix[user_id]
    unseen_movies = user_original_ratings.isna()
    unseen_prediction = user_predicted_ratings[unseen_movies]
    top_movies = unseen_prediction.sort_values(ascending = False).head(n)
    for movie_id, predicted_rating in top_movies.items():
        movie_title = movie_dict.get(int(movie_id), "khong tim thay phim")
        print(f"{movie_title}(ID: {movie_id})")
        print(f"diem du doan: {predicted_rating:.2f} sao")

def calculate_rmse(actual_df, predicted_df):
    actual = actual_df.values if isinstance(actual_df, pd.DataFrame) else actual_df
    predicted = predicted_df.values if isinstance(predicted_df, pd.DataFrame) else predicted_df
    mask = ~np.isnan(actual)
    actual_rating = actual[mask]
    predicted_rating = predicted[mask]
    squared_errors = (actual_rating - predicted_rating)**2
    mean_squared_error = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_error)
    return rmse

rating_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
df_ratings = pd.read_csv(r'C:\Users\Lenovo\PycharmProjects\NhapMon\data\ml-100k\u.data', sep = '\t', names = rating_cols)
# doc file u.item de su dung cho viec dung association rule de giai thich
item_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
    'unknown', 'Action', 'Adventure', 'Animation', 'Childrens',
    'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
    'Sci-Fi', 'Thriller', 'War', 'Western']
df_items = pd.read_csv(r'C:\Users\Lenovo\PycharmProjects\NhapMon\data\ml-100k\u.item', sep = '|', names = item_cols, encoding = 'latin-1')
# data pre processing de huan luyen mo hinh matrix factorization
movie_dict = dict(zip(df_items['movie_id'].astype(int), df_items['title']))
df_items = df_items.drop(['release_date', 'video_release_date', 'imdb_url'], axis = 1)
df_items['movie_id'] = pd.to_numeric(df_items['movie_id'], errors = 'coerce')
df_items['unknown'] = pd.to_numeric(df_items['unknown'], errors = 'coerce')
df_items = df_items.dropna(subset=['unknown'])
df_items['movie_id'] = df_items['movie_id'].astype(int)
df_items['unknown'] = df_items['unknown'].astype(int)
df_items_cleaned = df_items[df_items['unknown'] == 0]
df_items_cleaned = df_items_cleaned.drop('unknown', axis = 1)
# tao ma tran user item
user_item_matrix = df_ratings.pivot_table(index = 'movie_id', columns = 'user_id', values = 'rating')
train_df, test_df = train_test_split_matrix(user_item_matrix, test_size = 0.2)
interact_matrix = (~np.isnan(train_df)).astype(float)
# tinh trung binh cho tung cot user
col_means = train_df.mean(axis = 0)
# tru moi cot cho gia tri trung binh
user_item_matrix_normalize = train_df.sub(col_means, axis = 1)
# cho cac gia tri NaN bang 0
user_item_matrix_normalize = user_item_matrix_normalize.fillna(0)
# tim ma tran similarity giua cac user
R = user_item_matrix_normalize.values
# tich vo huong cua 2 vector u
dot_product = np.dot(R.T, R)
# vector hang cac norm cua tung user
norm = np.sqrt(np.sum(np.square(R), axis = 0))
#chuyen vector hang thanh ma tran bang cac reshape thanh 1 vector hang nhan voi 1 vector cot
denominator = np.dot(norm.reshape(-1, 1), norm.reshape(1, -1))
epsilon = 1e-9
# cong voi mot so sieu nho de tranh truong hop user khong rate bat ky item nao
cosine_sim_matrix = dot_product / (denominator + epsilon)
# chuyen tu mang thanh dataframe
user_familiar_matrix = pd.DataFrame(cosine_sim_matrix, index = user_item_matrix.columns, columns = user_item_matrix_normalize.columns)
k = 100
# loc ra top k nguoi gan nhat
knn_sim_matrix = cosine_sim_matrix.copy()
for i in range(cosine_sim_matrix.shape[0]):
    to_zero = np.argsort(cosine_sim_matrix[i])[:-k]
    knn_sim_matrix[i, to_zero] = 0
numerator = np.dot(R, knn_sim_matrix)
abs_sim_matrix = np.abs(knn_sim_matrix)
denominator_predicted_rating = np.dot(interact_matrix, abs_sim_matrix)
pred_normalized = numerator / (denominator_predicted_rating + epsilon)
predicted_rating = pred_normalized + col_means.values
predicted_rating = np.clip(predicted_rating, 1.0, 5.0)
recommendations_knn = user_item_matrix.copy()
recommendations_knn[:] = np.where(np.isnan(user_item_matrix.values), predicted_rating, user_item_matrix.values)
rmse_train = calculate_rmse(train_df, predicted_rating)
rmse_test = calculate_rmse(test_df, predicted_rating)
print(f'RMSE Train: {rmse_train: .4f}')
print(f'RMSE Test: {rmse_test: .4f}')
get_top_n_recommendations(user_id = 1, n = 5)



