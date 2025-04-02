import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load Pickle Files Correctly
file_paths = {

    "df": "C:/Users/lenov/OneDrive/Desktop/BookRecommendationProject/df.pkl",
    "books": "C:/Users/lenov/OneDrive/Desktop/BookRecommendationProject/books.pkl",
    
}

data = {}
for key, path in file_paths.items():
    try:
        with open(path, "rb") as f:
            data[key] = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading {key}: {e}")


book_rating = data.get("df", pd.DataFrame())
books = data.get("books", pd.DataFrame())



Users = pd.read_csv('users.csv')  
with open('users.pkl', 'wb') as h:  
    pickle.dump(Users, h)



st.title("***BOOK RECOMMENDATION SYSTEM***")

# Popular Books Recommendation
  
def popular_recommend(book_rating, books, threshold=250):
    num_ratings = book_rating.groupby('Book-Title')['Book-Rating'].count().reset_index()
    num_ratings.columns = ['Book-Title', 'num_ratings']
    
    avg_ratings = book_rating.groupby('Book-Title')['Book-Rating'].mean().reset_index()
    avg_ratings.columns = ['Book-Title', 'avg_rating']

    popular_df = num_ratings.merge(avg_ratings, on='Book-Title')
    popular_df = popular_df[popular_df['num_ratings'] >= threshold].sort_values('avg_rating', ascending=False).head(20)
    
    popular_df = popular_df.merge(books, on='Book-Title', how='left').drop_duplicates('Book-Title')

    
    for _, row in popular_df.iterrows():
        book = row['Book-Title']
        image_url = row['Image-URL-M']
        book_author = row['Book-Author']
        
        st.write(book)
        st.write("Book-Author:", book_author)
        st.image(image_url)
        st.write(round(book_rating[book_rating['Book-Title'] == book]['Book-Rating'].mean(), 2))
        print("-"*50)



def content_based1(book_title):
    original_book_title = str(book_title) 
    global book_rating, popular_df, df  
    if original_book_title in book_rating['Book-Title'].values:
        count_rate = pd.DataFrame(book_rating['Book-Title'].value_counts())
        count_rate = count_rate.reset_index()
        count_rate.columns = ['Book-Title', 'count']
        rare_books = count_rate[count_rate['count'] <= 10]['Book-Title']

        common_books = book_rating[~book_rating["Book-Title"].isin(rare_books)]

        if original_book_title in rare_books.values:
            print("**A rare book, so you may try our popular books:** \n ")
            print(popular_recommend(book_rating, book, num_ratings_threshold=250))

        else:
            common_books = common_books.drop_duplicates(subset=["Book-Title"])
            common_books.reset_index(inplace=True)

            common_books["index"] = [i for i in range(common_books.shape[0])]
            common_books['Book-Title'] = common_books['Book-Title'].astype('object')
            common_books['Book-Author'] = common_books['Book-Author'].astype('object')
            common_books['Publisher'] = common_books['Publisher'].astype('object')
            common_books['Image-URL-M'] = common_books['Image-URL-M'].astype('object')

            targets = ["Book-Title", "Book-Author", "Publisher"]
            common_books["all_features"] = [" ".join(common_books[targets].iloc[i, ].values) for i in range(common_books[targets].shape[0])]

            vectorizer = CountVectorizer()
            common_booksVector = vectorizer.fit_transform(common_books["all_features"])

            similarity = cosine_similarity(common_booksVector)

            if original_book_title in common_books['Book-Title'].values:
                index = common_books[common_books["Book-Title"] == original_book_title]["index"].values[0]
                similar_books = list(enumerate(similarity[index]))
                similar_booksSorted = sorted(similar_books, key=lambda x: x[1], reverse=True)[1:6]
                r_books = []  

                for i in range(len(similar_booksSorted)):
                    book_index = similar_booksSorted[i][0]
                    book_info = common_books[common_books["index"] == book_index]
                    rec_book_title = book_info["Book-Title"].item()
                    image_url = book_info["Image-URL-M"].item()
                    book_author = book_info["Book-Author"].item()

                    r_books.append((rec_book_title, image_url, book_author))

                print(f"**Recommend Books similar to** {original_book_title} :\n")
                for book, image_url, book_author in r_books:
                    print(book)
                    print("Book-Author:", book_author)
                    print("Image URL:", image_url)

                    print(round(df[df['Book-Title'] == book]['Book-Rating'].mean(), 2))
                    print("-"*50)

# Item-Based Recommendation
def item_based_recommendation(book_title):
    if book_title not in book_rating['Book-Title'].values:
        
        return(content_based1(book_title))

    common_books = book_rating.groupby('Book-Title').filter(lambda x: len(x) > 50)

    if book_title in common_books['Book-Title'].values:
        pivot_table = common_books.pivot_table(index="User-ID", columns="Book-Title", values="Book-Rating")
        similarity_scores = pivot_table.corrwith(pivot_table[book_title]).dropna()
        recommendations = similarity_scores.sort_values(ascending=False).iloc[1:10].index.tolist()
        
        st.write(f"**Recommended Books for** {book_title}:")
        for book in recommendations:
            book_info = books[books['Book-Title'] == book]
            if not book_info.empty:
                st.write(book)
                st.write(book_info['Book-Author'].values[0])
                st.image(book_info['Image-URL-M'].values[0])
    else:
        st.write("**This book is rare. Here are some popular books instead**:")
        popular_recommend(book_rating, books)

# User Selection
selected_book = st.selectbox("Select a Book:", books["Book-Title"].unique())

if st.button("Recommend"):
    item_based_recommendation(selected_book)






df=book_rating
new_df=df[df['User-ID'].map(df['User-ID'].value_counts()) > 100]
users_matrix=new_df.pivot_table(index=["User-ID"],columns=["Book-Title"],values="Book-Rating")
users_matrix.fillna(0, inplace=True)
def user_based(user_id):

    
    users_fav = new_df[new_df["User-ID"] == user_id].sort_values(["Book-Rating"], ascending=False).head(5)


    print("\n\n")
        

    st.write("\n\n")

    try:
        index = np.where(users_matrix.index == user_id)[0][0]  
    except IndexError:
        st.write("User not found in the users matrix.")
        return

    # Calculate cosine similarity between users
    similarity = cosine_similarity(users_matrix)
    similar_users = list(enumerate(similarity[index]))  
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[1:6]  

   
    users_id = [users_matrix.index[i[0]] for i in similar_users]
    
    recommend_books = []
    
 
    for user in users_id:
       
        similar_user_books = new_df[new_df["User-ID"] == user]
        sim_books = similar_user_books.loc[~similar_user_books["Book-Title"].isin(users_fav["Book-Title"]), :]


        sim_books = sim_books.sort_values(["Book-Rating"], ascending=False).head(5)
        
        for _, row in sim_books.iterrows():
            book_title = row["Book-Title"]
            image_url = row["Image-URL-M"] if pd.notna(row["Image-URL-M"]) else "DefaultImageURL"
            book_author = row["Book-Author"]

            recommend_books.append((book_title, image_url, book_author))

    st.write("**Recommended for you:** \n")

    # Display the recommended books
    for book, image_url, book_author in recommend_books:
        avg_rating = round(df[df['Book-Title'] == book]['Book-Rating'].mean(), 2)
        st.write(book)
        st.write(book_author)
        st.image(image_url)
        st.write(avg_rating) 
        
       

# User selection from Streamlit interface
selected_user = st.selectbox("Select Your User ID", new_df["User-ID"])
if st.button("MY BOOKS"):
    user_based(selected_user)
    





