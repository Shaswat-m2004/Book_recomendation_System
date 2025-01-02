from flask import Flask, render_template, request
import pickle
import numpy as np

popular = pickle.load(open('popular.pkl', 'rb'))
pt = pickle.load(open('pt.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))
similarity_score = pickle.load(open('similarity_score.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template(
        'index.html',
        book_name=list(popular['Book-Title'].values),
        author=list(popular['Book-Author'].values),
        image=list(popular['Image-URL-M'].values),
        votes=list(popular['num_ratings'].values),
        rating=list(popular['avg_ratings'].values),
    )


@app.route('/recommend')
def recommend_ui():
    return render_template('search.html')


@app.route('/recommend_books', methods=['POST'])
def books_recommendation():
    user_input = request.form.get('user_input')

    # Check if the book exists in the dataset
    if user_input not in pt.index:
        return "Book not found in the dataset. Please try another book."

    # Find similar items
    index = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(
        list(enumerate(similarity_score[index])),
        key=lambda x: x[1],
        reverse=True
    )[1:6]

    data = []
    for i in similar_items:
        item = []
        result = (books['Book-Title']) == pt.index[i[0]]
        temp_df = books[result]

        # Collect unique book data
        item.append(temp_df.drop_duplicates('Book-Title')['Book-Title'].values[0])
        item.append(temp_df.drop_duplicates('Book-Title')['Book-Author'].values[0])
        item.append(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values[0])
        data.append(item)
        print(data)

    # return render_template('recommendation.html', recommendations=data)
    return render_template('search.html',data = data)


if __name__ == '__main__':
    app.run(debug=True)
