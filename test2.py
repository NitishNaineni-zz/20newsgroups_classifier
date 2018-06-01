from flask import Flask, render_template,request
from sklearn.datasets import fetch_20newsgroups



app = Flask(__name__)

@app.route('/')
def student():
   return render_template('student.html')






@app.route('/result',methods = ['POST'])
def result():
   if request.method == 'POST':

      str1 = request.form['text1']
      categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

      twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
      from sklearn.feature_extraction.text import CountVectorizer
      count_vect = CountVectorizer()
      X_train_counts = count_vect.fit_transform(twenty_train.data)

      from sklearn.feature_extraction.text import TfidfTransformer
      tfidf_transformer = TfidfTransformer()
      X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

      from sklearn.naive_bayes import MultinomialNB
      clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
      docs_new = list()
      docs_new.append(str1)

      X_new_counts = count_vect.transform(docs_new)
      X_new_tfidf = tfidf_transformer.transform(X_new_counts)

      predicted = clf.predict(X_new_tfidf)
      print(predicted)

      for doc, category in zip(docs_new, predicted):
          result = twenty_train.target_names[category]



      return render_template("result.html",type=result)

if __name__ == '__main__':
    app.run(debug = True)