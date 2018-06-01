from flask import Flask, render_template,request
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.externals import joblib

import pickle
filename = 'finalized_model.sav'

clf = pickle.load(open(filename, 'r'))

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
      #filename = 'finalized_model.sav'
      #clf = pickle.load(open(filename, 'rb'))

      docs_new = list()
      docs_new.append(str1)

      print(docs_new)
      proc_article = Pipeline(['vect', CountVectorizer(), 'tfidf', TfidfTransformer()])
      proc_article._transform(docs_new)


      predicted = clf.predict(proc_article)



      for doc, category in zip(docs_new, predicted):
          result = twenty_train.target_names[category]



      return render_template("result.html",type=result)

if __name__ == '__main__':
    app.run(debug = True)